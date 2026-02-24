import urllib.parse
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from src.schema import Say, User
from src.secret import random_secret
from src.oauth_providers import get_oauth_provider
from src.auth.cookie import (
    set_oauth_state_cookie,
    validate_oauth_state_cookie,
    clear_oauth_state_cookie,
    set_auth_cookie,
    clear_auth_cookie,
)
from src.auth.jwt import verify_token
from src.auth import reusable_oauth
import src.vars as var
from typing import Annotated
from aws_lambda_powertools import Logger
import boto3
import json
from libs.cybozu import get_kintone_record
from mangum import Mangum
import urllib
import requests

app = FastAPI(
    title="Kajima App API",
    version="0.1.0",
    root_path=f"/{var.STAGE}" if var.STAGE else "/",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)

logger = Logger(service="kajima-app-api")

endpoint_url = None
if var.STAGE == "local":
    endpoint_url = "https://localhost.localstack.cloud:4566"
lambda_client = boto3.client("lambda", endpoint_url=endpoint_url)


def get_user_facing_url(request: Request) -> str:
    base_url = str(request.url.replace(query="", fragment=""))
    path = str(request.url.path)

    if var.STAGE and not path.startswith(f"/{var.STAGE}/"):
        parsed_url = urllib.parse.urlparse(base_url)

        # Get hostname
        host = parsed_url.netloc

        # Build new URL with correct stage prefix
        url = f"{parsed_url.scheme}://{host}/{var.STAGE}{path}"
        logger.info(f"Modified URL with stage prefix: {url}")
        return url

    return base_url


def _get_oauth_redirect_error(request: Request, error: str) -> Response:
    """Get the redirect response for an OAuth error."""
    params = urllib.parse.urlencode(
        {
            "error": error,
        }
    )
    response = RedirectResponse(url=str(request.url_for("login")) + "?" + params)
    return response


async def _authenticate_user(
    request: Request,
    access_token,
    user: User | None = None,
    redirect_to_callback: bool = False,
) -> Response:
    """Authenticate a user and return the response."""

    if not user:
        raise HTTPException(
            status_code=401,
            detail="credentialssignin",
        )

    response = _get_auth_response(access_token, redirect_to_callback)

    set_auth_cookie(request, response, access_token)

    return response


def _get_auth_response(token: str, redirect_to_callback: bool) -> Response:
    """Get the authentication response."""
    response_dict = {"success": True, "token": token}
    if redirect_to_callback:
        return RedirectResponse(
            url="/docs" if not var.STAGE else f"/{var.STAGE}/docs",
            status_code=302,
            headers={"Authorization": f"Bearer {token}"},
        )

    return JSONResponse(response_dict)


@app.get("/auth/oauth/{provider_id}", tags=["auth"])
async def oauth_login(provider_id: str, request: Request):
    """Redirect user to the OAuth provider login page."""
    provider = get_oauth_provider(provider_id)

    if not provider:
        raise HTTPException(
            status_code=404,
            detail=f"OAuth provider {provider_id} not found",
        )

    random = random_secret(32)
    params = urllib.parse.urlencode(
        {
            "client_id": provider.client_id,
            "redirect_uri": f"{get_user_facing_url(request)}/callback",
            "state": random,
            **provider.authorize_params,
        }
    )
    response = RedirectResponse(url=f"{provider.authorize_url}?{params}")

    set_oauth_state_cookie(response, random)

    return response


@app.get("/auth/oauth/{provider_id}/callback", tags=["auth"])
async def oauth_callback(
    provider_id: str,
    request: Request,
    error: str | None = None,
    code: str | None = None,
    state: str | None = None,
):
    """Handle the OAuth provider callback."""
    provider = get_oauth_provider(provider_id)

    if not provider:
        raise HTTPException(
            status_code=404,
            detail=f"OAuth provider {provider_id} not found",
        )

    if error:
        return _get_oauth_redirect_error(request, error)

    if not code or not state:
        raise HTTPException(
            status_code=400,
            detail="Missing code or state in the callback",
        )

    # Validate the state cookie
    try:
        logger.info("oauth state cookie: %s", request.cookies.get("oauth_state"))
        validate_oauth_state_cookie(request, state)
    except Exception as e:
        logger.exception(f"Unable to validate oauth state: {e}")
        raise HTTPException(status_code=401, detail=str(e))

    # Extract the base URL without the callback part
    redirect_uri = f"{get_user_facing_url(request)}"

    token = await provider.get_token(code, redirect_uri)
    (raw_user_data, default_user) = await provider.get_user_info(token)

    response = await _authenticate_user(
        request,
        token,
        default_user,
        True,
    )

    clear_oauth_state_cookie(response)
    return response


@app.get("/login", tags=["auth"])
async def login(request: Request, response: Response):
    """Handle the login request."""
    error = request.query_params.get("error")
    if error:
        return _get_oauth_redirect_error(request, error)

    # Check if the user is already authenticated
    token = request.cookies.get("access_token")
    base_url = request.base_url
    logger.info(f"Base URL: {base_url}")

    redirect_url = None
    headers = None
    if token:
        try:
            # Verify the token
            _ = verify_token(token)
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            redirect_url = f"{base_url}auth/oauth/aws-cognito"
            return RedirectResponse(url=redirect_url)

        redirect_url = f"{base_url}docs"
        headers = {
            "Authorization": f"Bearer {token}",
        }

    else:
        # Redirect to the OAuth provider login page
        redirect_url = f"{base_url}auth/oauth/aws-cognito"

    return RedirectResponse(url=redirect_url, headers=headers)


@app.get("/get_access_token", tags=["auth"])
def get_access_token(request: Request, token: str = Depends(reusable_oauth)):
    """Get the access token from the request."""
    try:
        # Verify the token
        payload = verify_token(token)
        return JSONResponse(status_code=200, content={"access_token": token})
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return JSONResponse(status_code=401, content={"message": "Unauthorized"})


@app.get("/logout", tags=["auth"])
async def logout(request: Request, response: Response):
    clear_auth_cookie(request, response)

    return {"success": True}


@app.get(
    "/health",
    responses={
        200: {
            "content": {"application/json": {"example": {"message": "I'm active now"}}}
        }
    },
    summary="Health check for GET method",
    description="This is a health check API for GET Method. This API will always return message I'm active now",
)
def health_check():
    logger.info("Request for GET health check")
    return JSONResponse(status_code=200, content={"message": "I'm active now"})


@app.post(
    "/return_same",
    responses={
        200: {"content": {"application/json": {"example": {"message": "Hello"}}}}
    },
    summary="Health check for POST method",
    description="This is a health check API for POST Method. This API will return the same message as request.",
)
def return_same(say: Say):
    logger.info("Request for POST health check")
    return JSONResponse(status_code=200, content={"message": say.message})


@app.post(
    "/trigger_classification",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "user_id": "User ID",
                        "record_id": "Record ID",
                        "message": "Classification triggered successfully",
                    }
                }
            }
        },
        404: {
            "content": {
                "application/json": {
                    "example": {
                        "message": "Record not found",
                    }
                }
            }
        },
        403: {
            "content": {
                "application/json": {
                    "example": {
                        "message": "Permission denied. Please check your Kintone API token.",
                    }
                }
            }
        },
        401: {
            "content": {
                "application/json": {
                    "example": {
                        "message": "Unauthorized. Please check your access token.",
                    }
                }
            }
        },
    },
    summary="Trigger classification for image included in record id",
    description="This API will trigger the classification for image included in record id. This API will return the same message as request.",
)
async def trigger_classification(
    request: Request,
    user_id: Annotated[str, Form()],
    record_id: Annotated[str, Form()],
    token: str = Depends(reusable_oauth),
):
    logger.info(
        f"Receive request from user {user_id} with record {record_id}",
        extra={
            "user_id": user_id,
            "record_id": record_id,
        },
    )
    if token:
        try:
            _ = verify_token(token)
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return HTTPException(status_code=401, detail=str(e))
    else:
        return HTTPException(status_code=401, detail="Unauthorized. Please login.")
    try:
        data = get_kintone_record(record_id)
        # Call the Lambda function asynchronously
        payload = {
            "body": {
                "user_id": user_id,
                "record_id": record_id,
                "data": data,
            }
        }
        response = lambda_client.invoke(
            FunctionName=var.PROCESSOR_LAMBDA_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps(payload),
        )
        logger.info(f"Lambda invoke response status code: {response['StatusCode']}")
        return JSONResponse(
            status_code=200,
            content={
                "user_id": user_id,
                "record_id": record_id,
                "message": "Classification triggered successfully",
            },
        )
    except requests.RequestException as e:
        logger.error(
            f"Error during Kintone record check {record_id}: \n{str(e)}\nHeader:{e.response.headers}"
        )

        if e.response.status_code == 404:
            return JSONResponse(
                status_code=404,
                content={"message": "Record not found"},
            )
        elif e.response.status_code == 403:
            return JSONResponse(
                status_code=403,
                content={
                    "message": "Permission denied. Please check your Kintone API token."
                },
            )
    except Exception as ex:
        logger.error(f"Error during trigger classification {record_id}: {str(ex)}")
        return JSONResponse(
            status_code=500, content={"message": "Internal server error"}
        )


original_mangum = Mangum(
    app,
    lifespan="off",
    api_gateway_base_path=f"/{var.STAGE}" if var.STAGE else "/",
)


def logging_handler(event, context):
    logger.info(f"Lambda received event: {event}")
    try:
        response = original_mangum(event, context)
        logger.info(f"Mangum returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in Mangum handler: {str(e)}")
        raise


handler = logging_handler
