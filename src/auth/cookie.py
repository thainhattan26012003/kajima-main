from fastapi import Request, Response
from fastapi.security import OAuth2
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.exceptions import HTTPException

""" Module level cookie settings. """
_cookie_samesite = "lax"
_cookie_secure = _cookie_samesite == "none"

_state_cookie_lifetime = 3 * 60  # 3m
_auth_cookie_name = "access_token"
_state_cookie_name = "oauth_state"


def set_auth_cookie(request: Request, response: Response, token: str):
    """
    Helper function to set the authentication cookie with secure parameters
    and remove any leftover chunks from a previously larger token.
    """

    _chunk_size = 3000

    existing_cookies = {
        k for k in request.cookies.keys() if k.startswith(_auth_cookie_name)
    }

    if len(token) > _chunk_size:
        chunks = [token[i : i + _chunk_size] for i in range(0, len(token), _chunk_size)]

        for i, chunk in enumerate(chunks):
            k = f"{_auth_cookie_name}_{i}"

            response.set_cookie(
                key=k,
                value=chunk,
                httponly=True,
                secure=_cookie_secure,
                samesite=_cookie_samesite,
                max_age=86400,
            )

            existing_cookies.discard(k)
    else:
        # Default (shorter cookies)
        response.set_cookie(
            key=_auth_cookie_name,
            value=token,
            httponly=True,
            secure=_cookie_secure,
            samesite=_cookie_samesite,
            max_age=86400,
        )

        existing_cookies.discard(_auth_cookie_name)

    # Delete remaining prior cookies/cookie chunks
    for k in existing_cookies:
        response.delete_cookie(
            key=k, path="/", secure=_cookie_secure, samesite=_cookie_samesite
        )


def _get_chunk_cookie(cookies: dict[str, str]) -> str | None:
    # Gather all chunk_i cookies and sorted by index
    chunk_parts = []

    i = 0
    while True:
        k = f"{_auth_cookie_name}_{i}"
        if k not in cookies:
            break
        chunk_parts.append(cookies[k])
        i += 1
    joined = "".join(chunk_parts)
    return joined if joined != "" else None


def get_token_from_cookies(cookies: dict[str, str]) -> str:
    """
    Read all chunk cookies and reconstruct the token
    """
    if value := cookies.get(_auth_cookie_name):
        return value

    return _get_chunk_cookie(cookies)


def clear_auth_cookie(request: Request, response: Response):
    """
    Helper function to clear the authentication cookie
    """

    existing_cookies = {
        k for k in request.cookies.keys() if k.startswith(_auth_cookie_name)
    }

    for k in existing_cookies:
        response.delete_cookie(
            key=k, path="/", secure=_cookie_secure, samesite=_cookie_samesite
        )


def set_oauth_state_cookie(response: Response, token: str):
    response.set_cookie(
        _state_cookie_name,
        token,
        httponly=True,
        samesite=_cookie_samesite,
        secure=_cookie_secure,
        max_age=_state_cookie_lifetime,
    )


def validate_oauth_state_cookie(request: Request, state: str):
    """Check the state from the oauth provider against the browser cookie."""

    oauth_state = request.cookies.get(_state_cookie_name)

    if oauth_state != state:
        raise Exception("oauth state does not correspond")


def clear_oauth_state_cookie(response: Response):
    """Oauth complete, delete state token."""
    response.delete_cookie(_state_cookie_name)  # Do we set path here?


class OAuth2PasswordBearerWithCookie(OAuth2):
    """
    OAuth2 password flow with cookie support with fallback to Authorization header
    """

    def __init__(
        self, tokenUrl: str, scheme_name: str | None = None, auto_error: bool = True
    ):
        super().__init__(
            flows={"password": {"tokenUrl": tokenUrl, "scopes": {}}},
            scheme_name=scheme_name,
            auto_error=auto_error,
        )
        self.tokenUrl = tokenUrl

    async def __call__(self, request: Request) -> str | None:
        token = get_token_from_cookies(request.cookies)

        if not token:
            authorization = request.headers.get("Authorization")
            if authorization:
                scheme, token = get_authorization_scheme_param(authorization)
                if scheme.lower() != "bearer":
                    if self.auto_error:
                        raise HTTPException(
                            status_code=401,
                            detail="Invalid authentication credentials",
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                    else:
                        return None
            else:
                if self.auto_error:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid authentication credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                else:
                    return None
        return token
