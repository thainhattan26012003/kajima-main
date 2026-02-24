from typing import Optional

import httpx
import os

from fastapi import HTTPException
from src.schema import User


class OAuthProvider:
    id: str
    env: list[str]
    client_id: str
    client_secret: str
    authorize_url: str
    authorize_params: dict[str, str]
    default_prompt: Optional[str] = None

    def is_configured(self) -> bool:
        return all([os.environ.get(env) for env in self.env])

    async def get_token(self) -> str:
        raise NotImplementedError("get_token method must be implemented in subclasses")

    async def get_user_info(self) -> dict:
        raise NotImplementedError(
            "get_user_info method must be implemented in subclasses"
        )

    def get_env_prefix(self) -> str:
        """Return env prefix, like COGNITO_ID"""
        return self.id.replace("-", "_").upper()

    def get_prompt(self) -> str:
        """Return OAuth prompt param."""
        if prompt := os.environ.get(f"OAUTH_{self.get_env_prefix()}_PROMPT"):
            return prompt

        if prompt := os.environ.get("OAUTH_PROMPT"):
            return prompt

        return self.default_prompt


class AWSCognitoOAuthProvider(OAuthProvider):
    id = "aws-cognito"
    env = [
        "OAUTH_COGNITO_CLIENT_ID",
        "OAUTH_COGNITO_CLIENT_SECRET",
        "OAUTH_COGNITO_DOMAIN",
        "AWS_REGION",
    ]
    authorize_url = f"https://{os.environ.get('OAUTH_COGNITO_DOMAIN')}/login"
    token_url = f"https://{os.environ.get('OAUTH_COGNITO_DOMAIN')}/oauth2/token"
    revoke_url = f"https://{os.environ.get('OAUTH_COGNITO_DOMAIN')}/oauth2/revoke"
    def __init__(self):
        self.client_id = os.environ.get("OAUTH_COGNITO_CLIENT_ID")
        self.client_secret = os.environ.get("OAUTH_COGNITO_CLIENT_SECRET")
        self.authorize_params = {
            "response_type": "code",
            "client_id": self.client_id,
            "scope": "openid profile email",
        }
        if prompt := self.get_prompt():
            self.authorize_params["prompt"] = prompt

    async def get_token(self, code: str, url: str) -> str:
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data=payload)
            response.raise_for_status()
            json = response.json()

            token = json.get("access_token")
            if not token:
                raise HTTPException(
                    status_code=400, detail="Failed to get the access token"
                )
            return token

    async def get_user_info(self, token: str):
        user_info_url = (
            f"https://{os.environ.get('OAUTH_COGNITO_DOMAIN')}/oauth2/userInfo"
        )

        async with httpx.AsyncClient() as client:
            response = await client.get(
                user_info_url, headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()

            cognito_user = response.json()

            user = User(
                identifier=cognito_user["email"],
                metadata={
                    "image": cognito_user.get("picture", ""),
                    "provider": "aws-cognito",
                },
            )
            return (cognito_user, user)
    async def revoke_token(self, token: str):
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "token": token,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.revoke_url, data=payload)
            response.raise_for_status()
            return response.json()

providers = [
    AWSCognitoOAuthProvider(),
]


def get_oauth_provider(provider: str) -> Optional[OAuthProvider]:
    for p in providers:
        if p.id == provider:
            return p
    return None
