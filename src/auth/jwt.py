from jose import jwt
from fastapi import HTTPException
import src.vars as var
import requests
import time
from typing import Dict, Any


class JWKSCache:
    """
    Class to handle caching of JSON Web Key Sets (JWKS) from Cognito.
    """

    def __init__(self, ttl: int = 3600):
        self._cache = {}
        self._cache_time = 0
        self._cache_ttl = ttl  # Default 1 hour

    def get_jwks(self) -> Dict[str, Any]:
        """
        Fetches the JSON Web Key Set from Cognito with caching.
        """
        current_time = time.time()

        # Return cached keys if they're still valid
        if self._cache and current_time < self._cache_time + self._cache_ttl:
            return self._cache

        # Construct the JWKS URL from Cognito details
        jwks_url = f"https://cognito-idp.{var.OAUTH_COGNITO_AWS_REGION}.amazonaws.com/{var.OAUTH_COGNITO_USER_POOL_ID}/.well-known/jwks.json"

        try:
            response = requests.get(jwks_url)
            response.raise_for_status()
            self._cache = response.json()
            self._cache_time = current_time
            return self._cache
        except requests.RequestException as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch JWKS: {str(e)}"
            )


jwks_cache = JWKSCache()


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verifies a JWT token issued by AWS Cognito.
    """
    try:
        # Get the header from the token
        headers = jwt.get_unverified_header(token)

        kid = headers.get("kid")
        if not kid:
            raise HTTPException(status_code=401, detail="No KID found in token header")

        # Find the matching key in the JWKS
        jwks = jwks_cache.get_jwks()
        key = None
        for jwk in jwks.get("keys", []):
            if jwk.get("kid") == kid:
                key = jwk
                break

        if not key:
            raise HTTPException(
                status_code=401, detail="No matching key found for token"
            )

        # Construct the Cognito issuer URL
        issuer = f"https://cognito-idp.{var.OAUTH_COGNITO_AWS_REGION}.amazonaws.com/{var.OAUTH_COGNITO_USER_POOL_ID}"

        # Verify the token
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=var.OAUTH_COGNITO_CLIENT_ID,
            issuer=issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_aud": True,
                "verify_iss": True,
            },
        )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTClaimsError:
        raise HTTPException(status_code=401, detail="Invalid token claims")
    except jwt.JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Failed to verify token: {str(e)}")
