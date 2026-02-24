from src.auth.cookie import OAuth2PasswordBearerWithCookie

reusable_oauth = OAuth2PasswordBearerWithCookie(tokenUrl="/login", auto_error=False)
