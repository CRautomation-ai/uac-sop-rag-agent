import os
import hmac
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.getenv("JWT_SECRET", "")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")

security = HTTPBearer(auto_error=False)


def verify_password(plain: str) -> bool:
    """Constant-time comparison against ACCESS_PASSWORD."""
    if not ACCESS_PASSWORD:
        return False
    return hmac.compare_digest(plain.encode("utf-8"), ACCESS_PASSWORD.encode("utf-8"))


def create_token() -> str:
    """Create a JWT with sub, signed with JWT_SECRET. No expiration."""
    payload = {"sub": "authenticated"}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> bool:
    """Decode and verify JWT signature. Returns True if valid (no expiration check)."""
    if not token or not JWT_SECRET:
        return False
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return True
    except jwt.PyJWTError:
        return False


def get_current_token(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> str:
    """FastAPI dependency: require valid Bearer token or raise 401."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing or invalid authorization")
    token = credentials.credentials
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return token
