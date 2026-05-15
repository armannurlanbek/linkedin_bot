"""Simple shared-password authentication with a sliding 3-day HMAC cookie."""

import base64
import hashlib
import hmac
import time

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from app.config import settings

router = APIRouter(prefix="/api/auth")

COOKIE_NAME = "lb_auth"
MAX_AGE_SECONDS = 3 * 24 * 60 * 60  # 3 days


def _sign(ts: int) -> str:
    """HMAC-SHA256 of the timestamp string."""
    msg = str(ts).encode()
    sig = hmac.new(settings.cookie_secret.encode(), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode()


def _make_value(ts: int) -> str:
    return f"{ts}.{_sign(ts)}"


def _verify(value: str) -> bool:
    """Return True if the cookie is valid and not older than MAX_AGE_SECONDS."""
    try:
        ts_str, sig = value.rsplit(".", 1)
        ts = int(ts_str)
        if not hmac.compare_digest(sig, _sign(ts)):
            return False
        if time.time() - ts > MAX_AGE_SECONDS:
            return False
        return True
    except Exception:
        return False


def set_auth_cookie(response: Response) -> None:
    ts = int(time.time())
    response.set_cookie(
        COOKIE_NAME,
        _make_value(ts),
        max_age=MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


def is_authenticated(request: Request) -> bool:
    if not settings.app_password:
        return True  # auth disabled when no password set
    value = request.cookies.get(COOKIE_NAME, "")
    return _verify(value)


class LoginBody(BaseModel):
    password: str


@router.post("/login")
def login(body: LoginBody, response: Response):
    if not settings.app_password:
        return {"ok": True}
    if not hmac.compare_digest(body.password, settings.app_password):
        from fastapi import HTTPException
        raise HTTPException(401, "Wrong password")
    set_auth_cookie(response)
    return {"ok": True}


@router.post("/logout")
def logout(response: Response):
    clear_auth_cookie(response)
    return {"ok": True}


@router.get("/check")
def check(request: Request, response: Response):
    ok = is_authenticated(request)
    if ok and settings.app_password:
        set_auth_cookie(response)  # refresh sliding window
    return {"authenticated": ok}
