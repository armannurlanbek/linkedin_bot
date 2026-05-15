from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api.auth import is_authenticated, set_auth_cookie, router as auth_router
from app.api.chats import router as chats_router
from app.api.library import router as library_router
from app.config import settings
from app.db import init_db, SessionLocal

app = FastAPI(title="LinkedIn Post Generator", version="0.3.0")


@app.on_event("startup")
def on_startup():
    init_db()


# ── Auth middleware ────────────────────────────────────────────────────────────
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path

    # Unprotected paths — always pass through
    if (
        path.startswith("/api/auth/")
        or path == "/health"
        or path == "/"
        or path.startswith("/static/")
    ):
        return await call_next(request)

    # Protected API paths
    if path.startswith("/api/"):
        if not is_authenticated(request):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        response = await call_next(request)
        if settings.app_password:
            set_auth_cookie(response)  # slide the cookie window
        return response

    return await call_next(request)


# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(chats_router)
app.include_router(library_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)
    finally:
        db.close()
