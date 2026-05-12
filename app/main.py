from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.db import init_db
from app.api.chats import router as chats_router

app = FastAPI(title="LinkedIn Post Generator", version="0.2.0")


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(chats_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def index():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
