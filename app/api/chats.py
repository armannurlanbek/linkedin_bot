import mimetypes

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import httpx

from app.config import settings
from app.db import get_db, Chat, Message
from app.services.agent import run_agent

router = APIRouter(prefix="/api")


class ChatCreate(BaseModel):
    title: str = "New chat"


class MessageBody(BaseModel):
    text: str


@router.post("/chats", status_code=201)
def create_chat(body: ChatCreate = ChatCreate(), db: Session = Depends(get_db)):
    chat = Chat(title=body.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"id": chat.id, "title": chat.title, "created_at": chat.created_at}


@router.get("/chats")
def list_chats(db: Session = Depends(get_db)):
    chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
    return [{"id": c.id, "title": c.title, "thumbnail_url": c.thumbnail_url, "updated_at": c.updated_at} for c in chats]


@router.get("/chats/{chat_id}")
def get_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )
    return {
        "id": chat.id,
        "title": chat.title,
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at}
            for m in messages
        ],
    }


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    db.delete(chat)
    db.commit()
    return Response(status_code=204)


@router.delete("/chats/{chat_id}/messages/since/{message_id}", status_code=204)
def truncate_messages(chat_id: int, message_id: int, db: Session = Depends(get_db)):
    """Delete message_id and all subsequent messages in the chat (for edit-and-resend)."""
    msg = db.query(Message).filter(Message.id == message_id, Message.chat_id == chat_id).first()
    if not msg:
        raise HTTPException(404, "Message not found")
    (
        db.query(Message)
        .filter(Message.chat_id == chat_id, Message.id >= message_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    return Response(status_code=204)


@router.get("/proxy-image")
def proxy_image(url: str = Query(...), download: bool = False):
    """Proxy a remote image. download=false → view in browser, download=true → save file."""
    if not url.startswith("http"):
        raise HTTPException(400, "Invalid URL")
    try:
        req_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": url,
        }
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers=req_headers)
        if not resp.is_success:
            raise HTTPException(502, "Remote image not available")
        raw_ct = resp.headers.get("content-type", "").split(";")[0].strip()
        data = resp.content

        # Detect image type from magic bytes — content-type headers from CDNs are unreliable
        if data[:3] == b'\xff\xd8\xff':
            content_type, ext = "image/jpeg", ".jpg"
        elif data[:4] == b'\x89PNG':
            content_type, ext = "image/png", ".png"
        elif data[:6] in (b'GIF87a', b'GIF89a'):
            content_type, ext = "image/gif", ".gif"
        elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            content_type, ext = "image/webp", ".webp"
        elif len(data) >= 12 and data[4:8] == b'ftyp' and b'avif' in data[8:16]:
            content_type, ext = "image/avif", ".avif"
        elif raw_ct == "image/svg+xml":
            content_type, ext = "image/svg+xml", ".svg"
        elif raw_ct.startswith("image/"):
            content_type = raw_ct
            ext = mimetypes.guess_extension(content_type) or ".jpg"
            if ext == ".jpe":
                ext = ".jpg"
        else:
            raise HTTPException(404, "Not an image")

        filename = url.split("/")[-1].split("?")[0] or ""
        if not any(filename.lower().endswith(e) for e in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif")):
            filename = (filename.rsplit(".", 1)[0] if "." in filename else filename or "image") + ext
        resp_headers = {}
        if download:
            resp_headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(content=data, media_type=content_type, headers=resp_headers)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(502, "Failed to proxy image")


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio via OpenAI Whisper. Accepts webm/mp4/wav/ogg."""
    from openai import OpenAI
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        fname = file.filename or "audio.webm"
        mime = file.content_type or "audio/webm"
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=(fname, audio_bytes, mime),
        )
        return {"text": resp.text}
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")


@router.post("/chats/{chat_id}/messages")
def send_message(chat_id: int, body: MessageBody, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    return StreamingResponse(
        run_agent(chat_id, body.text, db),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
