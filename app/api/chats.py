import base64
import io
from datetime import datetime, timezone
import json
import mimetypes
import queue
import threading

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import httpx

from app.config import settings
from app.db import get_db, Chat, LibraryPost, Message, SessionLocal
from app.services.agent import run_agent
from app.services.chat_meta import apply_chat_meta, is_post_message

router = APIRouter(prefix="/api")

_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_PDF_BYTES   = 10 * 1024 * 1024
_MAX_TEXT_BYTES  = 1 * 1024 * 1024


class ChatCreate(BaseModel):
    title: str = "New chat"


class MessageBody(BaseModel):
    text: str
    attachments: list[dict] = []


class ChatPatch(BaseModel):
    title: str | None = None
    thumbnail_url: str | None = None
    posted: bool | None = None
    posted_message_id: int | None = None


def _message_text(content) -> str:
    """Join the text blocks of a stored message."""
    if not isinstance(content, list):
        return content if isinstance(content, str) else ""
    return "\n".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def _serialize_chat(chat: Chat, posted_preview: str | None = None) -> dict:
    return {
        "id": chat.id,
        "title": chat.title,
        "title_source": chat.title_source,
        "thumbnail_url": chat.thumbnail_url,
        "thumbnail_source": chat.thumbnail_source,
        "posted_at": chat.posted_at,
        "posted_message_id": chat.posted_message_id,
        "posted_preview": posted_preview,
        "updated_at": chat.updated_at,
    }


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
    posted_ids = [c.posted_message_id for c in chats if c.posted_message_id]
    previews: dict[int, str] = {}
    if posted_ids:
        for m in db.query(Message).filter(Message.id.in_(posted_ids)).all():
            previews[m.id] = _message_text(m.content)[:120]
    return [
        _serialize_chat(c, previews.get(c.posted_message_id) if c.posted_message_id else None)
        for c in chats
    ]


@router.get("/chats/{chat_id}")
def get_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at, Message.id)
        .all()
    )
    return {
        **_serialize_chat(chat),
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at}
            for m in messages
        ],
    }


def _last_post_message_id(db: Session, chat_id: int) -> int | None:
    """The most recent assistant message that is a finished post."""
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id, Message.role == "assistant")
        .order_by(Message.created_at, Message.id)
        .all()
    )
    for m in reversed(messages):
        if is_post_message(m.content):
            return m.id
    return None


def _archive_posted(db: Session, chat_id: int, message_id: int | None) -> None:
    """File a published post in the style archive the agent writes from.

    Best-effort: the post is already marked as published either way, and the
    Library keeps its manual archive button. An embedding failure must not turn
    into a failed request.
    """
    if not message_id:
        return
    try:
        from app.api.library import promote_item

        item = db.query(LibraryPost).filter(LibraryPost.message_id == message_id).first()
        if item is None:
            msg = db.query(Message).filter(Message.id == message_id).first()
            body = _message_text(msg.content) if msg else ""
            if not body:
                return
            item = LibraryPost(text=body, chat_id=chat_id, message_id=message_id)
            db.add(item)
            db.commit()
            db.refresh(item)
        promote_item(db, item)
    except Exception:
        db.rollback()


@router.patch("/chats/{chat_id}")
def update_chat(chat_id: int, body: ChatPatch, db: Session = Depends(get_db)):
    """Rename a chat, set its cover image, or mark it as posted."""
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")

    updates: dict = {}
    archive_message_id: int | None = None

    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise HTTPException(400, "Title cannot be empty")
        # "manual" means no automatic pass ever renames this chat again.
        updates["title"] = title[:120]
        updates["title_source"] = "manual"

    if body.thumbnail_url is not None:
        if not body.thumbnail_url.startswith("http"):
            raise HTTPException(400, "thumbnail_url must be an http(s) URL")
        updates["thumbnail_url"] = body.thumbnail_url
        updates["thumbnail_source"] = "manual"

    if body.posted is not None:
        if body.posted:
            message_id = body.posted_message_id or _last_post_message_id(db, chat_id)
            updates["posted_message_id"] = message_id
            if chat.posted_at is None:
                updates["posted_at"] = datetime.now(timezone.utc)
            archive_message_id = message_id
        else:
            updates["posted_at"] = None
            updates["posted_message_id"] = None

    if updates:
        # Goes through apply_chat_meta so `updated_at` — which orders the
        # sidebar — is not bumped by a rename or a cover change.
        apply_chat_meta(db, chat_id, **updates)

    if archive_message_id:
        _archive_posted(db, chat_id, archive_message_id)

    db.refresh(chat)
    preview = None
    if chat.posted_message_id:
        msg = db.query(Message).filter(Message.id == chat.posted_message_id).first()
        if msg:
            preview = _message_text(msg.content)[:120]
    return _serialize_chat(chat, preview)


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
    first = (
        db.query(Message.id)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at, Message.id)
        .first()
    )
    (
        db.query(Message)
        .filter(Message.chat_id == chat_id, Message.id >= message_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    # Editing the opening message usually means a different building, so let the
    # next post name the chat and pick its cover again.
    if first and first[0] == message_id:
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if chat and chat.title_source != "manual":
            apply_chat_meta(
                db, chat_id, title_source="auto",
                thumbnail_url=None, thumbnail_source=None,
            )
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


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Convert an uploaded file to an Anthropic content block (image/document/text)."""
    data = await file.read()
    name = file.filename or "file"
    mime = file.content_type or ""
    ext  = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    # Image → native vision block
    if ext in ("jpg", "jpeg", "png", "gif", "webp") or mime.startswith("image/"):
        if len(data) > _MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image too large (max 5 MB)")
        if mime.startswith("image/"):
            media_type = mime
        else:
            media_type = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"
        block = {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(data).decode()}}
        return {"block": block, "name": name, "kind": "image"}

    # PDF → native document block
    if ext == "pdf" or mime == "application/pdf":
        if len(data) > _MAX_PDF_BYTES:
            raise HTTPException(413, "PDF too large (max 10 MB)")
        block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(data).decode()}}
        return {"block": block, "name": name, "kind": "pdf"}

    # DOCX → extract text
    if ext == "docx":
        if len(data) > _MAX_TEXT_BYTES:
            raise HTTPException(413, "File too large (max 1 MB)")
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            raise HTTPException(500, f"Could not read DOCX: {e}")
        block = {"type": "text", "text": f"[File: {name}]\n{text}"}
        return {"block": block, "name": name, "kind": "docx"}

    # TXT / MD → plain text
    if ext in ("txt", "md") or mime.startswith("text/"):
        if len(data) > _MAX_TEXT_BYTES:
            raise HTTPException(413, "File too large (max 1 MB)")
        text = data.decode("utf-8", errors="replace")
        block = {"type": "text", "text": f"[File: {name}]\n{text}"}
        return {"block": block, "name": name, "kind": "text"}

    raise HTTPException(415, f"Unsupported file type: .{ext or mime}")


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


def _agent_stream(chat_id: int, text: str, attachments: list[dict]):
    """Bridge the agent's work to the HTTP stream via a background worker thread.

    The worker runs run_agent to completion with its OWN db session and pushes SSE
    events into a queue. Because the worker is independent of the HTTP response, the
    full turn is generated and persisted even if the client disconnects mid-stream
    (e.g. a phone backgrounds Safari after pasting). Abandoning the HTTP generator
    does not stop the worker — it keeps draining into the unbounded queue and commits.
    """
    q: "queue.Queue[str | None]" = queue.Queue()

    def worker():
        db = SessionLocal()
        try:
            for sse in run_agent(chat_id, text, db, attachments):
                q.put(sse)
        except Exception as e:
            try:
                q.put(f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n")
            except Exception:
                pass
        finally:
            db.close()
            q.put(None)  # sentinel — signals the HTTP generator to stop

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item


@router.post("/chats/{chat_id}/messages")
def send_message(chat_id: int, body: MessageBody, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(404, "Chat not found")
    return StreamingResponse(
        _agent_stream(chat_id, body.text, body.attachments),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
