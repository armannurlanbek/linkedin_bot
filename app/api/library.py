"""Saved post library — save, list, update status, promote to style archive."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.db import LibraryPost, Post, get_db

router = APIRouter(prefix="/api/library")

EMBEDDING_MODEL = "text-embedding-3-small"


class SaveBody(BaseModel):
    text: str
    chat_id: int | None = None
    message_id: int | None = None


class PatchBody(BaseModel):
    status: str | None = None  # "draft" | "posted"


@router.post("", status_code=201)
def save_post(body: SaveBody, db: Session = Depends(get_db)):
    # Prevent exact duplicate saves from the same message
    if body.message_id:
        existing = db.query(LibraryPost).filter(LibraryPost.message_id == body.message_id).first()
        if existing:
            return {"id": existing.id, "already_saved": True}

    item = LibraryPost(
        text=body.text,
        chat_id=body.chat_id,
        message_id=body.message_id,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return _serialize(item)


@router.get("")
def list_posts(db: Session = Depends(get_db)):
    items = db.query(LibraryPost).order_by(LibraryPost.created_at.desc()).all()
    return [_serialize(i) for i in items]


@router.patch("/{item_id}")
def update_post(item_id: int, body: PatchBody, db: Session = Depends(get_db)):
    item = db.query(LibraryPost).filter(LibraryPost.id == item_id).first()
    if not item:
        raise HTTPException(404, "Not found")
    if body.status is not None:
        if body.status not in ("draft", "posted"):
            raise HTTPException(400, "status must be 'draft' or 'posted'")
        item.status = body.status
        if body.status == "posted" and item.posted_at is None:
            item.posted_at = datetime.now(timezone.utc)
        elif body.status == "draft":
            item.posted_at = None
    db.commit()
    db.refresh(item)
    return _serialize(item)


@router.delete("/{item_id}", status_code=204)
def delete_post(item_id: int, db: Session = Depends(get_db)):
    item = db.query(LibraryPost).filter(LibraryPost.id == item_id).first()
    if not item:
        raise HTTPException(404, "Not found")
    db.delete(item)
    db.commit()
    return Response(status_code=204)


@router.post("/{item_id}/promote")
def promote_post(item_id: int, db: Session = Depends(get_db)):
    """Embed the post text and add it to the RAG archive (posts table)."""
    item = db.query(LibraryPost).filter(LibraryPost.id == item_id).first()
    if not item:
        raise HTTPException(404, "Not found")
    if item.promoted:
        return {"ok": True, "already_promoted": True}

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[item.text])
    embedding = resp.data[0].embedding

    post = Post(
        text=item.text,
        embedding=embedding,
        char_count=len(item.text),
        source="library",
    )
    db.add(post)
    item.promoted = True
    db.commit()
    return {"ok": True}


def _serialize(item: LibraryPost) -> dict:
    return {
        "id": item.id,
        "text": item.text,
        "chat_id": item.chat_id,
        "message_id": item.message_id,
        "status": item.status,
        "promoted": item.promoted,
        "created_at": item.created_at,
        "posted_at": item.posted_at,
    }
