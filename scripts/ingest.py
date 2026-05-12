"""
Ingest LinkedIn posts from posts.json into pgvector.

Usage:
    python -m scripts.ingest                          # uses posts.json in project root
    python -m scripts.ingest --json path/to/file.json

Each JSON item is expected to have:
    - content   : the post text (Hebrew)
    - id        : LinkedIn post ID
    - postedAt.date : ISO timestamp (optional)
"""

import argparse
import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from sqlalchemy.orm import Session
from app.config import settings
from app.db import Post, SessionLocal, init_db

EMBEDDING_MODEL = "text-embedding-3-small"
MIN_CHARS = 50
DEFAULT_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "posts.json")


def clean(text: str) -> str | None:
    t = str(text).strip()
    if not t or t.lower() == "nan":
        return None
    if len(t) < MIN_CHARS:
        return None
    return t


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend([r.embedding for r in response.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return embeddings


def parse_date(item: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(item["postedAt"]["date"].replace("Z", "+00:00"))
    except Exception:
        return None


def ingest(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        raw_items = json.load(f)

    print(f"Loaded {len(raw_items)} items from {json_path}")

    # Build deduplicated list keyed by LinkedIn ID
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    records: list[dict] = []

    for item in raw_items:
        linkedin_id = str(item.get("id", "")).strip()
        text = clean(item.get("content", ""))
        if not text:
            continue
        if linkedin_id and linkedin_id in seen_ids:
            continue
        if text in seen_texts:
            continue
        seen_ids.add(linkedin_id)
        seen_texts.add(text)
        records.append({
            "linkedin_id": linkedin_id or None,
            "text": text,
            "posted_at": parse_date(item),
        })

    print(f"After cleaning: {len(records)} posts (dropped {len(raw_items) - len(records)})")

    openai_client = OpenAI(api_key=settings.openai_api_key)
    print("\nGenerating embeddings...")
    texts = [r["text"] for r in records]
    embeddings = embed_texts(openai_client, texts)

    init_db()
    db: Session = SessionLocal()
    try:
        inserted = 0
        skipped = 0
        for record, embedding in zip(records, embeddings):
            # Skip if this LinkedIn ID already exists
            if record["linkedin_id"]:
                exists = db.query(Post).filter(Post.linkedin_id == record["linkedin_id"]).first()
                if exists:
                    skipped += 1
                    continue
            post = Post(
                linkedin_id=record["linkedin_id"],
                text=record["text"],
                embedding=embedding,
                char_count=len(record["text"]),
                posted_at=record["posted_at"],
            )
            db.add(post)
            inserted += 1
        db.commit()
        print(f"\nDone. Inserted {inserted} new posts, skipped {skipped} already in DB.")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=DEFAULT_JSON, help="Path to posts.json")
    args = parser.parse_args()
    ingest(args.json)
