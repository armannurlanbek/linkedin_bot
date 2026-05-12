"""Retrieve the N most similar past posts from pgvector."""

from openai import OpenAI
from sqlalchemy.orm import Session
from app.config import settings
from app.db import Post

EMBEDDING_MODEL = "text-embedding-3-small"


def get_similar_posts(db: Session, query_text: str, n: int = 8) -> list[dict]:
    """Embed query_text and return the n closest posts by cosine distance."""
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query_text])
    query_embedding = response.data[0].embedding

    from pgvector.sqlalchemy import Vector
    from sqlalchemy import cast

    results = (
        db.query(
            Post,
            Post.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .order_by(Post.embedding.cosine_distance(query_embedding))
        .limit(n)
        .all()
    )
    return [
        {"text": post.text, "similarity": round(1 - float(distance), 3)}
        for post, distance in results
    ]
