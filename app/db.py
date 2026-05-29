import logging

from sqlalchemy import Boolean, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, DateTime, func, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from app.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url_sync)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    linkedin_id = Column(String, unique=True, nullable=True, index=True)
    text = Column(String, nullable=False)
    embedding = Column(Vector(1536))
    char_count = Column(Integer)
    source = Column(String, nullable=True)  # "linkedin" | "library" | null (legacy)
    posted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False, default="New chat")
    thumbnail_url = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    role = Column(String, nullable=False)    # "user" | "assistant"
    content = Column(JSONB, nullable=False)  # raw Anthropic content block list
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class LibraryPost(Base):
    __tablename__ = "library_posts"

    id         = Column(Integer, primary_key=True)
    text       = Column(String, nullable=False)
    chat_id    = Column(Integer, ForeignKey("chats.id", ondelete="SET NULL"), nullable=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    status     = Column(String, nullable=False, default="draft")    # "draft" | "posted"
    source     = Column(String, nullable=False, default="chat")     # "chat" | "manual"
    promoted   = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    posted_at  = Column(DateTime(timezone=True), nullable=True)


def init_db():
    # The `vector` extension is provisioned once by a superuser in db_roman.
    # The app's role ("roman") is NOT a superuser and cannot CREATE EXTENSION,
    # and on rare occasions the connection may briefly be read-only during a
    # failover. Attempt it best-effort so neither case crashes startup.
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as exc:
        logger.warning(
            "Skipping CREATE EXTENSION vector (already present or insufficient "
            "privilege): %s", exc,
        )
    Base.metadata.create_all(bind=engine)
    # Additive migrations — safe to re-run on every startup
    _migrate()


def _migrate():
    """Run each DDL statement in its own auto-committing transaction."""
    stmts = [
        "ALTER TABLE chats ADD COLUMN IF NOT EXISTS thumbnail_url TEXT",
        "ALTER TABLE posts ADD COLUMN IF NOT EXISTS source TEXT",
        "ALTER TABLE library_posts ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
        "ALTER TABLE library_posts ADD COLUMN IF NOT EXISTS posted_at TIMESTAMPTZ",
    ]
    for stmt in stmts:
        try:
            with engine.begin() as conn:   # auto-commits on exit, rolls back on exception
                conn.execute(text(stmt))
        except Exception:
            pass  # column already exists or table missing — safe to ignore


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
