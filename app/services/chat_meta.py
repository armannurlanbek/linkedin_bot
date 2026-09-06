"""Chat metadata — the sidebar title and cover image of a conversation.

Both are derived from the *finished post*, not from the opening message. The
first message is usually a bare URL, which is why titles used to read
"קישור לינקדאין קצר" and covers showed whatever image the source page happened
to carry. Everything here runs after the post exists, and every write goes
through `apply_chat_meta` so the sidebar's ordering is never disturbed.
"""

import json
import logging
import re

import anthropic
from sqlalchemy import text

from app.config import settings

logger = logging.getLogger(__name__)

TITLE_MODEL = "claude-sonnet-4-6"

# A finished post is long. This threshold keeps out the short interstitial
# replies — above all the disambiguation hard-stop ("מצאתי מספר פרופילים…"),
# which is an assistant turn with no tool_use and would otherwise be mistaken
# for the post itself.
_MIN_POST_CHARS = 300

_MAX_TITLE_CHARS = 45

# Ordered best-first. search_images is queried with the building's name and its
# results are relevance-ranked in search.py::_rank_images, so those images are
# about the subject by construction. scrape_url images come from the source
# page and are frequently a publisher cover or an unrelated illustration —
# which is exactly how the wrong building ended up in the sidebar.
IMAGE_TOOL_PRIORITY = ("search_images", "scrape_url")


def _blocks(content) -> list:
    """Normalise a stored message body to a list of content blocks.

    psycopg2 parses JSONB into Python objects, but a raw driver or another
    backend can hand the column back as text. Parse that rather than silently
    treating a whole conversation as having no blocks.
    """
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _block_get(b, key):
    if isinstance(b, dict):
        return b.get(key)
    return getattr(b, key, None)


def is_post_message(content) -> bool:
    """True when this assistant message is a finished post rather than a tool turn."""
    blocks = _blocks(content)
    if any(_block_get(b, "type") == "tool_use" for b in blocks):
        return False
    return len(_message_text(blocks)) >= _MIN_POST_CHARS


def _message_text(blocks) -> str:
    return "\n".join(
        _block_get(b, "text") or ""
        for b in blocks
        if _block_get(b, "type") == "text"
    ).strip()


def extract_post_text(content) -> str | None:
    """Return the post text if this assistant message is a finished post."""
    blocks = _blocks(content)
    if any(_block_get(b, "type") == "tool_use" for b in blocks):
        return None
    body = _message_text(blocks)
    return body if len(body) >= _MIN_POST_CHARS else None


def collect_candidates_from_messages(messages) -> dict[str, list[str]]:
    """Recover per-tool image candidates from stored messages.

    Tool results keep their images under `__images__` (stripped before the API
    call by agent._sanitize_block, kept in the DB row). The tool that produced
    them is recovered through tool_use_id, the same lookup the frontend does.
    """
    by_tool: dict[str, list[str]] = {}
    tool_name_by_id: dict[str, str] = {}
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        blocks = _blocks(content)
        if role == "assistant":
            tool_name_by_id = {
                _block_get(b, "id"): _block_get(b, "name")
                for b in blocks
                if _block_get(b, "type") == "tool_use" and _block_get(b, "id")
            }
            continue
        for b in blocks:
            if _block_get(b, "type") != "tool_result":
                continue
            images = _block_get(b, "__images__")
            if not isinstance(images, list):
                continue
            tool = tool_name_by_id.get(_block_get(b, "tool_use_id")) or "unknown"
            bucket = by_tool.setdefault(tool, [])
            for url in images:
                if isinstance(url, str) and url.startswith("http") and url not in bucket:
                    bucket.append(url)
    return by_tool


# Stock libraries, clipart and vector art make poor covers — a Freepik vector of
# "cross shapes" is not a photo of the building. search.py already rejects these
# at search time, but candidates recovered from older conversations were stored
# before that filter existed, so cover selection applies the standard again.
# Deliberately narrower than search.py's list: social CDN hosts stay allowed,
# because Instagram and Facebook post photos are extracted on purpose.
_COVER_SKIP = (
    "freepik", "shutterstock", "istockphoto", "dreamstime", "123rf",
    "vecteezy", "pixabay", "publicdomainpictures", "getdrawings",
    "pinimg", "pinterest", "etsy",
    "clipart", "/logo", "logo.", "favicon", "sprite", "placeholder", "avatar",
)


def _usable_cover(url) -> bool:
    if not isinstance(url, str) or not url.startswith("http"):
        return False
    low = url.lower()
    return not any(bad in low for bad in _COVER_SKIP)


def pick_thumbnail(by_tool: dict[str, list[str]]) -> str | None:
    """Choose the cover image: the subject-searched photo before the scraped one."""
    ordered = list(IMAGE_TOOL_PRIORITY) + [
        t for t in by_tool if t not in IMAGE_TOOL_PRIORITY
    ]
    for tool in ordered:
        for url in by_tool.get(tool) or []:
            if _usable_cover(url):
                return url
    return None


_TITLE_PROMPT = """להלן פוסט מוכן. תן לו שם קצר לרשימת השיחות.

הכלל: השם הוא שם הבניין / הפרויקט / הנושא שהפוסט עוסק בו — כפי שהוא מופיע בפוסט.
- בניין או פרויקט: שם הבניין, ואם השם לבדו לא ברור הוסף פסיק ואת העיר.
  לדוגמה: "מגדל עזריאלי שרונה" / "One Vanderbilt, ניו יורק"
- נושא שאינו בניין: 3-4 מילים שמתארות את הנושא.

חוקים: עד {max_chars} תווים. בלי מירכאות. בלי נקודה בסוף. בלי "פוסט על".
אל תכתוב שום דבר מלבד השם עצמו.

הפוסט:
{post}"""


def _clean_title(raw: str) -> str:
    title = (raw or "").strip().splitlines()[0].strip() if raw and raw.strip() else ""
    title = title.strip("\"'«»“”‚'`").strip()
    title = re.sub(r"^(הכותרת|כותרת|שם השיחה)\s*[:\-–]\s*", "", title).strip()
    title = title.rstrip(".،,;:").strip()
    if len(title) > _MAX_TITLE_CHARS:
        cut = title[:_MAX_TITLE_CHARS].rsplit(" ", 1)[0]
        title = (cut or title[:_MAX_TITLE_CHARS]).strip()
    return title


def generate_post_title(post_text: str) -> str | None:
    """One small model call: name the conversation after the post's subject."""
    if not post_text or not post_text.strip():
        return None
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=TITLE_MODEL,
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": _TITLE_PROMPT.format(
                    max_chars=_MAX_TITLE_CHARS, post=post_text[:3000]
                ),
            }],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        title = _clean_title("".join(parts))
        return title or None
    except Exception as exc:
        logger.warning("Title generation failed for a post: %s", exc)
        return None


def apply_chat_meta(db, chat_id: int, **fields) -> None:
    """Write chat metadata without disturbing `updated_at`.

    `chats.updated_at` carries `onupdate=func.now()`, and the sidebar is ordered
    by it. An ORM write here — especially the backfill's — would reshuffle the
    whole list into the order the rows happened to be touched. Raw SQL keeps
    `updated_at` as the timestamp of the last real conversation activity.
    """
    allowed = (
        "title", "title_source", "thumbnail_url", "thumbnail_source",
        "posted_at", "posted_message_id",
    )
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    assignments = ", ".join(f"{k} = :{k}" for k in sets)
    params = dict(sets, chat_id=chat_id)
    db.execute(text(f"UPDATE chats SET {assignments} WHERE id = :chat_id"), params)
    db.commit()


def candidates_from_db(db, chat_id: int) -> dict[str, list[str]]:
    """Recover image candidates for a chat from its stored messages."""
    rows = db.execute(
        text("SELECT role, content FROM messages WHERE chat_id = :id "
             "ORDER BY created_at, id"),
        {"id": chat_id},
    ).all()
    return collect_candidates_from_messages(
        [{"role": r[0], "content": r[1]} for r in rows]
    )


def finalize_chat_meta(db, chat_id: int, post_text: str, by_tool: dict[str, list[str]]) -> list[dict]:
    """Name the chat and set its cover once the post is written.

    `by_tool` holds the images gathered during this turn. It is empty when the
    post came from a turn that ran no image tools — the second half of the
    LinkedIn disambiguation flow, or a "rewrite the opening" follow-up — so the
    candidates are then recovered from the stored messages instead.

    Returns SSE-shaped events for the caller to yield. Never raises: metadata is
    a nicety and must not be able to fail a turn that already produced a post.
    """
    events: list[dict] = []
    try:
        row = db.execute(
            text("SELECT title_source, thumbnail_source FROM chats WHERE id = :id"),
            {"id": chat_id},
        ).first()
        if row is None:
            return events
        title_source, thumbnail_source = row[0] or "auto", row[1]

        updates: dict = {}

        # A title the user typed, or one already derived from this post, stands.
        if title_source == "auto":
            title = generate_post_title(post_text)
            if title:
                updates["title"] = title
                updates["title_source"] = "post"
                events.append({"type": "title", "title": title})

        # A cover the user chose stands. A mid-run placeholder gets upgraded.
        if thumbnail_source != "manual":
            candidates = by_tool or candidates_from_db(db, chat_id)
            url = pick_thumbnail(candidates)
            if url:
                updates["thumbnail_url"] = url
                updates["thumbnail_source"] = "auto"
                events.append({"type": "thumbnail", "url": url, "replace": True})

        if updates:
            apply_chat_meta(db, chat_id, **updates)
    except Exception as exc:
        logger.warning("Chat metadata finalisation failed for chat %s: %s", chat_id, exc)
        try:
            db.rollback()
        except Exception:
            pass
        return []
    return events
