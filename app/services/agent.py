import json
from pathlib import Path

import anthropic
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Chat, Message, SessionLocal
from app.services.tools import TOOLS, execute_tool, summarize_result

MODEL = "claude-sonnet-4-6"
STYLE_CARD_PATH    = Path(__file__).parent.parent.parent / "style_card.json"
EXAMPLE_POSTS_PATH = Path(__file__).parent.parent.parent / "example_posts.json"


def _load_style_card() -> str:
    if STYLE_CARD_PATH.exists():
        with open(STYLE_CARD_PATH, encoding="utf-8") as f:
            return json.dumps(json.load(f), ensure_ascii=False, indent=2)
    return "{}"


def _load_example_posts() -> str:
    if not EXAMPLE_POSTS_PATH.exists():
        return ""
    with open(EXAMPLE_POSTS_PATH, encoding="utf-8") as f:
        posts = json.load(f)
    parts = []
    for i, text in enumerate(posts, 1):
        parts.append(f"[דוגמה {i}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _build_system() -> str:
    style_card    = _load_style_card()
    examples_text = _load_example_posts()

    examples_section = ""
    if examples_text:
        examples_section = f"""
════════════════════════════════════════
דוגמאות אמיתיות לסגנון הכתיבה של רומן
(אלה פוסטים שהוא כתב — ממש ככה צריך להישמע הפלט)
════════════════════════════════════════

{examples_text}

════════════════════════════════════════
"""

    return f"""אתה כותב פוסטים ללינקדאין עבור רומן מינייב, מנכ"ל Orlanda — חברה ישראלית למעטפת בניין.
המשימה: לכתוב פוסט שנשמע בדיוק כמו רומן — לא כמו AI, לא כמו עיתונאי, כמו רומן.
{examples_section}
כרטיס הסגנון (מאפיינים שנחלצו מ-300 פוסטים):
{style_card}

════ כללי כתיבה ════
- כתוב בעברית, שפה שיחתית ופשוטה — כמו שרומן מדבר, לא כמו מאמר
- מונחים טכניים באנגלית: curtain wall, IGU, LEED, facade, GRC וכד'
- פתיחה: שאלה סקרנית / קביעה נועזת / עובדה מפתיעה — קצרה, מושכת
- תוכן: עומק טכני אמיתי — מידות, חומרים, שיטות, ספקים — לא כללי
- סיום: שאלה לקוראים OR קרדיטים (יזם / קבלן ראשי / אדריכל / מעטפת)
- אם פרט לא ידוע — "לא ידוע", אל תמציא
- 2-4 האשטאגים בסוף בלבד
- אל תכתוב "רומן מינייב" / "אני מנכ"ל" / "אני שמח לשתף"
- אורך: 800-2000 תווים

════ תהליך כתיבה ════
1. לאחר שאספת מידע (scrape + search + similar posts):
   א. בחר מתוך הדוגמאות הדומות את זו שהכי קרובה לנושא
   ב. הסתכל על מבנה הפתיחה שלה, קצב הפסקאות, האיזון בין טכני לשיחתי
   ג. כתוב פוסט שמשתמש בדיוק באותו מבנה ואותה שפה — רק עם תוכן חדש
2. אל תכתוב "להלן הפוסט" / "הנה הרשומה" — תן רק את הפוסט עצמו

════ שימוש בכלים ════
- URL חדש → scrape_url → search_web → retrieve_similar_posts → כתוב
- תיקון / שינוי → כתוב מחדש מהזיכרון, ללא כלים
- URL חדש נוסף → חזור על הכלים מהתחלה"""


def _save_message(db: Session, chat_id: int, role: str, content: list) -> None:
    msg = Message(chat_id=chat_id, role=role, content=content)
    db.add(msg)
    db.commit()


def _sanitize_block(block: dict) -> dict:
    """Strip SDK-internal fields that the Anthropic API rejects on replay."""
    t = block.get("type")
    if t == "text":
        return {"type": "text", "text": block.get("text", "")}
    if t == "thinking":
        out = {"type": "thinking", "thinking": block.get("thinking", "")}
        if "signature" in block:
            out["signature"] = block["signature"]
        return out
    if t == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id", ""),
            "name": block.get("name", ""),
            "input": block.get("input", {}),
        }
    if t == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id", ""),
            "content": block.get("content", ""),
        }
    return block


def _load_history(db: Session, chat_id: int) -> list:
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )
    result = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            content = [
                _sanitize_block(b) if isinstance(b, dict) else b
                for b in content
            ]
        result.append({"role": m.role, "content": content})
    return result


def _generate_title(user_text: str) -> str:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=30,
        messages=[
            {
                "role": "user",
                "content": (
                    f"תן כותרת קצרה (3-4 מילים בעברית) לשיחה שמתחילה כך: "
                    f"{user_text[:200]}\nרק הכותרת, ללא פיסוק."
                ),
            }
        ],
    )
    return resp.content[0].text.strip()


def run_agent(chat_id: int, user_text: str, db: Session):
    """Sync generator — yields SSE strings."""
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # Check if first message in chat (before saving current one)
        existing = db.query(Message).filter(Message.chat_id == chat_id).count()
        is_first = existing == 0

        # Save user message
        _save_message(db, chat_id, "user", [{"type": "text", "text": user_text}])

        # Generate and emit title on first message
        if is_first:
            try:
                title = _generate_title(user_text)
                db.query(Chat).filter(Chat.id == chat_id).update({"title": title})
                db.commit()
                yield _sse({"type": "title", "title": title})
            except Exception:
                pass

        # Build message history for Claude
        history = _load_history(db, chat_id)
        system = _build_system()

        # Tool-use loop (max 8 iterations)
        for _ in range(8):
            with client.messages.stream(
                model=MODEL,
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 10000},
                system=system,
                tools=TOOLS,
                messages=history,
            ) as stream:
                for event in stream:
                    event_type = type(event).__name__
                    if event_type == "RawContentBlockDeltaEvent":
                        delta = event.delta
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            yield _sse({"type": "text_delta", "text": delta.text})
                        elif delta_type == "thinking_delta":
                            yield _sse({"type": "thinking_delta", "text": delta.thinking})

                final = stream.get_final_message()

            # Serialize content blocks for DB storage
            content_blocks = []
            for block in final.content:
                if hasattr(block, "model_dump"):
                    content_blocks.append(block.model_dump())
                else:
                    content_blocks.append(dict(block))

            _save_message(db, chat_id, "assistant", content_blocks)
            history.append({"role": "assistant", "content": final.content})

            if final.stop_reason == "end_turn":
                yield _sse({"type": "done"})
                return

            # Handle tool_use
            tool_results = []
            for block in final.content:
                if getattr(block, "type", None) == "tool_use":
                    yield _sse(
                        {
                            "type": "tool_call",
                            "tool": block.name,
                            "input": block.input,
                            "id": block.id,
                        }
                    )
                    try:
                        result = execute_tool(block.name, block.input, db)
                    except Exception as e:
                        result = {"error": str(e)}
                    preview = summarize_result(block.name, result)
                    images = []
                    if block.name == "scrape_url" and isinstance(result.get("images"), list):
                        images = [u for u in result["images"] if isinstance(u, str) and u.startswith("http")][:12]
                    yield _sse({"type": "tool_result", "id": block.id, "preview": preview, "images": images})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

            if not tool_results:
                yield _sse({"type": "done"})
                return

            _save_message(db, chat_id, "user", tool_results)
            history.append({"role": "user", "content": tool_results})

        yield _sse({"type": "done"})

    except Exception as e:
        yield _sse({"type": "error", "message": str(e)})
