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


def _build_system() -> list:
    """Return a list of system content blocks with cache_control on the large static block."""
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

    static_block = f"""אתה כותב פוסטים ללינקדאין עבור רומן מינייב, מנכ"ל Orlanda — חברה ישראלית למעטפת בניין.
המשימה: לכתוב פוסט שנשמע בדיוק כמו רומן — לא כמו AI, לא כמו עיתונאי, כמו רומן.
{examples_section}
כרטיס הסגנון (מאפיינים שנחלצו מ-300 פוסטים):
{style_card}

════ מה זה אומר "לכתוב כמו רומן" ════
רומן הוא מישהו שמוצא בניינים מעניינים ורוצה לספר לך למה. הוא לא כותב דוח טכני ולא כתבה בעיתון — הוא מדבר איתך.
הדרך הכי קלה לבדוק: קרא את הפוסט בקול. אם הוא נשמע כמו בן אדם — טוב. אם הוא נשמע כמו מהנדס שכתב מצגת — התחל מחדש.

════ שפה ════
- עברית פשוטה וישירה, משפטים קצרים
- מונחים טכניים באנגלית בלבד: curtain wall, IGU, unitized, stick system, GRC, LEED, U-value, SHGC
- אסור להשתמש במילים האלו בעברית — הן נשמעות כמו תרגום: "ביוקלימטי", "פרמטרי", "טביעת רגל משולשת", "גיאומטריה"
- אסור: "מרשים", "מרהיב", "חזון אדריכלי", "מסמן עידן חדש", "ייחודי", "חדשני", "מהפכני"
- במקום תארים — עובדה: לא "מרשים" אלא "1,000 מ״ר פאנלים סולאריים בתוך המעטפת"

════ מבנה הפוסט ════
1. פתיחה — שורה אחת שמושכת פנימה:
   - שאלה: "אבל למה הרבה מהפריזאים סולדים ממנו?"
   - עובדה שמפתיעה: "ברגעים אלו ממש נבנה אחד מהמבנים השנויים במחלוקת בפריז"
   - מספר בולט: "180 מטר. 44 קומות. 10 שנות התנגדות."

2. פקטשיט קצר (key-value, לא bullet points):
   מיקום: ...
   שנת בנייה: ...
   אדריכל: ...  ← שמות מדויקים מהחיפוש בלבד
   גובה / קומות / שטח / שימוש

3. גוף הפוסט — הסיפור:
   - הסבר למה הדברים נעשו כך, לא רק מה נעשה
   - תוכן טכני אמיתי: חומרים, מידות, שיטות, ביצועים — אבל מוסבר בפשטות
   - "הצורה המשולשת גורמת לכך שהחלק העליון הכי צר — וככה מטיל הכי פחות צל על הרחוב"
     ולא: "triangular footprint מגיב לכיוון השמש ולרוחות השכיחות"

4. סיום — אנושי וזכיר:
   - השוואה, אירוניה, שאלה לקוראים, מקבילה היסטורית — כמו "גם למגדל אייפל היו המוני מתנגדים"
   - לא תחזית כללית ("אני מניח שבעוד עשר שנים...") — זה חלש

5. קרדיטים — בתוך הפקטשיט או בסוף, לפי מה שמתאים לפוסט:
   אדריכל / יזם / קבלן ראשי / מעטפת
   ⚠️ רק שמות שמופיעים במפורש במקורות. אם לא נמצא — "לא ידוע". אף פעם לא להמציא.

6. 2 האשטאגים בסוף בלבד (רומן משתמש ב-2, לא ב-4)

════ כללים נוקשים ════
- אל תמציא שמות של חברות, אנשים, ספקים — רק מה שמופיע במקורות
- אל תכתוב "רומן מינייב" / "אני מנכ"ל" / "אני שמח לשתף" / "להלן הפוסט"
- תן רק את הפוסט עצמו, ללא הקדמות
- אורך: 800-2000 תווים

════ תהליך כתיבה ════
1. בחר את הדוגמה הדומה ביותר מהארכיון
2. שים לב: מה הזווית שרומן בחר? מה הפתיחה? איך הוא מסביר דברים טכניים בפשטות?
3. כתוב עם אותו קול — תוכן חדש, אותה גישה

════ שימוש בכלים — כללים מחייבים ════
סדר קבוע לכל URL חדש (בדיוק בסדר הזה, בלי דילוגים):
  1. scrape_url
  2. search_web — שם הבניין + עיר + developer contractor architect facade (וגם בעברית)
  3. search_images — שם הבניין + עיר + "exterior facade"
  4. retrieve_similar_posts
  5. כתוב את הפוסט — STOP, אין כלים נוספים אחרי זה!

חוקים נוספים:
- תיקון / שינוי → כתוב מחדש מהזיכרון, ללא כלים בכלל
- URL נוסף → חזור על שלבים 1-5 מהתחלה
- אסור לקרוא לכלים אחרי שהתחלת לכתוב את הפוסט"""

    return [
        {
            "type": "text",
            "text": static_block,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _save_message(db: Session, chat_id: int, role: str, content: list) -> None:
    try:
        msg = Message(chat_id=chat_id, role=role, content=content)
        db.add(msg)
        db.commit()
    except Exception:
        db.rollback()
        raise


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


def _repair_tool_pairs(history: list) -> list:
    """Ensure every tool_use block has a matching tool_result in the next message.

    If the tool_result message was never saved (e.g. due to a DB error), the API
    rejects the history with a 400.  We inject synthetic tool_results so the
    conversation can continue cleanly.
    """
    def _block_type(b):
        return b.get("type") if isinstance(b, dict) else getattr(b, "type", None)

    def _block_id(b):
        return b.get("id") if isinstance(b, dict) else getattr(b, "id", None)

    def _block_tool_use_id(b):
        return b.get("tool_use_id") if isinstance(b, dict) else getattr(b, "tool_use_id", None)

    out = []
    i = 0
    while i < len(history):
        msg = history[i]
        if msg["role"] == "assistant":
            content = msg["content"] if isinstance(msg["content"], list) else []
            tool_use_ids = [_block_id(b) for b in content if _block_type(b) == "tool_use" and _block_id(b)]

            if tool_use_ids:
                out.append(msg)
                next_msg = history[i + 1] if i + 1 < len(history) else None

                if next_msg and next_msg["role"] == "user":
                    next_content = next_msg["content"] if isinstance(next_msg["content"], list) else []
                    covered = {_block_tool_use_id(b) for b in next_content if _block_type(b) == "tool_result"}
                    missing = [id for id in tool_use_ids if id not in covered]
                    if missing:
                        # Inject synthetic results for the missing ones
                        patched = [
                            {"type": "tool_result", "tool_use_id": id, "content": "(interrupted)"}
                            for id in missing
                        ] + list(next_content)
                        out.append({"role": "user", "content": patched})
                        i += 2
                        continue
                    # next_msg already has all tool_results — append it and advance
                    out.append(next_msg)
                    i += 2
                    continue
                else:
                    # No following user message at all — synthesize one entirely
                    out.append({
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": id, "content": "(interrupted)"}
                            for id in tool_use_ids
                        ],
                    })
                    i += 1
                    continue

        out.append(msg)
        i += 1
    return out


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
    return _repair_tool_pairs(result)


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
        system = _build_system()  # list of blocks with cache_control

        # Track yielded images to avoid duplicates across tool calls
        seen_images: set[str] = set()
        thumbnail_saved = False

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

            try:
                _save_message(db, chat_id, "assistant", content_blocks)
            except Exception:
                pass  # Don't let a DB write failure abort the stream
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
                    if block.name in ("scrape_url", "search_images") and isinstance(result.get("images"), list):
                        raw = [u for u in result["images"] if isinstance(u, str) and u.startswith("http")][:12]
                        images = [u for u in raw if u not in seen_images]
                        seen_images.update(images)
                    if images and not thumbnail_saved:
                        try:
                            db.query(Chat).filter(Chat.id == chat_id).update({"thumbnail_url": images[0]})
                            db.commit()
                            thumbnail_saved = True
                            yield _sse({"type": "thumbnail", "url": images[0]})
                        except Exception:
                            db.rollback()
                    yield _sse({"type": "tool_result", "id": block.id, "preview": preview, "images": images})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, ensure_ascii=False),
                            "__images__": images,
                        }
                    )

            if not tool_results:
                yield _sse({"type": "done"})
                return

            try:
                _save_message(db, chat_id, "user", tool_results)
            except Exception:
                pass
            history.append({"role": "user", "content": [_sanitize_block(r) for r in tool_results]})

        yield _sse({"type": "done"})

    except Exception as e:
        yield _sse({"type": "error", "message": str(e)})
