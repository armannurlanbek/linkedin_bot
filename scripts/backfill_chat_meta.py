"""
Rename existing conversations after the post they produced, and repoint their
cover images at the building the post is actually about.

Both were previously derived from the opening message — usually a bare URL —
which is why the sidebar reads "קישור לינקדאין קצר" and shows whichever photo
the scraped source page happened to carry.

Usage:
    python -m scripts.backfill_chat_meta --dry-run       # print old → new, change nothing
    python -m scripts.backfill_chat_meta                 # apply
    python -m scripts.backfill_chat_meta --titles        # titles only
    python -m scripts.backfill_chat_meta --thumbnails    # covers only
    python -m scripts.backfill_chat_meta --chat-id 348   # one chat
    python -m scripts.backfill_chat_meta --force         # redo 'post'/'auto' rows too
    python -m scripts.backfill_chat_meta --no-verify     # don't fetch-check covers

Covers are fetch-checked by default: a URL that 404s or is hotlink-protected
would show the placeholder icon, so the next candidate is tried instead.

Safe to re-run: each chat is committed on its own, and a chat is skipped once
its title or cover has a source that says it is already settled. Titles and
covers the user set by hand ('manual') are never touched, with or without
--force.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import text

import httpx

from app.db import SessionLocal, init_db
from app.services.chat_meta import (
    apply_chat_meta,
    collect_candidates_from_messages,
    extract_post_text,
    generate_post_title,
    iter_cover_candidates,
)

# Same headers /api/proxy-image sends — some publishers refuse a bare request
# but serve one that looks like a browser following a link from their own page.
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_IMAGE_SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG", b"GIF87a", b"GIF89a", b"RIFF")
_MAX_CANDIDATES_TRIED = 6


def _loads_as_image(client: httpx.Client, url: str) -> bool:
    """True when the URL actually returns image bytes.

    A cover that 404s or is hotlink-protected shows the 🏗 placeholder, so the
    backfill walks past it to the next candidate rather than storing it.
    """
    try:
        resp = client.get(url, headers={**_FETCH_HEADERS, "Referer": url})
    except Exception:
        return False
    if not resp.is_success:
        return False
    data = resp.content
    if any(data[:6].startswith(sig) for sig in _IMAGE_SIGNATURES):
        return True
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True
    return resp.headers.get("content-type", "").split(";")[0].strip().startswith("image/")


def _choose_cover(client, by_tool, title, verify: bool):
    """First candidate that is both plausible and (optionally) actually fetchable."""
    tried = 0
    first = None
    for url in iter_cover_candidates(by_tool, title):
        if first is None:
            first = url
        if not verify:
            return url, tried
        tried += 1
        if _loads_as_image(client, url):
            return url, tried
        if tried >= _MAX_CANDIDATES_TRIED:
            break
    # Nothing verified — keep the best guess rather than blanking the cover.
    return first, tried


def _last_post_text(messages) -> str | None:
    """The text of the most recent finished post in the chat."""
    for role, content in reversed(messages):
        if role != "assistant":
            continue
        post = extract_post_text(content)
        if post:
            return post
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print what would change")
    ap.add_argument("--titles", action="store_true", help="titles only")
    ap.add_argument("--thumbnails", action="store_true", help="covers only")
    ap.add_argument("--chat-id", type=int, help="only this chat")
    ap.add_argument("--limit", type=int, help="stop after N chats")
    ap.add_argument("--force", action="store_true", help="redo already-derived rows")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip checking that the chosen cover actually loads")
    args = ap.parse_args()

    # Neither flag given means both.
    do_titles = args.titles or not args.thumbnails
    do_thumbs = args.thumbnails or not args.titles

    init_db()  # ensures the new columns exist before we read them
    db = SessionLocal()
    verify = not args.no_verify
    http = httpx.Client(timeout=12, follow_redirects=True)

    where = "WHERE id = :chat_id" if args.chat_id else ""
    params = {"chat_id": args.chat_id} if args.chat_id else {}
    chats = db.execute(
        text(f"SELECT id, title, title_source, thumbnail_url, thumbnail_source "
             f"FROM chats {where} ORDER BY id"),
        params,
    ).all()

    counts = {"renamed": 0, "recovered": 0, "no_post": 0, "skipped": 0, "no_image": 0}
    processed = 0

    for chat_id, title, title_source, thumb_url, thumb_source in chats:
        if args.limit and processed >= args.limit:
            break

        title_eligible = do_titles and (
            title_source == "auto" or (args.force and title_source != "manual")
        )
        thumb_eligible = do_thumbs and (
            thumb_source in (None, "heuristic")
            or (args.force and thumb_source != "manual")
        )
        if not title_eligible and not thumb_eligible:
            counts["skipped"] += 1
            continue

        rows = db.execute(
            text("SELECT role, content FROM messages WHERE chat_id = :id "
                 "ORDER BY created_at, id"),
            {"id": chat_id},
        ).all()
        messages = [(r[0], r[1]) for r in rows]

        post_text = _last_post_text(messages)
        if not post_text:
            # No finished post — an abandoned chat, or one that only ever got a
            # refusal. Leave it exactly as it is.
            counts["no_post"] += 1
            print(f"{chat_id:>4}  — no finished post, left alone ({title!r})")
            continue

        processed += 1
        updates: dict = {}
        line = [f"{chat_id:>4}"]
        effective_title = title

        if title_eligible:
            new_title = generate_post_title(post_text)
            if new_title and new_title != title:
                updates["title"] = new_title
                updates["title_source"] = "post"
                effective_title = new_title
                line.append(f"title: {title!r} → {new_title!r}")
                counts["renamed"] += 1
            else:
                line.append(f"title: unchanged ({title!r})")

        if thumb_eligible:
            by_tool = collect_candidates_from_messages(
                [{"role": r, "content": c} for r, c in messages]
            )
            # The title names the building — cover selection weighs that above
            # which tool happened to return the image first.
            new_url, tried = _choose_cover(http, by_tool, effective_title, verify)
            if not new_url:
                counts["no_image"] += 1
                line.append("cover: no candidates, left alone")
            elif new_url != thumb_url:
                updates["thumbnail_url"] = new_url
                updates["thumbnail_source"] = "auto"
                src = ", ".join(f"{k}={len(v)}" for k, v in sorted(by_tool.items()))
                skipped = f", skipped {tried - 1} dead" if tried > 1 else ""
                line.append(f"cover: → {new_url[:70]}  [{src}{skipped}]")
                counts["recovered"] += 1
            else:
                updates["thumbnail_source"] = "auto"
                line.append("cover: already correct")

        print("  ".join(line))

        if updates and not args.dry_run:
            apply_chat_meta(db, chat_id, **updates)

    http.close()
    db.close()
    print()
    print(f"{'DRY RUN — nothing written' if args.dry_run else 'Applied'}: "
          f"{counts['renamed']} renamed, {counts['recovered']} covers changed, "
          f"{counts['no_post']} without a post, {counts['no_image']} without a candidate "
          f"image, {counts['skipped']} already settled.")


if __name__ == "__main__":
    main()
