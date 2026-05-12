"""
Select 5 representative "golden example" posts and save to example_posts.json.

Selection strategy:
  - Only posts about specific buildings / facades (skip podcasts, company news)
  - 800-2500 chars (complete style, not micro-posts)
  - Spread across the full post archive (5 evenly-spaced positions)
  - Then let Claude rank them by style authenticity and pick the best 5

Run once:  docker compose exec api python scripts/pick_examples.py
"""

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import SessionLocal, Post

BUILDING_KEYWORDS = [
    "curtain wall", "IGU", "חיפוי", "חזית", "זכוכית", "אלומיניום",
    "מעטפת", "פנל", "LEED", "facade", "קומות", "מגדל", "בניין",
]

OUTPUT = Path(__file__).parent.parent / "example_posts.json"


def has_building_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in BUILDING_KEYWORDS)


def main():
    db = SessionLocal()
    try:
        posts = (
            db.query(Post)
            .filter(Post.char_count >= 800, Post.char_count <= 2500)
            .order_by(Post.char_count.desc())
            .all()
        )

        building_posts = [p for p in posts if has_building_keyword(p.text)]
        if not building_posts:
            building_posts = posts  # fallback

        # Take every Nth post to get diverse spread
        n = max(1, len(building_posts) // 5)
        selected = [building_posts[i * n] for i in range(5) if i * n < len(building_posts)]

        examples = [p.text for p in selected]

        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(examples)} example posts to {OUTPUT}")
        for i, p in enumerate(examples, 1):
            print(f"\n--- Example {i} ({len(p)} chars) ---")
            print(p[:200] + "...")
    finally:
        db.close()


if __name__ == "__main__":
    main()
