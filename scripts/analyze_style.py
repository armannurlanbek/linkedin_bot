"""
Analyze writing style of all ingested posts and produce style_card.json.

Usage:
    python -m scripts.analyze_style

Reads all posts from the DB, sends them to Claude, and saves style_card.json
in the project root. That file will be injected into every generation prompt.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import anthropic
from sqlalchemy.orm import Session
from app.config import settings
from app.db import Post, SessionLocal, init_db

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style_card.json")
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert linguistic analyst specializing in Hebrew social media writing.
Your job is to analyze a corpus of LinkedIn posts and produce a precise, actionable style guide.
Respond ONLY with valid JSON — no markdown, no explanation outside the JSON object."""

ANALYSIS_PROMPT = """Below are {count} LinkedIn posts written by a CEO of a construction company
specializing in building envelopes (facades, curtain walls, cladding systems).
The posts are primarily in Hebrew.

Analyze the corpus and return a JSON object with exactly these keys:

{{
  "typical_length": {{
    "description": "short / medium / long and the typical character range",
    "avg_chars": <number>,
    "range": "<min>-<max> chars"
  }},
  "post_openings": {{
    "description": "how he typically starts posts",
    "patterns": ["<pattern 1>", "<pattern 2>", ...],
    "examples": ["<verbatim opening from corpus>", ...]
  }},
  "sentence_structure": {{
    "style": "short-punchy / long-flowing / mixed",
    "description": "<detail>"
  }},
  "emoji_usage": {{
    "frequency": "none / rare / moderate / heavy",
    "typical_emojis": ["<emoji>", ...],
    "placement": "opening / closing / inline / all"
  }},
  "hashtag_usage": {{
    "count_per_post": "<range>",
    "placement": "end / inline / mixed",
    "common_hashtags": ["<tag>", ...],
    "topics": ["<topic>", ...]
  }},
  "tone": {{
    "primary": "<authoritative / conversational / inspirational / educational>",
    "secondary": "<second tone>",
    "description": "<nuance>"
  }},
  "technical_vocabulary": {{
    "building_envelope_terms": ["<term>", ...],
    "hebrew_technical_terms": ["<term>", ...],
    "english_terms_used_in_hebrew": ["<term>", ...]
  }},
  "post_closings": {{
    "patterns": ["CTA / question / statement / credits"],
    "examples": ["<verbatim closing>", ...]
  }},
  "hebrew_english_mixing": {{
    "pattern": "<description of how Hebrew and English are mixed>",
    "english_usage": "<when does he switch to English words/phrases>"
  }},
  "readability": {{
    "level": "simple / moderate / complex",
    "description": "<detail on vocabulary complexity, sentence length, etc.>"
  }},
  "recurring_themes": ["<theme 1>", "<theme 2>", ...],
  "things_to_avoid": ["<pattern or style to never mimic>", ...]
}}

POSTS:
---
{posts}
---"""


def load_posts(db: Session) -> list[str]:
    return [p.text for p in db.query(Post).all()]


def analyze(db: Session):
    posts = load_posts(db)
    if not posts:
        print("No posts found in the database. Run ingest.py first.")
        sys.exit(1)

    print(f"Loaded {len(posts)} posts from DB.")

    # Send at most 200 posts to stay within context limits while being representative
    sample = posts[:200]
    posts_text = "\n\n---\n\n".join(sample)

    prompt = ANALYSIS_PROMPT.format(count=len(sample), posts=posts_text)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    print("Calling Claude for style analysis (this may take ~30 seconds)...")

    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if Claude added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    style_card = json.loads(raw)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(style_card, f, ensure_ascii=False, indent=2)

    print(f"\nStyle card saved to {OUTPUT_PATH}")
    print(json.dumps(style_card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    init_db()
    db: Session = SessionLocal()
    try:
        analyze(db)
    finally:
        db.close()
