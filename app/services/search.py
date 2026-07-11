"""Deep web search for building project info using Tavily."""

import re

from app.config import settings

_small_dim = re.compile(r'[-_x](\d+)[x×](\d+)[-_.]')


def search_project_info(query: str) -> str:
    """
    Search the web for project team info (developer, contractor, architect).
    Uses Tavily with advanced search depth and source relevance filtering.
    Returns formatted text or empty string on failure / missing API key.
    """
    if not settings.tavily_api_key:
        return ""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=5,
        )
        parts = []
        if response.get("answer"):
            parts.append(f"Summary: {response['answer']}")
        for r in response.get("results", []):
            if r.get("score", 0) < 0.4:
                continue
            title   = r.get("title", "")
            url     = r.get("url", "")
            content = (r.get("content") or "")[:400]
            parts.append(f"Source: {title} ({url})\n{content}")
        return "\n\n".join(parts)
    except Exception:
        return ""


def build_project_query(title: str, article_text: str) -> str:
    snippet = article_text[:200].replace("\n", " ").strip()
    return f"{title} developer contractor architect facade"


def find_linkedin_profiles(company_name: str) -> dict:
    """Search for a company's LinkedIn page. Returns up to 3 candidates."""
    if not settings.tavily_api_key:
        return {"candidates": [], "count": 0}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=f'"{company_name}" site:linkedin.com/company',
            search_depth="basic",
            max_results=5,
        )
        candidates = []
        seen_slugs: set[str] = set()
        for r in response.get("results", []):
            url = r.get("url", "")
            if "linkedin.com/company/" not in url:
                continue
            slug = url.split("linkedin.com/company/")[1].split("/")[0].split("?")[0]
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            title = r.get("title", company_name)
            for suffix in (" | LinkedIn", " - LinkedIn", "| LinkedIn", "- LinkedIn"):
                title = title.replace(suffix, "").strip()
            candidates.append({
                "title": title,
                "url": f"https://www.linkedin.com/company/{slug}",
                "description": (r.get("content") or "")[:100].strip(),
            })
            if len(candidates) >= 3:
                break
        return {"candidates": candidates, "count": len(candidates)}
    except Exception:
        return {"candidates": [], "count": 0}


# ── Image ranking for search_images ────────────────────────────────────────────
# URL substrings that mean "not a real building photo": junk assets, social CDNs,
# and (at the bottom) stock/clipart/community sites that reliably return off-topic
# images (e.g. Pixabay/PublicDomainPictures/catpedia cats). Dropped outright.
_IMG_SKIP_URL = (
    "logo", "icon", "avatar", "profile", "favicon",
    "pixel", "track", "sprite", "banner", "badge", "placeholder",
    "thumbnail", "thumb", "mini", "tiny", "small",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "licdn.com",
    "fbsbx.com", "lookaside", "gravatar.com", "wp-content/uploads/avatars",
    ".gif",
    # Stock / clipart / community image sites — high junk rate, low building relevance
    "pixabay", "publicdomainpictures", "pinterest", "pinimg", "etsy",
    "shutterstock", "dreamstime", "123rf", "freepik", "vecteezy",
    "getdrawings", "catpedia",
    # Query-string size hints from CDNs
    "w=50", "w=100", "w=150", "w=200", "w=250", "w=300",
    "width=50", "width=100", "width=150", "width=200", "width=300",
    "size=sm", "size=xs", "size=small",
    "format=thumbnail",
)

_PREFER_DOMAINS = (
    "archdaily", "dezeen", "architecturaldigest", "archello",
    "wikimedia", "wikipedia", "e-architect", "world-architects",
    "archpaper", "architizer", "architectural-review",
    "structurae", "skyscrapercity", "ctbuh", "archmarathon",
    "emporis", "archnet", "metalocus", "uncubemagazine",
)

# Tavily returns an AI-generated description per image. DENY-ONLY: an image is
# dropped only when its description positively names a non-building subject
# (cats/animals/clipart). Word boundaries avoid false hits like "cathedral" or
# "located". An image with NO description is NEVER dropped by this check.
_DESC_DENY_RE = re.compile(
    r"\b(cats?|kittens?|kitty|feline|siamese|dogs?|pupp(?:y|ies)|canine|"
    r"animals?|pets?|wildlife|bird|horse|cartoon|clip[\s-]*art)\b",
    re.IGNORECASE,
)
# Description terms that confirm an on-subject building image — rank it up.
_DESC_PREFER = (
    "building", "skyscraper", "tower", "facade", "faade", "architect",
    "skyline", "high-rise", "highrise", "observation deck", "glass",
    "structure", "construction", "rooftop", "cityscape", "city",
)


def _rank_images(items: list[tuple[str, str]]) -> list[str]:
    """Score (url, description) pairs and return up to 8 building-image URLs.

    Layers: junk/stock URL denylist; small-dimension guard; deny-only description
    filter (missing description never drops an image); and a boost for
    architecture domains and on-subject descriptions.
    """
    def _score(url: str, desc: str) -> int:
        ul = url.lower()
        if any(p in ul for p in _IMG_SKIP_URL):
            return -1
        # Filter URLs with explicit small dimensions, e.g. image-320x240.jpg
        m = _small_dim.search(ul)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            if w < 600 or h < 400:
                return -1
        d = (desc or "").lower()
        if d and _DESC_DENY_RE.search(d):
            return -1  # description names a cat/animal/etc. — not a building
        score = 0
        if any(pd in ul for pd in _PREFER_DOMAINS):
            score += 2
        if d and any(t in d for t in _DESC_PREFER):
            score += 2
        return score or 1  # no positive signal → neutral keep (never dropped)

    scored = [(u, _score(u, desc)) for (u, desc) in items if isinstance(u, str) and u.startswith("http")]
    return [u for u, s in sorted(scored, key=lambda x: -x[1]) if s >= 0][:8]


def search_images(query: str) -> list[str]:
    """Search for building/architecture images using Tavily's image search.

    Requests Tavily's per-image descriptions so `_rank_images` can drop off-topic
    results (e.g. the stock-photo cats that generic 'professional photography'
    queries sometimes surface) while keeping images that have no description.
    """
    if not settings.tavily_api_key:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            include_images=True,
            include_image_descriptions=True,
            max_results=15,
        )
        items: list[tuple[str, str]] = []
        for im in response.get("images", []):
            if isinstance(im, dict):
                items.append((im.get("url", "") or "", im.get("description") or ""))
            elif isinstance(im, str):
                items.append((im, ""))
        return _rank_images(items)
    except Exception:
        return []
