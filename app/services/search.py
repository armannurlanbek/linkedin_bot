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


def search_images(query: str) -> list[str]:
    """Search for building/architecture images using Tavily's image search."""
    if not settings.tavily_api_key:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            include_images=True,
            max_results=15,
        )
        images = response.get("images", [])

        _skip_patterns = (
            "logo", "icon", "avatar", "profile", "favicon",
            "pixel", "track", "sprite", "banner", "badge", "placeholder",
            "thumbnail", "thumb", "mini", "tiny", "small",
            "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
            "gravatar.com", "wp-content/uploads/avatars",
            ".gif",
            # Query-string size hints from CDNs
            "w=50", "w=100", "w=150", "w=200", "w=250", "w=300",
            "width=50", "width=100", "width=150", "width=200", "width=300",
            "size=sm", "size=xs", "size=small",
            "format=thumbnail",
        )

        _prefer_domains = (
            "archdaily", "dezeen", "architecturaldigest", "archello",
            "wikimedia", "wikipedia", "e-architect", "world-architects",
            "archpaper", "architizer", "architectural-review",
            "structurae", "skyscrapercity", "ctbuh", "archmarathon",
            "emporis", "archnet", "metalocus", "uncubemagazine",
        )

        def _score(u: str) -> int:
            ul = u.lower()
            if any(p in ul for p in _skip_patterns):
                return -1
            # Filter URLs with explicit small dimensions, e.g. image-320x240.jpg
            m = _small_dim.search(ul)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
                if w < 600 or h < 400:
                    return -1
            if any(d in ul for d in _prefer_domains):
                return 2
            return 1

        scored = [(u, _score(u)) for u in images if isinstance(u, str) and u.startswith("http")]
        filtered = [u for u, s in sorted(scored, key=lambda x: -x[1]) if s >= 0]
        return filtered[:8]
    except Exception:
        return []
