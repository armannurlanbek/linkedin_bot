"""Deep web search for building project info using Tavily."""

from app.config import settings


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
            max_results=10,
        )
        images = response.get("images", [])

        # Filter out non-architectural images: logos, avatars, icons, tracking pixels, social CDNs
        _skip_patterns = (
            "logo", "icon", "avatar", "profile", "thumb", "favicon",
            "pixel", "track", "sprite", "banner", "badge",
            "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
            "gravatar.com", "wp-content/uploads/avatars",
            ".gif",  # usually animated logos/banners
        )
        # Known architecture/real-estate photo sources get priority
        _prefer_domains = (
            "archdaily", "dezeen", "architecturaldigest", "archello",
            "wikimedia", "wikipedia", "buildipedia", "e-architect",
            "archinet", "world-architects", "archpaper",
        )

        def _score(u: str) -> int:
            ul = u.lower()
            if any(p in ul for p in _skip_patterns):
                return -1
            if any(d in ul for d in _prefer_domains):
                return 2
            return 1

        scored = [(u, _score(u)) for u in images if isinstance(u, str) and u.startswith("http")]
        filtered = [u for u, s in sorted(scored, key=lambda x: -x[1]) if s >= 0]
        return filtered[:8]
    except Exception:
        return []
