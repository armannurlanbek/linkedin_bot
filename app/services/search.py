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
