"""Anthropic tool schemas and execution dispatcher."""

from app.services.scraper import scrape
from app.services.search import search_project_info
from app.services.retrieval import get_similar_posts

TOOLS = [
    {
        "name": "scrape_url",
        "description": "Fetch an article or building page from a URL. Returns title, full body text, and image URLs. Call this first when the user provides a URL.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL to fetch"}},
            "required": ["url"]
        }
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for project credits: developer (יזם), main contractor (קבלן ראשי), "
            "architect (אדריכל), and facade/envelope contractor (קבלן מעטפת). "
            "Search twice if needed — once in Hebrew, once in English — to maximise chances of finding exact names. "
            "Only use names that appear explicitly in search results. Never invent or guess names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query, e.g. 'Tower Name City developer contractor architect facade'"}},
            "required": ["query"]
        }
    },
    {
        "name": "retrieve_similar_posts",
        "description": "Retrieve similar past LinkedIn posts from the CEO archive for writing style reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic or building type"},
                "n": {"type": "integer", "description": "Number of posts (default 5)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "find_linkedin_profiles",
        "description": (
            "Find the LinkedIn company page for the single most important company in the post "
            "(developer > main contractor > architect > facade contractor). "
            "Call this tool ONCE only — not for every company mentioned. "
            "Rules based on result:\n"
            "• count=1 → confirmed; write @CompanyName in the post\n"
            "• count=0 → no @mention; write plain company name\n"
            "• count≥2 → DO NOT write the post yet; a disambiguation UI will appear; "
            "respond briefly that profiles were found and you will write after user selects"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "Exact company name to look up on LinkedIn"}
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "search_images",
        "description": "Search the internet for high-quality photos of this specific building or architectural project. Use when the user asks for images, or to find building exterior/facade photos to supplement the article.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Building name + city + 'facade exterior architecture' (e.g. 'Azrieli Tower Tel Aviv glass facade exterior')"}},
            "required": ["query"]
        },
        "cache_control": {"type": "ephemeral"},
    },
]


def execute_tool(name: str, args: dict, db) -> dict:
    if name == "scrape_url":
        return scrape(args["url"])
    if name == "search_web":
        raw = search_project_info(args["query"])
        sources = [l[len("Source:"):].strip() for l in raw.splitlines() if l.startswith("Source:")]
        return {"results": raw, "source_count": len(sources), "sources": sources, "has_answer": bool(raw)}
    if name == "retrieve_similar_posts":
        results = get_similar_posts(db, args["query"], args.get("n", 8))
        # Format as a style guide so Claude knows how to use the examples
        sections = []
        for i, item in enumerate(results, 1):
            sim_pct = int(item["similarity"] * 100)
            sections.append(
                f"[פוסט דומה {i} — דמיון {sim_pct}%]\n{item['text']}"
            )
        formatted = (
            "להלן פוסטים אמיתיים של רומן על נושאים קרובים.\n"
            "השתמש בהם כתבנית: העתק את מבנה הפתיחה, קצב הפסקאות, האיזון טכני/שיחתי, וסגנון הסיום.\n\n"
            + "\n\n---\n\n".join(sections)
        )
        return {"style_examples": formatted, "count": len(results)}
    if name == "find_linkedin_profiles":
        from app.services.search import find_linkedin_profiles
        return find_linkedin_profiles(args["company_name"])
    if name == "search_images":
        from app.services.search import search_images
        images = search_images(args["query"])
        return {"images": images, "count": len(images)}
    return {"error": f"Unknown tool: {name}"}


def summarize_result(name: str, result: dict) -> str:
    if name == "scrape_url":
        chars = len(result.get("text", ""))
        imgs = len(result.get("images", []))
        title = result.get("title", "page")[:40]
        return f"Scraped '{title}' ({chars:,} chars, {imgs} images)"
    if name == "search_web":
        if not result.get("has_answer") or not result.get("results"):
            return "No web results found"
        count = result.get("source_count", 0)
        sources = result.get("sources", [])
        label = f"{count} sources" if count else "web summary"
        snippet = (sources[0][:60] + "…") if sources else ""
        return f"Found {label}" + (f" — {snippet}" if snippet else "")
    if name == "retrieve_similar_posts":
        n = result.get("count", 0)
        return f"Retrieved {n} similar posts for style reference"
    if name == "find_linkedin_profiles":
        count = result.get("count", 0)
        if count == 0:
            return "No LinkedIn profile found"
        if count == 1:
            return f"Found: {result['candidates'][0]['url']}"
        return f"{count} candidates — user selection needed"
    if name == "search_images":
        n = result.get("count", 0)
        return f"Found {n} building photos"
    return "Tool completed"
