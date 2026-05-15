"""Fetch an article URL and extract text + image URLs."""

import re
import httpx
from bs4 import BeautifulSoup

from app.config import settings

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

_SKIP_PATTERNS = re.compile(
    r"(logo|icon|favicon|sprite|avatar|thumb|pixel|badge|banner|placeholder|blank|gif)",
    re.IGNORECASE,
)


def _is_photo(url: str) -> bool:
    return bool(url.startswith("http")) and not _SKIP_PATTERNS.search(url)


def _parse_html(html: str, url: str) -> dict:
    """Parse raw HTML into {title, text, images, url}."""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title:
        title = soup.title.string or ""
    for prop in ("og:title", "twitter:title"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            title = tag["content"]
            break

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    container = soup.find("article") or soup.find("main") or soup.body
    paragraphs = container.find_all(["p", "h1", "h2", "h3", "li"]) if container else []
    text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True))

    seen: set[str] = set()
    images: list[str] = []

    def _add(src: str) -> None:
        if src and src not in seen and _is_photo(src):
            seen.add(src)
            images.append(src)

    for prop in ("og:image", "og:image:secure_url", "twitter:image"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            _add(tag["content"])

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-original"):
            val = img.get(attr, "")
            if val:
                _add(val)

    return {"title": title.strip(), "text": text.strip(), "images": images, "url": url}


def _tavily_extract(url: str) -> dict | None:
    """Use Tavily's extract API to get content from JS-heavy or login-walled pages."""
    if not settings.tavily_api_key:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.extract(urls=[url])
        results = resp.get("results", [])
        if not results:
            return None
        r = results[0]
        raw_content = r.get("raw_content") or r.get("content") or ""
        if not raw_content:
            return None
        # raw_content may be HTML or plain text
        if "<html" in raw_content[:200].lower() or "<body" in raw_content[:200].lower():
            return _parse_html(raw_content, url)
        # Plain text response
        return {
            "title": "",
            "text": raw_content.strip(),
            "images": [],
            "url": url,
        }
    except Exception:
        return None


_LOGIN_WALL_DOMAINS = (
    "facebook.com", "fb.com", "instagram.com",
    "tiktok.com", "twitter.com", "x.com",
)

_LOGIN_WALL_PATTERNS = ("login", "signin", "sign-in", "auth/", "checkpoint")


def _is_login_wall(original_url: str, final_url: str) -> bool:
    """True if the response redirected to a login/auth page."""
    final = final_url.lower()
    return any(p in final for p in _LOGIN_WALL_PATTERNS)


def scrape(url: str) -> dict:
    """Return {title, text, images, url}. Uses Tavily extract for social media and login-walled pages."""
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower().lstrip("www.")
    is_social = any(host == d or host.endswith("." + d) for d in _LOGIN_WALL_DOMAINS)

    # Social platforms: go straight to Tavily (httpx always gets login walls)
    if is_social:
        tavily_result = _tavily_extract(url)
        if tavily_result and len(tavily_result.get("text", "")) >= 50:
            return tavily_result
        return {"title": "", "text": "Could not extract content — the post may be private or require login.", "images": [], "url": url}

    # All other URLs: try httpx first
    result = None
    try:
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
            resp = client.get(url)
            resp.raise_for_status()
            final_url = str(resp.url)
        # Redirect to a login wall — fall through to Tavily
        if _is_login_wall(url, final_url):
            result = None
        else:
            result = _parse_html(resp.text, url)
            if len(result["text"]) >= 200:
                return result
    except Exception:
        result = None

    # Fallback: Tavily extract for JS-rendered or partially blocked pages
    tavily_result = _tavily_extract(url)
    if tavily_result and len(tavily_result.get("text", "")) >= 50:
        return tavily_result

    if result is not None:
        return result
    return {"title": "", "text": "", "images": [], "url": url}
