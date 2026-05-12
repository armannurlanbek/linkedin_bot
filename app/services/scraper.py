"""Fetch an article URL and extract text + image URLs."""

import re
import httpx
from bs4 import BeautifulSoup

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


def scrape(url: str) -> dict:
    """Return {title, text, images, url}. Raises on HTTP error."""
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
        resp = client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title = ""
    if soup.title:
        title = soup.title.string or ""
    for prop in ("og:title", "twitter:title"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            title = tag["content"]
            break

    # Remove nav/footer/script noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Main text — prefer <article>, fall back to <main>, then <body>
    container = soup.find("article") or soup.find("main") or soup.body
    paragraphs = container.find_all(["p", "h1", "h2", "h3", "li"]) if container else []
    text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs if p.get_text(strip=True))

    # Images — collect from multiple sources, deduplicate
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
