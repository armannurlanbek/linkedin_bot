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

# `licdn`/`linkedin` filter out LinkedIn CDN assets — when a logged-out LinkedIn
# post/company URL is scraped, its og:image is the publisher's static company
# *cover*, not the post photo, so it would otherwise recur for every post from
# that page (e.g. media.licdn.com/.../company-background.../<page>_cover).
_SKIP_PATTERNS = re.compile(
    r"(logo|icon|favicon|sprite|avatar|thumb|pixel|badge|banner|placeholder|blank|gif|licdn|linkedin)",
    re.IGNORECASE,
)


def _is_photo(url: str) -> bool:
    return bool(url.startswith("http")) and not _SKIP_PATTERNS.search(url)


# ── Instagram / Facebook post handling ────────────────────────────────────────
# Share sheets append tracking params, and they make Tavily's fetch fail outright
# at basic depth. Measured against a live public post:
#   .../p/<code>                 → OK   (10,835 chars)
#   .../p/<code>/?igsh=...       → FAIL ("Error fetching content")
#   .../p/<code>/?img_index=1    → FAIL ("Error fetching content")
#   instagram.com (no www)       → OK   but only 1,412 chars
# Stripping them first removes a whole class of "sometimes it just doesn't work".
_TRACKING_PARAMS = {
    "igsh", "igshid", "img_index", "fbclid", "mibextid", "rdid", "share_url",
    "si", "ref", "source", "utm_source", "utm_medium", "utm_campaign",
}

_HOST_ALIASES = {
    "m.facebook.com": "www.facebook.com",
    "web.facebook.com": "www.facebook.com",
    "l.facebook.com": "www.facebook.com",
    "fb.com": "www.facebook.com",
    "www.fb.com": "www.facebook.com",
    "instagram.com": "www.instagram.com",
    "m.instagram.com": "www.instagram.com",
}


def _normalize_social_url(url: str) -> str:
    """Canonicalise an Instagram/Facebook URL before handing it to Tavily."""
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    try:
        parts = urlparse(url)
    except Exception:
        return url
    if not parts.netloc:
        return url  # not a real URL — hand it back untouched rather than mangle it
    host = parts.netloc.lower()
    host = _HOST_ALIASES.get(host, host)
    kept = [(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in _TRACKING_PARAMS]
    return urlunparse(
        (parts.scheme or "https", host, parts.path, parts.params, urlencode(kept), "")
    )


# Instagram serves the poster's avatar from the t51.2885-19 bucket and the post's
# own photos from t51.82787-15 / t51.2885-15. The avatar arrives FIRST in Tavily's
# list, and agent.py uses images[0] as the chat thumbnail — so without this the
# sidebar would show the account's profile picture instead of the building.
_IG_AVATAR_BUCKET = "t51.2885-19"

# lookaside.instagram.com/seo/google_widget/crawler/ is Instagram's crawler
# endpoint; it surfaces unrelated media and is not the post's photo.
_SOCIAL_IMG_SKIP = ("lookaside.", "/seo/google_widget/", "profile_pic", "/rsrc.php/")

# Instagram/Facebook encode the rendered size in the stp param, e.g.
# "dst-jpg_s150x150_tt6" (an avatar) or "dst-jpg_e35_s640x640_tt6" (a real post
# photo). Match the DIMENSIONS, not the surrounding letters — "e35" is a standard
# transform code, so a substring rule on "_e35_s" silently discards every 640x640
# photo. A URL with no size marker is full-size and is always kept.
_SOCIAL_SIZE_RE = re.compile(r"[_-][sp](\d{2,4})x(\d{2,4})")
_MIN_SOCIAL_DIM = 500


def _is_small_social_image(url: str) -> bool:
    match = _SOCIAL_SIZE_RE.search(url)
    if not match:
        return False
    return int(match.group(1)) < _MIN_SOCIAL_DIM or int(match.group(2)) < _MIN_SOCIAL_DIM


def _social_images(raw: list | None, limit: int = 12) -> list[str]:
    """Pick the post's own photos out of Tavily's image list, in order."""
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, dict):
            candidate = item.get("url") or ""
        else:
            continue
        if not candidate.startswith("http") or candidate in seen:
            continue
        low = candidate.lower()
        if _IG_AVATAR_BUCKET in low:
            continue
        if any(s in low for s in _SOCIAL_IMG_SKIP):
            continue
        if _is_small_social_image(low):
            continue
        seen.add(candidate)
        out.append(candidate)
        if len(out) >= limit:
            break
    return out


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


def _tavily_extract(url: str, *, depth: str = "basic", with_images: bool = False) -> dict | None:
    """Use Tavily's extract API to get content from JS-heavy or login-walled pages.

    depth="advanced" is markedly more capable on social permalinks (measured: it
    rescues URLs that fail outright at basic depth, and returned 8,886 chars where
    basic returned 1,412). with_images asks Tavily for the page's images — without
    it the response carries none at all, which is why social posts never had any.
    """
    if not settings.tavily_api_key:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        kwargs: dict = {"extract_depth": depth}
        if with_images:
            kwargs["include_images"] = True
        resp = client.extract(urls=[url], **kwargs)
        results = resp.get("results", [])
        if not results:
            return None
        r = results[0]
        raw_content = r.get("raw_content") or r.get("content") or ""
        if not raw_content:
            return None
        images = _social_images(r.get("images")) if with_images else []
        if "<html" in raw_content[:200].lower() or "<body" in raw_content[:200].lower():
            parsed = _parse_html(raw_content, url)
            if images and not parsed["images"]:
                parsed["images"] = images
            return parsed
        return {
            "title": "",
            "text": raw_content.strip(),
            "images": images,
            "url": url,
        }
    except Exception:
        return None


def _tavily_search(url: str) -> dict | None:
    """Search Tavily's index for the URL — hits cached content, more resilient than extract."""
    if not settings.tavily_api_key:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(query=url, max_results=3, include_raw_content=True)
        results = resp.get("results", [])
        if not results:
            return None
        # Prefer a result whose URL closely matches the original
        best = None
        for r in results:
            content = r.get("raw_content") or r.get("content") or ""
            if len(content) >= 50:
                best = r
                break
        if not best:
            return None
        content = best.get("raw_content") or best.get("content") or ""
        if "<html" in content[:200].lower() or "<body" in content[:200].lower():
            return _parse_html(content, url)
        return {
            "title": best.get("title", ""),
            # Deliberately no images: the search index returns photos belonging to
            # OTHER results for the same query (measured: a Facebook post query
            # returned lookaside.instagram.com crawler URLs from unrelated pages),
            # so they cannot be trusted to be this post's own photos.
            "text": content.strip(),
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

_PLATFORM_NAMES = {
    "instagram.com": "Instagram",
    "facebook.com": "Facebook",
    "fb.com": "Facebook",
    "tiktok.com": "TikTok",
    "twitter.com": "X",
    "x.com": "X",
}


def _platform_name(host: str) -> str:
    for domain, label in _PLATFORM_NAMES.items():
        if host == domain or host.endswith("." + domain):
            return label
    return "social media"


def _is_login_wall(final_url: str) -> bool:
    """True if the response redirected to a login/auth page."""
    final = final_url.lower()
    return any(p in final for p in _LOGIN_WALL_PATTERNS)


def scrape(url: str) -> dict:
    """Return {title, text, images, url}. Uses Tavily extract for social media and login-walled pages."""
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc.lower()
    host = netloc[4:] if netloc.startswith("www.") else netloc
    is_social = any(host == d or host.endswith("." + d) for d in _LOGIN_WALL_DOMAINS)

    # Social platforms: go straight to Tavily (httpx always gets login walls)
    if is_social:
        target = _normalize_social_url(url)

        # Three attempts, each a DIFFERENT strategy rather than the same call
        # repeated. Repeating an identical failing extract was measured at 3
        # failures out of 3 attempts, ~6.7s each — blind retries buy only latency.
        # These three fail independently, so a later one can rescue an earlier one.
        for depth, with_images in (("advanced", True), ("basic", False)):
            tavily_result = _tavily_extract(target, depth=depth, with_images=with_images)
            if tavily_result and len(tavily_result.get("text", "")) >= 50:
                return tavily_result

        search_result = _tavily_search(target)
        if search_result and len(search_result.get("text", "")) >= 50:
            return search_result

        return {
            "title": "",
            "text": (
                f"Could not read this {_platform_name(host)} post. It may be private or "
                f"deleted, or {_platform_name(host)} is currently blocking automated "
                "access (this often clears up after a few minutes). Do NOT guess or "
                "invent what the post said. Tell the user the post could not be read, "
                "and ask them to paste its text into the chat (and attach the photo) "
                "so you can write from that."
            ),
            "images": [],
            "url": url,
        }

    # All other URLs: try httpx first
    result = None
    try:
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=20) as client:
            resp = client.get(url)
            resp.raise_for_status()
            final_url = str(resp.url)
        # Redirect to a login wall — fall through to Tavily
        if _is_login_wall(final_url):
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

    # Last resort: Tavily search index
    search_result = _tavily_search(url)
    if search_result and len(search_result.get("text", "")) >= 50:
        return search_result

    if result is not None:
        return result
    return {"title": "", "text": "", "images": [], "url": url}
