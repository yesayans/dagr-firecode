"""
Web roadmap fallback when no public GitHub backlog exists.

Searches the public web for:
  - current product capabilities (features / help / store listing)
  - announced plans / roadmaps / changelogs
  - interviews / keynotes that promise features not clearly shipped
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from silent_stakeholder import EvidenceItem, ProductContext

UA = (
    "Mozilla/5.0 (compatible; SilentStakeholder/1.0; "
    "+https://github.com/silent-stakeholder)"
)

FEATURE_HINTS = re.compile(
    r"\b(feature|supports?|you can|ability to|includes?|built[- ]in|"
    r"offline|sync|export|import|dark mode|notification|widget|"
    r"integration|api|share|backup|encrypt)\b",
    re.I,
)
PLAN_HINTS = re.compile(
    r"\b(roadmap|coming soon|planned|we('?re| are) working|"
    r"will (add|launch|introduce|ship)|upcoming|next (version|release)|"
    r"on our (list|backlog)|in development|beta)\b",
    re.I,
)
INTERVIEW_HINTS = re.compile(
    r"\b(interview|podcast|keynote|we announced|we promised|"
    r"told (reporters|users)|in an interview|our vision|"
    r"long[- ]term|eventually)\b",
    re.I,
)


def _eid(prefix: str, url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{h}"


def _search(query: str, max_results: int = 8) -> list[dict]:
    """DuckDuckGo text search via ddgs."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # type: ignore

    out: list[dict] = []
    with DDGS() as ddgs:
        for row in ddgs.text(query, max_results=max_results):
            out.append(
                {
                    "title": row.get("title") or "",
                    "url": row.get("href") or row.get("link") or "",
                    "body": row.get("body") or row.get("snippet") or "",
                }
            )
    return out


def _fetch_text(url: str, session: requests.Session, max_chars: int = 8000) -> str:
    try:
        r = session.get(url, timeout=20, headers={"User-Agent": UA})
        if r.status_code != 200 or not r.text:
            return ""
        ctype = (r.headers.get("content-type") or "").lower()
        if "html" not in ctype and "text" not in ctype:
            return ""
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:max_chars]
    except Exception:
        return ""


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if 40 <= len(p.strip()) <= 280]


def _host_ok(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    blocked = ("facebook.com", "instagram.com", "tiktok.com", "pinterest.com")
    return bool(host) and not any(b in host for b in blocked)


def _classify_and_extract(
    title: str, snippet: str, page_text: str
) -> tuple[list[str], list[str], list[str], str]:
    """Return (current, planned, promised_unshipped, primary_kind)."""
    blob = f"{title}\n{snippet}\n{page_text[:2500]}"
    current: list[str] = []
    planned: list[str] = []
    promised: list[str] = []

    primary = "other"
    if PLAN_HINTS.search(blob) or "roadmap" in title.lower():
        primary = "planned"
    elif INTERVIEW_HINTS.search(blob):
        primary = "interview_signal"
    elif FEATURE_HINTS.search(blob) or "feature" in title.lower():
        primary = "current_feature"

    for sent in _sentences(blob)[:40]:
        if PLAN_HINTS.search(sent):
            planned.append(sent)
            # Interview + plan language → promised but maybe unshipped
            if INTERVIEW_HINTS.search(blob) or INTERVIEW_HINTS.search(sent):
                promised.append(sent)
        elif FEATURE_HINTS.search(sent):
            current.append(sent)

    # Prefer title+snippet as compact bullets when page is thin
    if not current and not planned and snippet:
        if primary == "planned":
            planned.append(f"{title}: {snippet}"[:240])
        elif primary == "interview_signal":
            promised.append(f"{title}: {snippet}"[:240])
        else:
            current.append(f"{title}: {snippet}"[:240])

    return (
        _uniq(current)[:8],
        _uniq(planned)[:8],
        _uniq(promised)[:8],
        primary,
    )


def _uniq(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        key = x.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(x.strip())
    return out


def fetch_web_roadmap(
    ctx: ProductContext,
    *,
    app_name: str | None = None,
    max_queries: int = 4,
    fetch_pages: bool = True,
) -> ProductContext:
    """
    Search the web for product capabilities + plans when GitHub is missing.

    Queries cover: features, roadmap, changelog, interviews/promises.
    Results must mention the product name (or package) to reduce off-topic noise.
    """
    name = app_name or ctx.display_name or ctx.package_name or ctx.product_id
    pkg = ctx.package_name or ""
    name_tokens = [t for t in re.split(r"\W+", name.lower()) if len(t) >= 3]
    # Avoid ultra-generic single tokens like "kernel" alone when package exists
    must_match = name.lower()
    alt_match = pkg.lower() if pkg else ""

    queries = [
        f"\"{name}\" android app features",
        f"\"{name}\" (roadmap OR changelog OR \"coming soon\" OR \"feature request\")",
        f"\"{name}\" (interview OR founder OR announced OR \"we are working on\")",
        f"\"{name}\" OR \"{pkg}\" site:github.com",
    ]
    if pkg:
        queries.append(f"\"{pkg}\" features OR documentation")
    queries = queries[: max(max_queries, 4)]

    session = requests.Session()
    session.headers["User-Agent"] = UA

    hits: list[dict] = []
    for q in queries:
        try:
            hits.extend(_search(q, max_results=6))
        except Exception as e:
            ctx.notes += f" Search failed for '{q}': {e}."
        time.sleep(0.4)

    def _mentions_product(title: str, body: str, url: str) -> bool:
        blob = f"{title} {body} {url}".lower()
        if must_match and must_match in blob:
            return True
        if alt_match and alt_match in blob:
            return True
        # Multi-token names: require all significant tokens
        if len(name_tokens) >= 2 and all(t in blob for t in name_tokens[:3]):
            return True
        return False

    # Dedupe by URL + relevance filter
    seen_urls: set[str] = set()
    unique_hits: list[dict] = []
    for h in hits:
        url = (h.get("url") or "").strip()
        title = h.get("title") or ""
        body = h.get("body") or ""
        if not url or url in seen_urls or not _host_ok(url):
            continue
        if not _mentions_product(title, body, url):
            continue
        seen_urls.add(url)
        unique_hits.append(h)

        # Auto-discover GitHub repo from search hits (prefer name match)
        if "github.com/" in url.lower():
            m = re.search(r"github\.com/([^/]+)/([^/#?\s]+)", url, re.I)
            if m:
                owner, repo = m.group(1), m.group(2)
                if owner.lower() not in {"topics", "search", "orgs", "settings"}:
                    candidate = f"{owner}/{repo}"
                    repo_blob = f"{owner} {repo}".lower().replace("-", " ").replace("_", " ")
                    name_hit = any(t in repo_blob for t in name_tokens[:3]) or (
                        must_match.replace(" ", "") in repo_blob.replace(" ", "")
                    )
                    if not ctx.github_repo:
                        if name_hit or not name_tokens:
                            ctx.github_repo = candidate
                            ctx.notes += f" Discovered GitHub repo from web: {ctx.github_repo}."
                    elif name_hit and not any(
                        t in (ctx.github_repo or "").lower() for t in name_tokens[:2]
                    ):
                        ctx.github_repo = candidate
                        ctx.notes += f" Prefer name-matched GitHub repo: {ctx.github_repo}."

    for h in unique_hits[:12]:
        url = h["url"]
        title = h.get("title") or url
        snippet = h.get("body") or ""
        page = _fetch_text(url, session) if fetch_pages else ""
        # If fetched page doesn't mention product, skip extraction
        if page and not _mentions_product(title, page[:1500], url):
            continue
        current, planned, promised, kind = _classify_and_extract(title, snippet, page)

        ctx.current_features.extend(current)
        ctx.planned_items.extend(planned)
        ctx.promised_unshipped.extend(promised)

        ctx.evidence.append(
            EvidenceItem(
                id=_eid("web", url),
                source="interview" if kind == "interview_signal" else "web_page",
                title=title,
                url=url,
                snippet=(snippet or page[:240]),
                kind=kind if kind != "other" else "current_feature",
            )
        )

    ctx.current_features = _uniq(ctx.current_features)[:25]
    ctx.planned_items = _uniq(ctx.planned_items)[:25]
    ctx.promised_unshipped = _uniq(ctx.promised_unshipped)[:25]

    if ctx.evidence:
        if ctx.roadmap_source == "github":
            ctx.roadmap_source = "hybrid"
        elif ctx.roadmap_source == "none":
            ctx.roadmap_source = "web"
        ctx.notes += (
            f" Web roadmap: {len(ctx.evidence)} sources, "
            f"{len(ctx.current_features)} current, "
            f"{len(ctx.planned_items)} planned, "
            f"{len(ctx.promised_unshipped)} promised-unshipped signals."
        )
    else:
        ctx.notes += " Web search returned no usable roadmap/feature evidence."

    return ctx
