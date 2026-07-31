"""Roadmap resolution: local cache → GitHub → web → none."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings, get_settings
from src.data_ingestion import GitHubScraper, WebRoadmapScraper, _mentions_product, _product_tokens

logger = logging.getLogger(__name__)


@dataclass
class ResolveResult:
    roadmap_source: str
    github_repo: str | None
    web_urls: list[str] | None
    roadmap_items: pd.DataFrame
    degraded: list[str]
    notes: str = ""

    def items_as_records(self) -> list[dict[str, Any]]:
        if self.roadmap_items is None or self.roadmap_items.empty:
            return []
        return self.roadmap_items.to_dict(orient="records")


class RoadmapResolver:
    """
    Wrap silent_stakeholder helpers + api scrapers.
    Order: local data/roadmaps/{package}.json (unless refresh) → GitHub → web → none.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.gh = GitHubScraper(self.settings)
        self.web = WebRoadmapScraper(self.settings)
        self.cache_dir = self.settings.data_dir / "roadmaps"

    def resolve(
        self,
        app_name: str,
        package_name: str,
        github_repo: str | None = None,
        refresh: bool = False,
        *,
        external_roadmap_urls: list[str] | None = None,
        external_roadmap_text: str | None = None,
    ) -> ResolveResult:
        degraded: list[str] = []
        notes_parts: list[str] = []

        # User-supplied roadmap (closed-source / non-GitHub) wins over discovery.
        external = self._resolve_external(
            app_name=app_name,
            package_name=package_name,
            urls=external_roadmap_urls or [],
            text=external_roadmap_text or "",
            degraded=degraded,
        )
        if external is not None:
            return external

        # Fast-path repo guess
        repo = github_repo or self._guess_repo(package_name, app_name, degraded)

        cache_path = self.cache_dir / f"{package_name}.json"
        if cache_path.exists() and not refresh:
            cached = self._load_cache(cache_path, app_name, package_name)
            if cached is not None:
                source, items, urls, cache_notes, cache_deg = cached
                degraded.extend(cache_deg)
                # Prefer live GitHub when token present and repo known
                live_repo = repo or self._repo_from_cache_file(cache_path)
                if live_repo and self.settings.github_token_present:
                    gh_items, gh_deg = self.gh.fetch_issues_and_milestones(live_repo)
                    degraded.extend(gh_deg)
                    if not gh_items.empty:
                        return ResolveResult(
                            roadmap_source="github",
                            github_repo=live_repo,
                            web_urls=urls,
                            roadmap_items=gh_items.rename(
                                columns={"issue_id": "item_id", "text": "text"}
                            ).assign(source="github"),
                            degraded=degraded,
                            notes="live GitHub over cache",
                        )
                    degraded.append("GitHub live fetch empty; using filtered cache")
                elif live_repo and not self.settings.github_token_present:
                    degraded.append("GitHub unauthenticated; using roadmap cache")

                if items.empty:
                    return ResolveResult(
                        roadmap_source="none",
                        github_repo=live_repo,
                        web_urls=urls,
                        roadmap_items=items,
                        degraded=degraded + ["cache had no relevant roadmap items"],
                        notes=cache_notes,
                    )
                # If cache claimed github but items are from web evidence, keep source
                return ResolveResult(
                    roadmap_source=source if source != "none" else ("web" if not items.empty else "none"),
                    github_repo=live_repo,
                    web_urls=urls,
                    roadmap_items=items,
                    degraded=degraded,
                    notes=cache_notes,
                )

        # No usable cache — try GitHub then web
        if repo and self.settings.github_token_present:
            gh_items, gh_deg = self.gh.fetch_issues_and_milestones(repo)
            degraded.extend(gh_deg)
            if not gh_items.empty:
                return ResolveResult(
                    roadmap_source="github",
                    github_repo=repo,
                    web_urls=None,
                    roadmap_items=gh_items.rename(
                        columns={"issue_id": "item_id"}
                    ).assign(source="github"),
                    degraded=degraded,
                    notes="github live",
                )
        elif repo and not self.settings.github_token_present:
            degraded.append("GitHub unauthenticated")

        # Web fallback via silent_stakeholder search + relevance-gated page fetch
        web_urls: list[str] = []
        try:
            from silent_stakeholder.roadmap_web import fetch_web_roadmap
            from silent_stakeholder.schema import ProductContext

            ctx = ProductContext(
                product_id=package_name,
                display_name=app_name,
                package_name=package_name,
                github_repo=repo,
            )
            ctx = fetch_web_roadmap(ctx, app_name=app_name, fetch_pages=False)
            if ctx.github_repo and not repo:
                repo = ctx.github_repo
                notes_parts.append(f"discovered repo {repo}")
            web_urls = [e.url for e in ctx.evidence if e.url]
        except Exception as e:
            degraded.append(f"web search failed: {e}")

        if web_urls:
            pages, web_deg = self.web.fetch_roadmap_pages(
                web_urls,
                product_name=app_name,
                package_name=package_name,
                github_repo=repo,
            )
            degraded.extend(web_deg)
            items = self._pages_to_items(pages, app_name, package_name)
            if not items.empty:
                source = "hybrid" if repo else "web"
                return ResolveResult(
                    roadmap_source=source,
                    github_repo=repo,
                    web_urls=web_urls,
                    roadmap_items=items,
                    degraded=degraded,
                    notes="; ".join(notes_parts) or "web gated",
                )

        return ResolveResult(
            roadmap_source="none",
            github_repo=repo,
            web_urls=web_urls or None,
            roadmap_items=_empty_items(),
            degraded=degraded,
            notes="; ".join(notes_parts) or "no roadmap",
        )

    def _resolve_external(
        self,
        *,
        app_name: str,
        package_name: str,
        urls: list[str],
        text: str,
        degraded: list[str],
    ) -> ResolveResult | None:
        """Build roadmap items from user URLs and/or pasted text → roadmap_source=web."""
        cleaned_urls = _split_urls(urls if isinstance(urls, list) else [str(urls)])
        paste_items = _paste_text_to_items(text)
        page_items = _empty_items()
        if cleaned_urls:
            pages, web_deg = self.web.fetch_roadmap_pages(
                cleaned_urls,
                product_name=app_name,
                package_name=package_name,
                github_repo=None,
            )
            degraded.extend(web_deg)
            page_items = self._pages_to_items(pages, app_name, package_name)
            if page_items.empty:
                degraded.append(
                    "external roadmap URLs yielded no extractable text; "
                    "using pasted items if any"
                )

        frames = [f for f in (page_items, paste_items) if f is not None and not f.empty]
        if not frames:
            if cleaned_urls or (text or "").strip():
                degraded.append("external roadmap provided but produced zero items")
            return None

        items = pd.concat(frames, ignore_index=True)
        notes = []
        if cleaned_urls:
            notes.append(f"{len(cleaned_urls)} user URL(s)")
        if not paste_items.empty:
            notes.append(f"{len(paste_items)} pasted item(s)")
        return ResolveResult(
            roadmap_source="web",
            github_repo=None,
            web_urls=cleaned_urls or None,
            roadmap_items=items,
            degraded=degraded,
            notes="external roadmap: " + ", ".join(notes),
        )

    def _guess_repo(
        self, package_name: str, app_name: str, degraded: list[str]
    ) -> str | None:
        from silent_stakeholder.roadmap_github import guess_repo

        repo = guess_repo(package_name) or guess_repo(app_name)
        if repo:
            return repo
        # GitHub search API fallback
        q = f"{app_name} android in:name,description"
        found, deg = self.gh.search_repo(q)
        degraded.extend(deg)
        if found:
            return found
        if package_name:
            found2, deg2 = self.gh.search_repo(f"{package_name} in:readme")
            degraded.extend(deg2)
            return found2
        return None

    def _repo_from_cache_file(self, path: Path) -> str | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("github_repo")
        except Exception:
            return None

    def _load_cache(
        self, path: Path, app_name: str, package_name: str
    ) -> tuple[str, pd.DataFrame, list[str] | None, str, list[str]] | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return ("none", _empty_items(), None, f"cache read error: {e}", [str(e)])

        tokens = _product_tokens(app_name, package_name)
        source = data.get("roadmap_source") or "none"
        notes = data.get("notes") or ""
        degraded: list[str] = []
        if "403" in notes:
            degraded.append("cached roadmap notes mention GitHub 403")

        urls = [e.get("url") for e in (data.get("evidence") or []) if e.get("url")]
        rows: list[dict[str, Any]] = []

        def _accept(text: str, title: str = "", url: str = "") -> bool:
            return _mentions_product(title, text, url, tokens, app_name, package_name)

        # Planned / promised first
        for i, text in enumerate(data.get("planned_items") or []):
            if not _accept(text):
                continue
            rows.append(
                {
                    "item_id": f"cache-planned-{i}",
                    "text": text,
                    "labels": "",
                    "milestone_title": None,
                    "state": "open",
                    "age_days": None,
                    "closed_at": None,
                    "url": "",
                    "updated_at": None,
                    "created_at": None,
                    "kind": "planned",
                    "source": "cache",
                }
            )
        for i, text in enumerate(data.get("promised_unshipped") or []):
            if not _accept(text):
                continue
            rows.append(
                {
                    "item_id": f"cache-promised-{i}",
                    "text": text,
                    "labels": "",
                    "milestone_title": None,
                    "state": "open",
                    "age_days": None,
                    "closed_at": None,
                    "url": "",
                    "updated_at": None,
                    "created_at": None,
                    "kind": "promised",
                    "source": "cache",
                }
            )
        # Evidence as roadmap signals (title+snippet), relevance gated
        for e in data.get("evidence") or []:
            title = e.get("title") or ""
            snippet = e.get("snippet") or ""
            url = e.get("url") or ""
            blob = f"{title}. {snippet}".strip()
            if not blob or not _accept(blob, title, url):
                continue
            # Drop known-garbage hosts that slipped into old cache
            if any(
                bad in url.lower()
                for bad in (
                    "cleantechnica.com",
                    "cartelinsider.com",
                    "thespinoff.co.nz",
                    "flying-phoenix.dev",
                    "podcasts.apple.com/us/podcast/saas",
                )
            ):
                continue
            kind = e.get("kind") or "other"
            state = "closed" if kind == "current_feature" else "open"
            rows.append(
                {
                    "item_id": e.get("id") or f"cache-ev-{len(rows)}",
                    "text": blob[:800],
                    "labels": kind,
                    "milestone_title": None,
                    "state": state,
                    "age_days": None,
                    "closed_at": None,
                    "url": url,
                    "updated_at": None,
                    "created_at": None,
                    "kind": kind,
                    "source": "cache",
                }
            )

        # Current features as shipped items (for MISUNDERSTOOD matching)
        for i, text in enumerate(data.get("current_features") or []):
            if not _accept(text):
                continue
            # Skip obvious junk / nav chrome
            if len(text) < 40 or text.count("-") > 12:
                continue
            rows.append(
                {
                    "item_id": f"cache-feature-{i}",
                    "text": text[:800],
                    "labels": "shipped",
                    "milestone_title": "shipped",
                    "state": "closed",
                    "age_days": 400,
                    "closed_at": "2020-01-01T00:00:00Z",
                    "url": "",
                    "updated_at": "2020-01-01T00:00:00Z",
                    "created_at": None,
                    "kind": "current_feature",
                    "source": "cache",
                }
            )

        if not rows:
            degraded.append("filtered cache produced zero relevant items")
            items = _empty_items()
            # Downgrade source when nothing relevant remains
            return ("none", items, urls or None, notes, degraded)

        # Deduplicate by normalised text
        seen: set[str] = set()
        uniq = []
        for r in rows:
            key = re.sub(r"\s+", " ", r["text"].lower())[:200]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        items = pd.DataFrame(uniq)
        # If original source was web/github but we only have filtered cache, keep web/github
        if source == "none":
            source = "web"
        return (source, items, urls or None, notes, degraded)

    def _pages_to_items(
        self, pages: pd.DataFrame, app_name: str, package_name: str
    ) -> pd.DataFrame:
        if pages is None or pages.empty:
            return _empty_items()
        rows = []
        for i, row in pages.iterrows():
            text = str(row.get("text") or "")
            url = str(row.get("source_url") or "")
            title = str(row.get("title") or "")
            if not text:
                continue
            state = "open"
            if re.search(r"\b(changelog|released?|shipped)\b", title + text, re.I):
                state = "closed"
            rows.append(
                {
                    "item_id": f"web-{i}",
                    "text": f"{title}. {text}"[:1000],
                    "labels": "",
                    "milestone_title": None,
                    "state": state,
                    "age_days": None,
                    "closed_at": None if state == "open" else "2020-01-01T00:00:00Z",
                    "url": url,
                    "updated_at": None,
                    "created_at": None,
                    "kind": "web_page",
                    "source": "web",
                }
            )
        return pd.DataFrame(rows) if rows else _empty_items()


def _empty_items() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "item_id",
            "text",
            "labels",
            "milestone_title",
            "state",
            "age_days",
            "closed_at",
            "url",
            "updated_at",
            "created_at",
            "kind",
            "source",
        ]
    )


def _split_urls(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        for part in re.split(r"[\s,]+", (raw or "").strip()):
            part = part.strip()
            if not part:
                continue
            if not re.match(r"^https?://", part, re.I):
                continue
            if part in seen:
                continue
            seen.add(part)
            out.append(part)
    return out


def _paste_text_to_items(text: str) -> pd.DataFrame:
    """One roadmap item per non-empty line (or blank-line-separated paragraph)."""
    blob = (text or "").strip()
    if not blob:
        return _empty_items()
    # Prefer paragraphs when blank lines present; else one item per line.
    if "\n\n" in blob:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", blob) if c.strip()]
    else:
        chunks = [c.strip() for c in blob.splitlines() if c.strip()]
    rows = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < 8:
            continue
        rows.append(
            {
                "item_id": f"paste-{i}",
                "text": chunk[:1000],
                "labels": "",
                "milestone_title": None,
                "state": "open",
                "age_days": None,
                "closed_at": None,
                "url": "",
                "updated_at": None,
                "created_at": None,
                "kind": "planned",
                "source": "web",
            }
        )
    return pd.DataFrame(rows) if rows else _empty_items()


def main() -> None:
    r = RoadmapResolver()
    result = r.resolve("AntennaPod", "de.danoeh.antennapod", refresh=False)
    print(
        {
            "source": result.roadmap_source,
            "repo": result.github_repo,
            "items": len(result.roadmap_items),
            "degraded": result.degraded,
            "sample": result.items_as_records()[:2],
        }
    )


if __name__ == "__main__":
    main()
