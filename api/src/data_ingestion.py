"""Review, GitHub, and web roadmap ingestion with offline fallbacks."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

ReviewProvenance = Literal["hf", "parquet_cache", "fixture", "csv_upload"]

TEXT_COLUMN_ALIASES = (
    "review_text",
    "review",
    "text",
    "body",
    "content",
    "comment",
    "comments",
)
RATING_COLUMN_ALIASES = ("rating", "star", "stars", "score", "star_rating")
DATE_COLUMN_ALIASES = (
    "created_at",
    "date",
    "review_date",
    "at",
    "time",
    "timestamp",
    "submitted_at",
)


@dataclass
class ReviewFetchResult:
    df: pd.DataFrame
    degraded: list[str]
    provenance: ReviewProvenance


@dataclass
class CsvIngestResult:
    df: pd.DataFrame
    column_mapping: dict[str, str | None]
    warnings: list[str]
    rows_raw: int
    rows_kept: int

UA = "dagr/1.0 (+https://github.com/silent-stakeholder)"

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BOILERPLATE_HEADER_LINE = re.compile(
    r"(?im)^###\s*("
    r"Checklist|Environment|App version|Device model|Android version|"
    r"AntennaPod version|How to reproduce|Steps to reproduce"
    r")\b.*$"
)
_CHECKBOX_LINE = re.compile(r"(?m)^\s*[-*]\s*\[[ xX]\]\s*.*$")
_SEARCH_FN_LINE = re.compile(
    r"(?im)^\s*[-*]?\s*\[[ xX]?\]?\s*I have used the search function.*$"
)
_DOCS_LINE = re.compile(
    r"(?im)^\s*[-*]?\s*\[[ xX]?\]?\s*I have read (the )?(relevant )?(documentation|wiki).*$"
)
_ENV_FIELD_LINE = re.compile(
    r"(?im)^\s*("
    r"Android version|AntennaPod version|App version|Device model|"
    r"Device|OS version|Free space"
    r")\s*:.*$"
)
_WS = re.compile(r"\s+")
# Release milestones named like "1.5.2" / "v3.0" carry no thematic signal for matching
_VERSION_MILESTONE_TITLE = re.compile(
    r"^v?\d+(?:\.\d+)+(?:[-._]?[a-z0-9]+)?$", re.I
)


def is_version_milestone_title(title: str) -> bool:
    return bool(_VERSION_MILESTONE_TITLE.match((title or "").strip()))


def strip_github_boilerplate(text: str) -> str:
    """Remove checklist / environment template noise that dilutes every issue vector."""
    if not text:
        return ""
    out = _HTML_COMMENT.sub(" ", text)
    # Drop template headers/lines but keep surrounding prose (never swallow EOF)
    out = _BOILERPLATE_HEADER_LINE.sub(" ", out)
    out = _SEARCH_FN_LINE.sub(" ", out)
    out = _DOCS_LINE.sub(" ", out)
    out = _CHECKBOX_LINE.sub(" ", out)
    out = _ENV_FIELD_LINE.sub(" ", out)
    return _WS.sub(" ", out).strip()


def compose_issue_text(title: str, body: str = "", *, title_weight: int = 3) -> str:
    """Title-heavy embed text: title repeated, then cleaned body."""
    title = (title or "").strip()
    body_clean = strip_github_boilerplate(body or "")
    if not title and not body_clean:
        return ""
    if not title:
        return body_clean
    weighted = ". ".join([title] * max(1, title_weight))
    if body_clean:
        return f"{weighted}. {body_clean}"
    return weighted


WELL_KNOWN_CHANGELOG_HOSTS = frozenset(
    {
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "changelog.com",
        "keepachangelog.com",
        "medium.com",
        "dev.to",
        "hashnode.dev",
        "substack.com",
        "wordpress.com",
        "blogspot.com",
        "f-droid.org",
        "play.google.com",
        "apps.apple.com",
        "en.wikipedia.org",
        "reddit.com",
        "www.reddit.com",
    }
)

PLAN_HINTS = re.compile(
    r"\b(roadmap|coming soon|planned|we('?re| are) working|"
    r"will (add|launch|introduce|ship)|upcoming|next (version|release)|"
    r"on our (list|backlog)|in development|beta|milestone)\b",
    re.I,
)


def _stable_review_id(package_name: str, text: str, rating: Any, created_at: Any) -> str:
    raw = f"{package_name}|{rating}|{created_at}|{text.strip().lower()}"
    # Non-cryptographic stable id for dedupe keys (not a security boundary).
    return hashlib.sha1(
        raw.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:16]


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


class ReviewScraper:
    """Load app reviews from local parquet cache or HuggingFace datasets."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = self.settings.data_dir / "reviews"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, package_name: str) -> Path:
        safe = package_name.replace("/", "_")
        return self.cache_dir / f"{safe}.parquet"

    def meta_path(self, package_name: str) -> Path:
        return self.cache_path(package_name).with_suffix(".meta.json")

    def fetch_reviews(
        self,
        package_name: str,
        max_reviews: int | None = None,
        *,
        force_refresh: bool = False,
    ) -> ReviewFetchResult:
        """
        Cache-first load of reviews. Provenance is always recorded so synthetic
        fixtures can never silently masquerade as real demo data.
        """
        max_reviews = max_reviews or self.settings.max_reviews
        degraded: list[str] = []
        path = self.cache_path(package_name)

        if path.exists() and not force_refresh:
            df = self._finalize(pd.read_parquet(path), max_reviews)
            provenance = self._read_provenance(package_name, path)
            if provenance == "fixture":
                degraded.append(
                    "review provenance=fixture (test data — not for demo attribution)"
                )
            return ReviewFetchResult(df=df, degraded=degraded, provenance=provenance)

        try:
            df = self._load_from_hf(package_name, max_reviews)
            if df.empty:
                degraded.append(f"no reviews in HF for {package_name}")
            else:
                self._write_cache(package_name, df, provenance="hf")
            return ReviewFetchResult(df=df, degraded=degraded, provenance="hf")
        except Exception as e:
            if path.exists():
                degraded.append(f"HF fetch failed ({e}); using parquet cache")
                df = self._finalize(pd.read_parquet(path), max_reviews)
                provenance = self._read_provenance(package_name, path)
                return ReviewFetchResult(
                    df=df, degraded=degraded, provenance=provenance
                )
            degraded.append(f"HF fetch failed and no cache: {e}")
            return ReviewFetchResult(
                df=self._empty(), degraded=degraded, provenance="hf"
            )

    def _write_cache(
        self, package_name: str, df: pd.DataFrame, *, provenance: ReviewProvenance
    ) -> None:
        path = self.cache_path(package_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        meta = {
            "provenance": provenance,
            "package_name": package_name,
            "rows": int(len(df)),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "dataset": self.settings.hf_dataset if provenance == "hf" else None,
        }
        self.meta_path(package_name).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def _read_provenance(
        self, package_name: str, path: Path
    ) -> ReviewProvenance:
        # Paths under tests/fixtures are always fixtures
        parts = {p.lower() for p in path.parts}
        if "fixtures" in parts or "tests" in parts:
            return "fixture"
        meta_p = self.meta_path(package_name)
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                prov = meta.get("provenance")
                if prov in ("hf", "parquet_cache", "fixture", "csv_upload"):
                    return prov  # type: ignore[return-value]
            except Exception:
                pass
        # Legacy cache without sidecar — treat as real parquet cache, never fixture
        return "parquet_cache"

    def ingest_csv(
        self,
        package_name: str,
        file_bytes: bytes,
        *,
        max_reviews: int | None = None,
        filename: str | None = None,
    ) -> CsvIngestResult:
        """Parse a flexible review CSV, filter/dedupe, and write the parquet cache."""
        max_reviews = max_reviews or self.settings.max_reviews
        parsed = parse_reviews_csv(file_bytes, filename=filename)
        if parsed.df.empty:
            raise ValueError(
                "No usable reviews after filtering. Need a text column with "
                "reviews of at least 10 words (5-star reviews are dropped)."
            )
        filtered = self._filter_dedupe(parsed.df, package_name, max_reviews)
        if filtered.empty:
            raise ValueError(
                "No reviews remained after dropping short texts, 5-star ratings, "
                "and duplicates."
            )
        self._write_cache(package_name, filtered, provenance="csv_upload")
        warnings = list(parsed.warnings)
        warnings.append(f"kept {len(filtered)} of {parsed.rows_raw} uploaded rows")
        return CsvIngestResult(
            df=filtered,
            column_mapping=parsed.column_mapping,
            warnings=warnings,
            rows_raw=parsed.rows_raw,
            rows_kept=len(filtered),
        )

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["review_id", "review_text", "rating", "created_at"]
        )

    def _finalize(self, df: pd.DataFrame, max_reviews: int) -> pd.DataFrame:
        if df is None or df.empty:
            return self._empty()
        out = df.copy()
        cols = {c.lower(): c for c in out.columns}
        rename = {}
        for want in ("review_id", "review_text", "rating", "created_at"):
            if want not in out.columns and want in cols:
                rename[cols[want]] = want
        if rename:
            out = out.rename(columns=rename)
        for c in ("review_id", "review_text", "rating", "created_at"):
            if c not in out.columns:
                out[c] = None
        out = out[["review_id", "review_text", "rating", "created_at"]]
        return out.head(max_reviews).reset_index(drop=True)

    def _load_from_hf(self, package_name: str, max_reviews: int) -> pd.DataFrame:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError(
                f"HuggingFace datasets unavailable ({e}). "
                "Install api/requirements.txt and ensure no broken torch install "
                "is present in the venv."
            ) from e

        ds_name = self.settings.hf_dataset
        revision = self.settings.hf_dataset_revision
        # Streaming filter to avoid full download when possible. Cap the scan so
        # a rare package cannot walk the entire corpus unbounded.
        target = max(max_reviews * 4, max_reviews)
        max_scan = 400_000
        try:
            ds = load_dataset(
                ds_name, split="train", streaming=True, revision=revision
            )
            rows: list[dict[str, Any]] = []
            scanned = 0
            for row in ds:
                scanned += 1
                pkg = (
                    row.get("package_name")
                    or row.get("appId")
                    or row.get("app_id")
                    or ""
                )
                if pkg == package_name:
                    rows.append(dict(row))
                    if len(rows) >= target:
                        break
                if scanned >= max_scan and not rows:
                    break
            raw = pd.DataFrame(rows)
        except Exception:
            ds = load_dataset(ds_name, split="train", revision=revision)
            raw = ds.to_pandas()
            pkg_col = None
            for c in ("package_name", "appId", "app_id"):
                if c in raw.columns:
                    pkg_col = c
                    break
            if pkg_col is None:
                return self._empty()
            raw = raw[raw[pkg_col] == package_name]

        return self._filter_dedupe(raw, package_name, max_reviews)

    def _filter_dedupe(
        self, raw: pd.DataFrame, package_name: str, max_reviews: int
    ) -> pd.DataFrame:
        if raw.empty:
            return self._empty()

        text_col = next(
            (
                c
                for c in ("review_text", "text", "review", "content", "body")
                if c in raw.columns
            ),
            None,
        )
        rating_col = next(
            (c for c in ("rating", "score", "star", "stars") if c in raw.columns),
            None,
        )
        date_col = next(
            (
                c
                for c in ("created_at", "date", "review_date", "at")
                if c in raw.columns
            ),
            None,
        )
        if text_col is None:
            return self._empty()

        records = []
        seen_norm: set[str] = set()
        for _, row in raw.iterrows():
            text = str(row.get(text_col) or "").strip()
            if _word_count(text) < 10:
                continue
            rating = row.get(rating_col) if rating_col else None
            try:
                rating_f = float(rating) if rating is not None else None
            except (TypeError, ValueError):
                rating_f = None
            if rating_f is not None and rating_f >= 5.0:
                continue
            norm = _norm_text(text)
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            created = row.get(date_col) if date_col else None
            if created is not None and not isinstance(created, str):
                created = str(created)
            rid = _stable_review_id(package_name, text, rating_f, created)
            records.append(
                {
                    "review_id": rid,
                    "review_text": text,
                    "rating": rating_f if rating_f is not None else 3.0,
                    "created_at": created,
                }
            )
            if len(records) >= max_reviews:
                break
        return pd.DataFrame(records)


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _pick_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    norm_map = {_normalize_header(c): c for c in columns}
    for alias in aliases:
        if alias in norm_map:
            return norm_map[alias]
    for alias in aliases:
        for norm, original in norm_map.items():
            if alias in norm:
                return original
    return None


def parse_reviews_csv(
    file_bytes: bytes, *, filename: str | None = None
) -> CsvIngestResult:
    """
    Flexible CSV → raw frame with review_text / rating / created_at columns
    (pre-filter). Raises ValueError when no text column can be detected.
    """
    if not file_bytes or not file_bytes.strip():
        raise ValueError("CSV file is empty")

    warnings: list[str] = []
    last_err: Exception | None = None
    raw: pd.DataFrame | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            from io import BytesIO

            raw = pd.read_csv(BytesIO(file_bytes), dtype=str, encoding=enc)
            break
        except Exception as e:
            last_err = e
            continue
    if raw is None:
        raise ValueError(f"Could not parse CSV: {last_err}")

    if raw.empty:
        raise ValueError("CSV has no data rows")

    text_col = _pick_column(list(raw.columns), TEXT_COLUMN_ALIASES)
    if text_col is None:
        raise ValueError(
            "Could not detect a review text column. Expected one of: "
            + ", ".join(TEXT_COLUMN_ALIASES)
        )
    rating_col = _pick_column(list(raw.columns), RATING_COLUMN_ALIASES)
    date_col = _pick_column(list(raw.columns), DATE_COLUMN_ALIASES)

    mapping: dict[str, str | None] = {
        "review_text": text_col,
        "rating": rating_col,
        "created_at": date_col,
    }
    if rating_col is None:
        warnings.append("no rating column detected; defaulting ratings to 3.0")
    if date_col is None:
        warnings.append("no date column detected; created_at left empty")
    if filename:
        warnings.append(f"parsed {filename}")

    out = pd.DataFrame(
        {
            "review_text": raw[text_col].fillna("").astype(str),
            "rating": raw[rating_col] if rating_col else None,
            "created_at": raw[date_col] if date_col else None,
            "review": raw[text_col].fillna("").astype(str),
            "star": raw[rating_col] if rating_col else None,
            "date": raw[date_col] if date_col else None,
        }
    )
    return CsvIngestResult(
        df=out,
        column_mapping=mapping,
        warnings=warnings,
        rows_raw=len(out),
        rows_kept=len(out),
    )


class GitHubScraper:
    """Fetch open+closed issues and milestones with token auth."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _session(self) -> requests.Session:
        s = requests.Session()
        s.headers["User-Agent"] = UA
        s.headers["Accept"] = "application/vnd.github+json"
        token = self.settings.effective_github_token
        if token:
            s.headers["Authorization"] = f"Bearer {token}"
        return s

    def fetch_issues_and_milestones(
        self, repo: str, *, max_issues: int = 200
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Return DataFrame columns:
        [issue_id, text, labels, milestone_title, state, age_days, closed_at, url,
         updated_at, created_at, kind]
        """
        degraded: list[str] = []
        if not self.settings.github_token_present:
            degraded.append("GitHub unauthenticated")
            return self._empty(), degraded

        session = self._session()
        rows: list[dict[str, Any]] = []

        # Repo probe
        probe = session.get(f"https://api.github.com/repos/{repo}", timeout=25)
        if probe.status_code == 404:
            degraded.append(f"GitHub repo not found: {repo}")
            return self._empty(), degraded
        if probe.status_code == 403:
            remaining = probe.headers.get("X-RateLimit-Remaining", "?")
            degraded.append(f"GitHub 403/rate-limit (remaining={remaining})")
            return self._empty(), degraded
        if probe.status_code != 200:
            degraded.append(f"GitHub repo fetch HTTP {probe.status_code}")
            return self._empty(), degraded

        # Milestones
        ms = session.get(
            f"https://api.github.com/repos/{repo}/milestones",
            params={"state": "all", "per_page": 100},
            timeout=25,
        )
        if ms.status_code == 200 and isinstance(ms.json(), list):
            now = datetime.now(timezone.utc)
            for m in ms.json():
                title = (m.get("title") or "").strip()
                if not title or is_version_milestone_title(title):
                    continue
                created = m.get("created_at")
                updated = m.get("updated_at") or created
                closed = m.get("closed_at")
                age = _age_days(updated, now)
                desc = strip_github_boilerplate(m.get("description") or "")
                rows.append(
                    {
                        "issue_id": f"milestone-{m.get('number')}",
                        "text": compose_issue_text(title, desc),
                        "title": title,
                        "labels": "",
                        "milestone_title": title,
                        "state": m.get("state") or "open",
                        "age_days": age,
                        "closed_at": closed,
                        "url": m.get("html_url") or "",
                        "updated_at": updated,
                        "created_at": created,
                        "kind": "milestone",
                        "due_on": m.get("due_on"),
                    }
                )
        elif ms.status_code in (403, 429):
            degraded.append(f"GitHub milestones HTTP {ms.status_code}")

        # Issues (open + closed), paginated, exclude PRs
        per_page = 100
        pages = max(1, (max_issues + per_page - 1) // per_page)
        now = datetime.now(timezone.utc)
        fetched = 0
        for page in range(1, pages + 1):
            r = session.get(
                f"https://api.github.com/repos/{repo}/issues",
                params={
                    "state": "all",
                    "per_page": per_page,
                    "page": page,
                    "sort": "updated",
                    "direction": "desc",
                },
                timeout=30,
            )
            if r.status_code in (403, 429):
                degraded.append(f"GitHub issues HTTP {r.status_code}")
                break
            if r.status_code != 200:
                degraded.append(f"GitHub issues HTTP {r.status_code}")
                break
            batch = r.json()
            if not isinstance(batch, list) or not batch:
                break
            for issue in batch:
                if "pull_request" in issue:
                    continue
                title = (issue.get("title") or "").strip()
                body = (issue.get("body") or "")[:4000]
                labels = [
                    (lb.get("name") or "") for lb in (issue.get("labels") or [])
                ]
                ms_obj = issue.get("milestone") or {}
                ms_title = ms_obj.get("title") if isinstance(ms_obj, dict) else None
                created = issue.get("created_at")
                updated = issue.get("updated_at") or created
                closed = issue.get("closed_at")
                state = issue.get("state") or "open"
                rows.append(
                    {
                        "issue_id": f"issue-{issue.get('number')}",
                        "text": compose_issue_text(title, body),
                        "title": title,
                        "labels": ", ".join(labels),
                        "milestone_title": ms_title,
                        "state": state,
                        "age_days": _age_days(updated, now),
                        "closed_at": closed,
                        "url": issue.get("html_url") or "",
                        "updated_at": updated,
                        "created_at": created,
                        "kind": "issue",
                        "due_on": None,
                    }
                )
                fetched += 1
                if fetched >= max_issues:
                    break
            if fetched >= max_issues:
                break

        if not rows:
            return self._empty(), degraded
        return pd.DataFrame(rows), degraded

    def search_repo(self, query: str) -> tuple[str | None, list[str]]:
        """GitHub search API fallback for guess_repo."""
        degraded: list[str] = []
        if not self.settings.github_token_present:
            degraded.append("GitHub unauthenticated; repo search skipped")
            return None, degraded
        session = self._session()
        r = session.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": 5, "sort": "stars"},
            timeout=25,
        )
        if r.status_code in (403, 429):
            degraded.append(f"GitHub search HTTP {r.status_code}")
            return None, degraded
        if r.status_code != 200:
            degraded.append(f"GitHub search HTTP {r.status_code}")
            return None, degraded
        items = (r.json() or {}).get("items") or []
        if not items:
            return None, degraded
        return items[0].get("full_name"), degraded

    def _empty(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "issue_id",
                "text",
                "title",
                "labels",
                "milestone_title",
                "state",
                "age_days",
                "closed_at",
                "url",
                "updated_at",
                "created_at",
                "kind",
                "due_on",
            ]
        )


def _age_days(ts: str | None, now: datetime) -> float | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return max(0.0, (now - dt).total_seconds() / 86400.0)
    except Exception:
        return None


class WebRoadmapScraper:
    """
    Fetch roadmap/changelog pages with a strict relevance gate.
    Accepted hosts: product official domain, github.com/<repo>, well-known changelog hosts.
    Require product name in title or extracting section; drop unrelated sentences.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def fetch_roadmap_pages(
        self,
        urls: list[str],
        *,
        product_name: str,
        package_name: str = "",
        github_repo: str | None = None,
        official_domains: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        degraded: list[str] = []
        session = requests.Session()
        session.headers["User-Agent"] = UA
        rows: list[dict[str, Any]] = []
        allowed_hosts = self._allowed_hosts(
            product_name, package_name, github_repo, official_domains
        )
        name_tokens = _product_tokens(product_name, package_name)

        for url in urls:
            if not url or not self._url_allowed(url, allowed_hosts, github_repo):
                continue
            try:
                r = session.get(url, timeout=20)
            except Exception as e:
                degraded.append(f"web fetch failed {url}: {e}")
                continue
            if r.status_code != 200 or not r.text:
                continue
            ctype = (r.headers.get("content-type") or "").lower()
            if "html" not in ctype and "text" not in ctype:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()
            # Prefer main/article sections
            sections = soup.find_all(["article", "main", "section"]) or [soup]
            kept_chunks: list[str] = []
            for sec in sections:
                sec_text = " ".join(sec.get_text(" ").split())
                if not sec_text:
                    continue
                if not _mentions_product(title, sec_text[:2000], url, name_tokens, product_name, package_name):
                    continue
                for sent in _sentences(sec_text):
                    if _mentions_product(title, sent, url, name_tokens, product_name, package_name):
                        kept_chunks.append(sent)
            if not kept_chunks:
                # Title-level accept with short snippet only when title matches
                page_text = " ".join(soup.get_text(" ").split())[:4000]
                if _mentions_product(title, page_text[:500], url, name_tokens, product_name, package_name):
                    if product_name.lower() in title.lower() or (
                        package_name and package_name.lower() in (title + url).lower()
                    ):
                        for sent in _sentences(page_text)[:20]:
                            if any(t in sent.lower() for t in name_tokens) or PLAN_HINTS.search(
                                sent
                            ):
                                # Still require product mention in surrounding window
                                if _mentions_product(
                                    title, sent, url, name_tokens, product_name, package_name
                                ):
                                    kept_chunks.append(sent)
            if not kept_chunks:
                continue
            text = " ".join(kept_chunks)[:6000]
            rows.append({"source_url": url, "text": text, "title": title})

        if not rows:
            return pd.DataFrame(columns=["source_url", "text", "title"]), degraded
        return pd.DataFrame(rows), degraded

    def _allowed_hosts(
        self,
        product_name: str,
        package_name: str,
        github_repo: str | None,
        official_domains: list[str] | None,
    ) -> set[str]:
        hosts = set(WELL_KNOWN_CHANGELOG_HOSTS)
        if official_domains:
            for d in official_domains:
                hosts.add(d.lower().lstrip("."))
        # Heuristic official domains from product name
        slug = re.sub(r"[^a-z0-9]", "", product_name.lower())
        if slug:
            hosts.add(f"{slug}.org")
            hosts.add(f"{slug}.com")
            hosts.add(f"www.{slug}.org")
            hosts.add(f"www.{slug}.com")
            hosts.add(f"forum.{slug}.org")
        if package_name:
            # e.g. de.danoeh.antennapod → antennapod.org already covered
            pass
        if github_repo:
            hosts.add("github.com")
        return hosts

    def _url_allowed(
        self, url: str, allowed_hosts: set[str], github_repo: str | None
    ) -> bool:
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
        except Exception:
            return False
        if not host:
            return False
        if host.startswith("www."):
            host_bare = host[4:]
        else:
            host_bare = host
        if host in allowed_hosts or host_bare in allowed_hosts:
            if host.endswith("github.com") or host == "github.com":
                if github_repo:
                    return github_repo.lower() in url.lower()
                return True
            return True
        # Allow subdomain of official *.org/*.com when product slug matches
        for h in allowed_hosts:
            if host.endswith("." + h) or host_bare.endswith("." + h):
                return True
        return False


def _product_tokens(product_name: str, package_name: str) -> list[str]:
    tokens = [t for t in re.split(r"\W+", product_name.lower()) if len(t) >= 3]
    if package_name:
        tail = package_name.rsplit(".", 1)[-1].lower()
        if len(tail) >= 3 and tail not in tokens:
            tokens.append(tail)
    return tokens


def _mentions_product(
    title: str,
    text: str,
    url: str,
    tokens: list[str],
    product_name: str,
    package_name: str,
) -> bool:
    blob = f"{title} {text} {url}".lower()
    if product_name and product_name.lower() in blob:
        return True
    if package_name and package_name.lower() in blob:
        return True
    if len(tokens) >= 2 and all(t in blob for t in tokens[:3]):
        return True
    if len(tokens) == 1 and tokens[0] in blob:
        return True
    return False


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if 40 <= len(p.strip()) <= 320]


def write_synthetic_reviews(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    settings = get_settings()
    scraper = ReviewScraper(settings)
    pkg = "de.danoeh.antennapod"
    path = scraper.cache_path(pkg)
    if not path.exists():
        print(
            {
                "cache_missing": str(path),
                "hint": "fetch via ReviewScraper.fetch_reviews(..., force_refresh=True)",
            }
        )
    else:
        result = scraper.fetch_reviews(pkg, max_reviews=20)
        print(
            {
                "rows": len(result.df),
                "provenance": result.provenance,
                "degraded": result.degraded,
                "sample": result.df.head(2).to_dict("records"),
            }
        )
    gh = GitHubScraper(settings)
    print({"github_token": settings.github_token_present})
    if settings.github_token_present:
        issues, deg = gh.fetch_issues_and_milestones(
            "AntennaPod/AntennaPod", max_issues=5
        )
        print({"issues": len(issues), "degraded": deg})
    else:
        print({"issues": 0, "degraded": ["no GITHUB_TOKEN"]})


if __name__ == "__main__":
    main()
