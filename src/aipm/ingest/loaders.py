"""Dataset loading.

`apps_reviews.csv` is ~94 MB / ~470k rows, so everything here streams in chunks
and nothing holds the full review text in memory unless it was explicitly asked
for. The per-app aggregate pass is cached to Parquet because demo selection
re-runs far more often than the raw data changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from aipm.config import Settings
from aipm.ingest import normalize as norm
from aipm.ingest.validators import (
    DatasetValidationError,
    ValidationReport,
    resolve_columns,
    validate_apps,
    validate_reviews,
)
from aipm.schemas import App, Review
from aipm.utils.hashing import stable_hash
from aipm.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 100_000

#: Bump when `_aggregate_chunk` / `_combine_partials` change shape, so a stale
#: Parquet cache is regenerated instead of silently returning the old columns.
STATS_CACHE_VERSION = 2

#: A review shorter than this is treated as low-information for the *quality*
#: signal used in demo selection. It is not dropped here.
SUBSTANTIVE_TEXT_CHARS = 80


def _normalised_entropy(counts: "np.ndarray") -> "np.ndarray":
    """Shannon entropy of each row, scaled to 0..1 against a uniform distribution.

    1.0 means every star level is equally represented, 0.0 means the app's
    reviews all carry a single rating. Used to detect degenerate scrape coverage.
    """
    totals = counts.sum(axis=1, keepdims=True)
    probs = np.where(totals > 0, counts / np.maximum(totals, 1.0), 0.0)
    # Clamp before the log so zero-probability levels contribute exactly 0
    # instead of raising a divide warning and leaking -0.0 into the output.
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    return np.clip(entropy / np.log(counts.shape[1]), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Frame -> model mapping
#
# Shared by the batch loader and the upload page. Both take a DataFrame of
# unknown provenance and need the identical alias resolution, normalisation and
# id derivation - keeping two copies is how the two paths drift apart and start
# producing different review ids for the same row.
# ---------------------------------------------------------------------------

APP_COLUMNS = (
    "app_id", "app_name", "description", "score",
    "ratings_count", "downloads", "categories", "section",
)
REVIEW_COLUMNS = ("app_id", "review_text", "review_score", "review_date", "helpful_count")


def frame_to_apps(frame: pd.DataFrame, *, source: str = "seed") -> dict[str, App]:
    """Map an apps frame onto `App` models, keyed by normalised id."""
    cols = resolve_columns(frame, APP_COLUMNS)
    apps: dict[str, App] = {}
    for row in frame.to_dict("records"):
        app_id = norm.normalize_app_id(row.get(cols.get("app_id", "")))
        if not app_id:
            continue
        categories = norm.normalize_categories(row.get(cols.get("categories", ""), ""))
        # `section` is the store shelf ("Popular apps"); useful as a fallback
        # label when `categories` held nothing but a chart placement.
        if not categories and "section" in cols:
            categories = norm.normalize_categories(row.get(cols["section"], ""))
        downloads_raw = row.get(cols.get("downloads", ""))
        apps[app_id] = App(
            app_id=app_id,
            name=norm.normalize_text(row.get(cols.get("app_name", ""))) or app_id,
            description=norm.normalize_text(row.get(cols.get("description", ""))),
            score=norm.parse_score(row.get(cols.get("score", ""))),
            ratings_count=norm.parse_int(row.get(cols.get("ratings_count", "")), 0) or None,
            downloads_raw=norm.normalize_text(downloads_raw) or None,
            downloads_numeric=norm.parse_downloads(downloads_raw),
            categories=categories,
            source=source,
        )
    return apps


def frame_to_reviews(
    frame: pd.DataFrame, app_id: str, *, limit: int | None = None
) -> list[Review]:
    """Map one app's rows onto `Review` models, most recent first.

    The source CSVs carry no review id, so one is derived deterministically from
    the content. `stable_hash`, never builtin `hash()`: the latter is salted per
    process and would break evidence links between runs.
    """
    cols = resolve_columns(frame, REVIEW_COLUMNS)
    subset = frame[frame[cols["app_id"]].map(norm.normalize_app_id) == app_id].copy()
    if "review_date" in cols:
        subset["_sort_date"] = pd.to_datetime(subset[cols["review_date"]], errors="coerce")
        # NaT sorts last, so undated reviews are only pulled in when there are
        # not enough dated ones.
        subset = subset.sort_values("_sort_date", ascending=False, na_position="last")
    if limit is not None:
        subset = subset.head(limit)

    reviews: list[Review] = []
    seen: set[str] = set()
    for position, row in enumerate(subset.to_dict("records")):
        text = norm.normalize_text(row.get(cols["review_text"]))
        if not text:
            continue
        review_date = norm.parse_date(row.get(cols.get("review_date", "")))
        review_id = f"r_{app_id}_{stable_hash([app_id, text, str(review_date), position])}"
        if review_id in seen:
            continue
        seen.add(review_id)
        score = norm.parse_score(row.get(cols.get("review_score", "")))
        reviews.append(
            Review(
                review_id=review_id,
                app_id=app_id,
                text=text,
                score=int(score) if score is not None else None,
                review_date=review_date,
                helpful_count=norm.parse_int(row.get(cols.get("helpful_count", "")), 0),
            )
        )
    return reviews


class ReviewDataset(Protocol):
    """The contract the rest of the pipeline depends on.

    Implemented by `CsvReviewDataset` today; a Postgres-backed version can be
    dropped in without touching selection, preprocessing or analysis.
    """

    def load_apps(self) -> dict[str, App]: ...

    def app_review_stats(self) -> pd.DataFrame: ...

    def load_reviews_for(
        self, app_ids: Iterable[str], *, limit_per_app: int | None = None
    ) -> dict[str, list[Review]]: ...


class CsvReviewDataset:
    """Streaming CSV-backed implementation of `ReviewDataset`."""

    def __init__(
        self,
        apps_csv: Path,
        reviews_csv: Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        cache_dir: Path | None = None,
    ) -> None:
        self.apps_csv = Path(apps_csv)
        self.reviews_csv = Path(reviews_csv)
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._apps: dict[str, App] | None = None
        self._stats: pd.DataFrame | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> CsvReviewDataset:
        return cls(
            apps_csv=settings.apps_csv,
            reviews_csv=settings.reviews_csv,
            cache_dir=settings.data_dir / "processed",
        )

    # -- validation --------------------------------------------------------

    def validate(self) -> tuple[ValidationReport, ValidationReport]:
        """Validate both files. Reads only a sample of the (large) review file."""
        self._require_files()
        apps_report = validate_apps(pd.read_csv(self.apps_csv))
        reviews_head = pd.read_csv(self.reviews_csv, nrows=5_000)
        reviews_report = validate_reviews(reviews_head)
        # n_rows from a 5k sample would be misleading in the report.
        reviews_report.n_rows = 0
        return apps_report, reviews_report

    def _require_files(self) -> None:
        missing = [p for p in (self.apps_csv, self.reviews_csv) if not p.exists()]
        if missing:
            raise DatasetValidationError(
                "missing input file(s): " + ", ".join(str(p) for p in missing)
            )

    # -- apps --------------------------------------------------------------

    def load_apps(self) -> dict[str, App]:
        if self._apps is not None:
            return self._apps
        self._require_files()
        frame = pd.read_csv(self.apps_csv)
        validate_apps(frame).raise_for_errors()
        apps = frame_to_apps(frame, source="seed")
        log.info("loaded %d apps from %s", len(apps), self.apps_csv.name)
        self._apps = apps
        return apps

    # -- streaming ---------------------------------------------------------

    def _iter_chunks(self, usecols: Sequence[str] | None = None) -> Iterator[pd.DataFrame]:
        reader = pd.read_csv(
            self.reviews_csv,
            chunksize=self.chunk_size,
            usecols=usecols,
            on_bad_lines="warn",
        )
        for chunk in reader:
            yield chunk

    def _review_columns(self) -> dict[str, str]:
        head = pd.read_csv(self.reviews_csv, nrows=100)
        report = validate_reviews(head)
        report.raise_for_errors()
        return resolve_columns(
            head, ("app_id", "review_text", "review_score", "review_date", "helpful_count")
        )

    # -- per-app aggregates ------------------------------------------------

    @property
    def _stats_cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"app_review_stats.v{STATS_CACHE_VERSION}.parquet"

    def app_review_stats(self, *, use_cache: bool = True) -> pd.DataFrame:
        """One streaming pass producing the signals demo selection scores on.

        Returns a frame indexed by ``app_id`` with review volume, rating shape,
        text-quality proxies, recency and time coverage. Review text is measured
        and discarded chunk by chunk, never accumulated.
        """
        if self._stats is not None:
            return self._stats

        cache = self._stats_cache_path
        if use_cache and cache and cache.exists():
            source_mtime = self.reviews_csv.stat().st_mtime
            if cache.stat().st_mtime >= source_mtime:
                log.info("app review stats: cache hit (%s)", cache.name)
                self._stats = pd.read_parquet(cache)
                return self._stats

        self._require_files()
        cols = self._review_columns()
        log.info("app review stats: streaming %s", self.reviews_csv.name)

        parts: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(usecols=list(cols.values())):
            parts.append(self._aggregate_chunk(chunk, cols))

        if not parts:
            raise DatasetValidationError(f"{self.reviews_csv.name} produced no rows")

        stats = self._combine_partials(pd.concat(parts, ignore_index=True))

        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            stats.to_parquet(cache)
            log.info("app review stats: cached to %s", cache)
        self._stats = stats
        return stats

    @staticmethod
    def _aggregate_chunk(chunk: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
        """Reduce one chunk to per-app partial sums that combine associatively."""
        frame = pd.DataFrame(
            {
                "app_id": chunk[cols["app_id"]].map(norm.normalize_app_id),
                "text_len": chunk[cols["review_text"]].astype("string").fillna("").str.len(),
            }
        )
        frame["substantive"] = frame["text_len"] >= SUBSTANTIVE_TEXT_CHARS
        frame["score"] = (
            pd.to_numeric(chunk[cols["review_score"]], errors="coerce")
            if "review_score" in cols
            else pd.NA
        )
        dates = (
            pd.to_datetime(chunk[cols["review_date"]], errors="coerce")
            if "review_date" in cols
            else pd.Series(pd.NaT, index=chunk.index)
        )
        frame["date"] = dates
        frame["month"] = dates.dt.to_period("M").astype("string")
        frame["helpful"] = (
            pd.to_numeric(chunk[cols["helpful_count"]], errors="coerce").fillna(0)
            if "helpful_count" in cols
            else 0
        )

        grouped = frame.groupby("app_id", dropna=True)
        partial = grouped.agg(
            n_reviews=("text_len", "size"),
            sum_text_len=("text_len", "sum"),
            n_substantive=("substantive", "sum"),
            sum_score=("score", "sum"),
            n_scored=("score", "count"),
            sum_helpful=("helpful", "sum"),
            n_helpful_pos=("helpful", lambda s: int((s > 0).sum())),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        # Per-star counts. This corpus is a quota-capped scrape (some apps hold
        # exactly N reviews per star, others only 1-star reviews), so the star
        # distribution is a *sampling* signal, not just a statistic - selection
        # uses it to avoid apps whose rating coverage is degenerate.
        for star in range(1, 6):
            partial[f"n_score_{star}"] = grouped["score"].agg(
                lambda s, star=star: int((s == star).sum())
            )
        # Month sets are unioned later; storing them keeps coverage exact rather
        # than approximating it from the date range.
        partial["months"] = grouped["month"].agg(lambda s: set(s.dropna().unique()))
        return partial.reset_index()

    @staticmethod
    def _combine_partials(partials: pd.DataFrame) -> pd.DataFrame:
        grouped = partials.groupby("app_id")
        aggregations = {
            "n_reviews": ("n_reviews", "sum"),
            "sum_text_len": ("sum_text_len", "sum"),
            "n_substantive": ("n_substantive", "sum"),
            "sum_score": ("sum_score", "sum"),
            "n_scored": ("n_scored", "sum"),
            "sum_helpful": ("sum_helpful", "sum"),
            "n_helpful_pos": ("n_helpful_pos", "sum"),
            "date_min": ("date_min", "min"),
            "date_max": ("date_max", "max"),
        }
        aggregations.update({f"n_score_{s}": (f"n_score_{s}", "sum") for s in range(1, 6)})
        out = grouped.agg(**aggregations)
        out["months"] = grouped["months"].agg(lambda sets: set().union(*sets))
        out["n_months"] = out["months"].map(len)
        out = out.drop(columns=["months"])

        out["avg_text_len"] = out["sum_text_len"] / out["n_reviews"].clip(lower=1)
        out["share_substantive"] = out["n_substantive"] / out["n_reviews"].clip(lower=1)
        out["avg_score"] = out["sum_score"] / out["n_scored"].clip(lower=1)
        out["avg_helpful"] = out["sum_helpful"] / out["n_reviews"].clip(lower=1)
        out["share_helpful"] = out["n_helpful_pos"] / out["n_reviews"].clip(lower=1)

        star_cols = [f"n_score_{s}" for s in range(1, 6)]
        stars = out[star_cols].to_numpy(dtype=float)
        out["n_star_levels"] = (stars > 0).sum(axis=1)
        out["rating_entropy"] = _normalised_entropy(stars)

        return out.drop(columns=["sum_text_len", "n_substantive", "sum_score", "sum_helpful"])

    def recent_review_shares(self, window_days: int = 365) -> pd.Series:
        """Share of each app's reviews inside the most recent `window_days`.

        Recency is measured against the **dataset's own maximum date**, not the
        wall clock: a corpus scraped in 2025 must not score as "stale" simply
        because it is read in 2026.
        """
        cols = self._review_columns()
        if "review_date" not in cols:
            return pd.Series(dtype=float)

        counts: dict[str, list[int]] = {}
        dataset_max = self.app_review_stats()["date_max"].max()
        if pd.isna(dataset_max):
            return pd.Series(dtype=float)
        cutoff = dataset_max - pd.Timedelta(days=window_days)

        for chunk in self._iter_chunks(usecols=[cols["app_id"], cols["review_date"]]):
            app_ids = chunk[cols["app_id"]].map(norm.normalize_app_id)
            dates = pd.to_datetime(chunk[cols["review_date"]], errors="coerce")
            recent = dates >= cutoff
            for app_id, is_recent in zip(app_ids, recent, strict=True):
                bucket = counts.setdefault(app_id, [0, 0])
                bucket[0] += int(bool(is_recent))
                bucket[1] += 1
        return pd.Series(
            {a: (r / t if t else 0.0) for a, (r, t) in counts.items()},
            name="share_recent",
            dtype=float,
        )

    # -- full review loading ----------------------------------------------

    def load_reviews_for(
        self, app_ids: Iterable[str], *, limit_per_app: int | None = None
    ) -> dict[str, list[Review]]:
        """Load reviews for the selected apps in a single streaming pass.

        When `limit_per_app` is set, the most recent reviews win: a demo should
        show what users are saying now, not what they said in 2012.
        """
        wanted = {norm.normalize_app_id(a) for a in app_ids}
        if not wanted:
            return {}
        self._require_files()
        cols = self._review_columns()

        frames: list[pd.DataFrame] = []
        for chunk in self._iter_chunks(usecols=list(cols.values())):
            chunk = chunk.copy()
            chunk["_app_id"] = chunk[cols["app_id"]].map(norm.normalize_app_id)
            hit = chunk[chunk["_app_id"].isin(wanted)]
            if not hit.empty:
                frames.append(hit)

        if not frames:
            return {app_id: [] for app_id in wanted}

        merged = pd.concat(frames, ignore_index=True)

        # Sorting and truncation both live in `frame_to_reviews`. Doing the sort
        # here only when the group exceeds the cap would mean small apps get
        # source order and large apps get date order - and since review ids are
        # derived from row position, the same review would take a different id
        # depending on how many siblings it had.
        out: dict[str, list[Review]] = {
            str(app_id): frame_to_reviews(group, str(app_id), limit=limit_per_app)
            for app_id, group in merged.groupby("_app_id")
        }

        for app_id in wanted:
            out.setdefault(app_id, [])
        total = sum(len(v) for v in out.values())
        log.info("loaded %d reviews across %d apps", total, len(out))
        return out

