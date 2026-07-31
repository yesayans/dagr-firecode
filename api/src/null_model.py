"""
Null-model calibration for roadmap match scores.

Unrelated apps' need-bearing review clusters are scored against the target
roadmap. A live match must exceed a high percentile of that distribution —
i.e. beat matches we get from reviews that have nothing to do with the product.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import Settings, get_settings
from src.embedding_engine import adaptive_k_range, min_cluster_size
from src.need_filter import select_need_bearing
from src.review_match import aggregate_cluster_match

logger = logging.getLogger(__name__)

CONTROL_PACKAGES = (
    "com.ichi2.anki",
    "org.isoron.uhabits",
    "org.wordpress.android",
)


@dataclass
class NullCalibration:
    threshold: float
    percentile: float
    n_control_clusters: int
    n_control_reviews: int
    scores: list[float]
    roadmap_hash: str
    control_packages: list[str]
    method: str = "review_agree_mean_top1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NullCalibration":
        return cls(**data)


def roadmap_content_hash(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(str(t).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _cache_path(settings: Settings, roadmap_hash: str) -> Path:
    root = settings.data_dir / "cache" / "null_thresholds"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{roadmap_hash}.json"


def _load_control_need_reviews(
    settings: Settings, packages: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    from src.data_ingestion import ReviewScraper

    scraper = ReviewScraper(settings)
    out: dict[str, pd.DataFrame] = {}
    for pkg in packages:
        try:
            result = scraper.fetch_reviews(pkg, max_reviews=settings.max_reviews)
            df = result.df if hasattr(result, "df") else result
            need, _ = select_need_bearing(df)
            if need is not None and not need.empty:
                out[pkg] = need
        except Exception as e:
            logger.warning("null-model: skip control %s (%s)", pkg, e)
    return out


def _fit_space(texts: list[str], n_components: int = 256) -> tuple[Any, TruncatedSVD, np.ndarray]:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=40000,
        sublinear_tf=True,
        min_df=1,
    )
    X = vec.fit_transform(texts)
    n_comp = min(n_components, max(2, X.shape[1] - 1), max(2, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = normalize(svd.fit_transform(X), norm="l2")
    return vec, svd, emb


def _transform(vec: Any, svd: TruncatedSVD, texts: list[str]) -> np.ndarray:
    clean = [t if t and str(t).strip() else " " for t in texts]
    return normalize(svd.transform(vec.transform(clean)), norm="l2")


def _cluster_embeddings(emb: np.ndarray) -> list[list[int]]:
    n = len(emb)
    if n == 0:
        return []
    min_size = min_cluster_size(n)
    if n < min_size:
        return [list(range(n))]
    k_min, k_max = adaptive_k_range(n)
    best_labels = None
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb)
        if len(set(labels.tolist())) < 2:
            continue
        # Prefer more clusters that still clear min_size
        sizes = [int(np.sum(labels == c)) for c in set(labels.tolist())]
        if max(sizes) < min_size:
            continue
        score = float(np.mean([s for s in sizes if s >= min_size]))
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_labels is None:
        best_labels = KMeans(n_clusters=k_min, random_state=42, n_init=10).fit_predict(emb)
    groups: list[list[int]] = []
    for cid in sorted(set(best_labels.tolist())):
        idxs = [i for i, lab in enumerate(best_labels.tolist()) if lab == cid]
        if len(idxs) >= min_size:
            groups.append(idxs)
    return groups or [list(range(n))]


def compute_null_calibration(
    roadmap_texts: list[str],
    settings: Settings | None = None,
    *,
    control_packages: tuple[str, ...] = CONTROL_PACKAGES,
    percentile: float = 95.0,
    use_cache: bool = True,
) -> NullCalibration:
    settings = settings or get_settings()
    texts = [str(t) for t in roadmap_texts if t and str(t).strip()]
    if not texts:
        return NullCalibration(
            threshold=1.0,
            percentile=percentile,
            n_control_clusters=0,
            n_control_reviews=0,
            scores=[],
            roadmap_hash="empty",
            control_packages=list(control_packages),
        )

    rhash = roadmap_content_hash(texts)
    cache = _cache_path(settings, rhash)
    if use_cache and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if (
                data.get("method") == "review_agree_mean_top1"
                and float(data.get("percentile", 0)) == float(percentile)
            ):
                return NullCalibration.from_dict(data)
        except Exception as e:
            logger.warning("null-model cache unreadable (%s); recomputing", e)

    controls = _load_control_need_reviews(settings, control_packages)
    control_texts: list[str] = []
    control_slices: dict[str, tuple[int, int]] = {}
    for pkg, df in controls.items():
        start = len(control_texts)
        control_texts.extend(str(t) for t in df["review_text"].tolist())
        control_slices[pkg] = (start, len(control_texts))

    if not control_texts:
        cal = NullCalibration(
            threshold=1.0,
            percentile=percentile,
            n_control_clusters=0,
            n_control_reviews=0,
            scores=[],
            roadmap_hash=rhash,
            control_packages=list(control_packages),
        )
        return cal

    vec, svd, _ = _fit_space(texts + control_texts)
    item_emb = _transform(vec, svd, texts)
    all_idx = list(range(len(texts)))

    scores: list[float] = []
    n_reviews = 0
    for pkg, (start, end) in control_slices.items():
        member_texts = control_texts[start:end]
        n_reviews += len(member_texts)
        emb = _transform(vec, svd, member_texts)
        for member_rows in _cluster_embeddings(emb):
            agg = aggregate_cluster_match(
                emb[member_rows],
                item_emb,
                all_idx,
                require_agreement=False,
            )
            scores.append(float(agg.score) if agg is not None else 0.0)

    if not scores:
        thr = 1.0
    else:
        thr = float(np.percentile(scores, percentile))

    cal = NullCalibration(
        threshold=thr,
        percentile=percentile,
        n_control_clusters=len(scores),
        n_control_reviews=n_reviews,
        scores=[round(s, 6) for s in scores],
        roadmap_hash=rhash,
        control_packages=list(controls.keys()),
    )
    if use_cache:
        try:
            cache.write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("null-model cache write failed: %s", e)
    return cal


def get_null_threshold(
    roadmap_texts: list[str],
    settings: Settings | None = None,
    *,
    percentile: float | None = None,
) -> NullCalibration:
    settings = settings or get_settings()
    pct = (
        float(percentile)
        if percentile is not None
        else float(getattr(settings, "null_percentile", 95.0))
    )
    return compute_null_calibration(
        roadmap_texts, settings, percentile=pct, use_cache=True
    )
