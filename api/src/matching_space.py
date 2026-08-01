"""
Shared TF-IDF(char_wb)+SVD space for roadmap matching and null calibration.

Fitted on roadmap texts ∪ control-app need-bearing reviews so scores are
comparable across the target app and the null model, and cacheable per roadmap.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Settings, get_settings
from src.null_model import (
    CONTROL_PACKAGES,
    NullCalibration,
    _cluster_embeddings,
    _fit_space,
    _load_control_need_reviews,
    _transform,
    roadmap_content_hash,
)
from src.review_match import AggregatedMatch, aggregate_cluster_match

logger = logging.getLogger(__name__)


@dataclass
class MatchingSpace:
    vec: Any
    svd: Any
    item_emb: np.ndarray
    roadmap_texts: list[str]
    null: NullCalibration

    @property
    def threshold(self) -> float:
        return float(self.null.threshold)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return _transform(self.vec, self.svd, texts)

    def match_cluster(
        self,
        member_texts: list[str],
        indices: list[int] | None = None,
    ) -> AggregatedMatch | None:
        if not member_texts:
            return None
        idxs = indices if indices is not None else list(range(len(self.item_emb)))
        emb = self.embed_texts(member_texts)
        return aggregate_cluster_match(emb, self.item_emb, idxs)

    def accepts(self, match: AggregatedMatch | None) -> bool:
        if match is None:
            return False
        return match.score >= self.threshold


def _artifact_paths(settings: Settings, rhash: str) -> tuple[Path, Path]:
    root = (settings.data_dir / "cache" / "null_thresholds").resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Filename is our own hex hash — reject path separators just in case.
    safe = "".join(c for c in rhash if c.isalnum())[:32] or "empty"
    return root / f"{safe}.json", root / f"{safe}.pkl"


def _assert_under_cache(path: Path, settings: Settings) -> Path:
    root = (settings.data_dir / "cache" / "null_thresholds").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"cache path escapes null_thresholds dir: {resolved}")
    return resolved


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_cached_space(
    meta_path: Path,
    pkl_path: Path,
    texts: list[str],
    pct: float,
    settings: Settings,
) -> MatchingSpace | None:
    meta_path = _assert_under_cache(meta_path, settings)
    pkl_path = _assert_under_cache(pkl_path, settings)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("method") != "review_agree_mean_top1":
        return None
    if float(meta.get("percentile", -1)) != pct:
        return None
    expected = meta.get("artifact_sha256")
    if not expected or not isinstance(expected, str):
        # Legacy caches without integrity hash — refuse and rebuild.
        logger.info("matching-space cache missing artifact_sha256; rebuilding")
        return None
    raw = pkl_path.read_bytes()
    if _sha256_bytes(raw) != expected:
        logger.warning("matching-space cache integrity mismatch; rebuilding")
        return None
    # Local cache only; path confined + SHA-256 integrity checked above.
    blob = pickle.loads(raw)  # nosec B301
    if not isinstance(blob, dict) or "vec" not in blob or "svd" not in blob:
        return None
    cal_keys = {
        "threshold",
        "percentile",
        "n_control_clusters",
        "n_control_reviews",
        "scores",
        "roadmap_hash",
        "control_packages",
        "method",
    }
    cal = {k: meta[k] for k in cal_keys if k in meta}
    return MatchingSpace(
        vec=blob["vec"],
        svd=blob["svd"],
        item_emb=np.asarray(blob["item_emb"]),
        roadmap_texts=texts,
        null=NullCalibration.from_dict(cal),
    )


def build_matching_space(
    roadmap_texts: list[str],
    settings: Settings | None = None,
    *,
    percentile: float | None = None,
    use_cache: bool = True,
    control_packages: tuple[str, ...] = CONTROL_PACKAGES,
) -> MatchingSpace:
    settings = settings or get_settings()
    pct = float(
        percentile
        if percentile is not None
        else getattr(settings, "null_percentile", 95.0)
    )
    texts = [str(t) for t in roadmap_texts if t and str(t).strip()]
    rhash = roadmap_content_hash(texts) if texts else "empty"
    meta_path, pkl_path = _artifact_paths(settings, rhash)

    if use_cache and meta_path.exists() and pkl_path.exists():
        try:
            cached = _load_cached_space(meta_path, pkl_path, texts, pct, settings)
            if cached is not None:
                return cached
        except Exception as e:
            logger.warning("matching-space cache miss/unreadable (%s)", e)

    controls = _load_control_need_reviews(settings, control_packages)
    control_texts: list[str] = []
    control_slices: dict[str, tuple[int, int]] = {}
    for pkg, df in controls.items():
        start = len(control_texts)
        control_texts.extend(str(t) for t in df["review_text"].tolist())
        control_slices[pkg] = (start, len(control_texts))

    fit_corpus = texts + control_texts if control_texts else (texts or [" "])
    vec, svd, _ = _fit_space(fit_corpus)
    item_emb = (
        _transform(vec, svd, texts)
        if texts
        else np.zeros((0, getattr(svd, "n_components", 2)))
    )
    all_idx = list(range(len(texts)))

    scores: list[float] = []
    n_reviews = 0
    for pkg, (start, end) in control_slices.items():
        member_texts = control_texts[start:end]
        n_reviews += len(member_texts)
        emb = _transform(vec, svd, member_texts)
        for rows in _cluster_embeddings(emb):
            # Plurality score even without agreement — keeps the null distribution
            # informative (otherwise most mass sits at 0 and p95 collapses).
            agg = aggregate_cluster_match(
                emb[rows], item_emb, all_idx, require_agreement=False
            )
            scores.append(float(agg.score) if agg is not None else 0.0)

    if scores:
        thr = float(np.percentile(scores, pct))
    else:
        # No controls available — fall back to configured absolute floor
        thr = float(settings.match_threshold_tfidf)

    null = NullCalibration(
        threshold=thr,
        percentile=pct,
        n_control_clusters=len(scores),
        n_control_reviews=n_reviews,
        scores=[round(s, 6) for s in scores],
        roadmap_hash=rhash,
        control_packages=list(controls.keys()),
    )

    if use_cache and texts:
        try:
            meta_path = _assert_under_cache(meta_path, settings)
            pkl_path = _assert_under_cache(pkl_path, settings)
            payload = pickle.dumps(
                {"vec": vec, "svd": svd, "item_emb": item_emb},
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            meta = null.to_dict()
            meta["method"] = "review_agree_mean_top1"
            meta["artifact_sha256"] = _sha256_bytes(payload)
            pkl_path.write_bytes(payload)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("matching-space cache write failed: %s", e)

    return MatchingSpace(
        vec=vec, svd=svd, item_emb=item_emb, roadmap_texts=texts, null=null
    )
