"""Dimensionality reduction before clustering.

HDBSCAN degrades badly in high dimensions - distances concentrate and density
stops meaning anything - so this step is not optional. UMAP is preferred;
TruncatedSVD is the dependency-free fallback and is used automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aipm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ReductionConfig:
    n_components: int = 8
    n_neighbors: int = 15
    min_dist: float = 0.0  # 0.0 packs points tightly, which is what HDBSCAN wants
    metric: str = "cosine"
    random_state: int = 42


@dataclass
class ReductionResult:
    embedding: np.ndarray  # (n, n_components) for clustering
    projection_2d: np.ndarray  # (n, 2) for the UI scatter
    method: str
    fallback: bool


def umap_available() -> bool:
    try:
        import umap  # noqa: F401

        return True
    except Exception:
        return False


def reduce_embeddings(vectors: np.ndarray, config: ReductionConfig | None = None) -> ReductionResult:
    """Reduce for clustering and, separately, to 2D for display.

    The 2D projection is computed independently rather than by slicing the first
    two clustering dimensions: the first two components of an 8D embedding make a
    misleading scatter plot.
    """
    cfg = config or ReductionConfig()
    n_samples = len(vectors)
    if n_samples == 0:
        return ReductionResult(
            np.zeros((0, cfg.n_components), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            "none",
            False,
        )

    # Both reducers need n_components < n_samples.
    n_components = max(2, min(cfg.n_components, n_samples - 1, vectors.shape[1]))

    if umap_available() and n_samples > cfg.n_neighbors + 1:
        try:
            return _reduce_umap(vectors, cfg, n_components)
        except Exception as exc:
            log.warning("UMAP failed (%s: %s); falling back to SVD", type(exc).__name__, exc)

    return _reduce_svd(vectors, cfg, n_components)


def _reduce_umap(vectors: np.ndarray, cfg: ReductionConfig, n_components: int) -> ReductionResult:
    import umap

    n_neighbors = min(cfg.n_neighbors, len(vectors) - 1)
    common = {
        "n_neighbors": n_neighbors,
        "min_dist": cfg.min_dist,
        "metric": cfg.metric,
        "random_state": cfg.random_state,
        "verbose": False,
    }
    log.info("UMAP: %d vectors -> %dD (n_neighbors=%d)", len(vectors), n_components, n_neighbors)
    embedding = umap.UMAP(n_components=n_components, **common).fit_transform(vectors)
    projection = (
        embedding[:, :2]
        if n_components == 2
        else umap.UMAP(n_components=2, **common).fit_transform(vectors)
    )
    return ReductionResult(
        np.asarray(embedding, dtype=np.float32),
        np.asarray(projection, dtype=np.float32),
        "umap",
        False,
    )


def _reduce_svd(vectors: np.ndarray, cfg: ReductionConfig, n_components: int) -> ReductionResult:
    from sklearn.decomposition import TruncatedSVD

    log.info("TruncatedSVD: %d vectors -> %dD", len(vectors), n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=cfg.random_state)
    embedding = svd.fit_transform(vectors)
    return ReductionResult(
        np.asarray(embedding, dtype=np.float32),
        np.asarray(embedding[:, :2], dtype=np.float32),
        "svd",
        True,
    )
