"""The embedding service: provider + cache + an addressable matrix.

Callers hand in `ReviewUnit`s and get back an `EmbeddingMatrix` whose row order
matches the units they passed. Cache lookup, deduplication of repeated text and
fitting of corpus-fitted providers all happen here so no caller repeats them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aipm.embeddings.cache import EmbeddingCache, NullEmbeddingCache
from aipm.embeddings.provider import EmbeddingProvider
from aipm.schemas import ReviewUnit
from aipm.utils.hashing import text_hash
from aipm.utils.logging import get_logger

log = get_logger(__name__)

CacheLike = EmbeddingCache | NullEmbeddingCache


@dataclass
class EmbeddingMatrix:
    """Vectors plus the ids that address them. Rows are L2-normalised."""

    unit_ids: list[str]
    vectors: np.ndarray  # (n_units, dim) float32
    model: str
    dim: int

    def __post_init__(self) -> None:
        if len(self.unit_ids) != len(self.vectors):
            raise ValueError(
                f"unit_ids/vectors mismatch: {len(self.unit_ids)} vs {len(self.vectors)}"
            )
        self._index = {uid: i for i, uid in enumerate(self.unit_ids)}

    def __len__(self) -> int:
        return len(self.unit_ids)

    def row(self, unit_id: str) -> np.ndarray:
        return self.vectors[self._index[unit_id]]

    def rows(self, unit_ids: Sequence[str]) -> np.ndarray:
        if not unit_ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.vectors[[self._index[u] for u in unit_ids]]

    def subset(self, unit_ids: Sequence[str]) -> EmbeddingMatrix:
        return EmbeddingMatrix(
            unit_ids=list(unit_ids),
            vectors=self.rows(unit_ids),
            model=self.model,
            dim=self.dim,
        )

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity. Rows are already unit length, so this is a dot."""
        return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T


class EmbeddingService:
    """Embeds text through a cache. Injected with any `EmbeddingProvider`."""

    def __init__(self, provider: EmbeddingProvider, cache: CacheLike | None = None) -> None:
        self.provider = provider
        self.cache = cache if cache is not None else NullEmbeddingCache()

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def supports_semantic_similarity(self) -> bool:
        return self.provider.supports_semantic_similarity

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed arbitrary text, hitting the cache where possible."""
        if not texts:
            return np.zeros((0, self.provider.dim), dtype=np.float32)

        # Identical text embeds once no matter how often it appears.
        unique: dict[str, str] = {}  # text_hash -> text
        for text in texts:
            unique.setdefault(text_hash(text), text)

        model = self.provider.model
        cached = self.cache.get_many(model, list(unique.values()))
        missing = [t for h, t in unique.items() if h not in cached]

        if missing:
            log.info(
                "embedding %d/%d unique texts (%d cache hits) via %s",
                len(missing), len(unique), len(cached), model,
            )
            fresh = self.provider.embed(missing)
            self.cache.put_many(model, missing, fresh)
            for text, vector in zip(missing, fresh, strict=True):
                cached[text_hash(text)] = vector
        else:
            log.info("embedding: all %d unique texts served from cache", len(unique))

        dim = len(next(iter(cached.values()))) if cached else self.provider.dim
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = cached[text_hash(text)]
        return out

    def embed_units(self, units: Sequence[ReviewUnit]) -> EmbeddingMatrix:
        """Embed units, fitting the provider first when it needs a corpus."""
        if not units:
            return EmbeddingMatrix([], np.zeros((0, self.provider.dim), dtype=np.float32),
                                   self.provider.model, self.provider.dim)

        texts = [u.text for u in units]
        # Corpus-fitted providers (TF-IDF/SVD) must see the corpus before the
        # cache is consulted: fitting is what determines the cache namespace.
        self.provider.fit(texts)

        vectors = self.embed_texts(texts)
        return EmbeddingMatrix(
            unit_ids=[u.unit_id for u in units],
            vectors=vectors,
            model=self.provider.model,
            dim=int(vectors.shape[1]),
        )

    def annotate_units(self, units: Sequence[ReviewUnit]) -> list[ReviewUnit]:
        """Stamp each unit with its embedding hash for persistence."""
        return [u.model_copy(update={"embedding_hash": text_hash(u.text)}) for u in units]
