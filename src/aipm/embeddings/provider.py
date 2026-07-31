"""Embedding backends.

Selected by `EMBED_BACKEND`; neither the model name nor the dimension is
hardcoded in config, because each backend has its own natural identity:

* ``api``     - any OpenAI-compatible ``/embeddings`` endpoint.
* ``local``   - sentence-transformers ``all-MiniLM-L6-v2`` (384d), automatically
                degrading to TF-IDF + TruncatedSVD when torch is not installed.
* ``fixture`` - deterministic hashed vectors. Offline, instant, for tests.

Every provider returns L2-normalised ``float32`` so that a dot product is a
cosine similarity everywhere downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from aipm.config import Settings
from aipm.utils.hashing import stable_hash
from aipm.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_API_MODEL = "text-embedding-3-small"
FALLBACK_SVD_DIM = 256
DEFAULT_FIXTURE_DIM = 128


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length; zero rows are left alone rather than NaN'd."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class EmbeddingProvider(ABC):
    """The interface the pipeline is injected with."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Cache namespace. Must change whenever the vectors would change."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    def backend(self) -> str:
        return type(self).__name__

    @property
    def supports_semantic_similarity(self) -> bool:
        """Whether cosine between *paraphrases* is meaningful for this backend.

        False for bag-of-words backends: a need statement written in the model's
        own words shares little vocabulary with the reviews it summarises, so a
        similarity threshold would reject valid citations wholesale. Consumers
        must degrade the relevance check rather than trust a near-zero score.
        """
        return True

    def fit(self, corpus: Sequence[str]) -> None:
        """Hook for corpus-fitted backends. No-op for pretrained models."""
        return None

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` L2-normalised float32 matrix."""

    def describe(self) -> str:
        return f"{self.backend}(model={self.model}, dim={self.dim})"


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------


class ApiEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_API_MODEL,
        dim: int | None = None,
        batch_size: int = 256,
        timeout: float = 60.0,
        max_retries: int = 3,
        user_agent: str = "",
    ) -> None:
        if not api_key:
            raise ValueError("EMBED_BACKEND=api requires EMBED_API_KEY or LLM_API_KEY")
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            # See OpenAICompatibleClient: some routers' WAFs 403 the SDK's own
            # User-Agent.
            default_headers={"User-Agent": user_agent} if user_agent else None,
        )
        self._model = model
        # Resolved lazily from the first response rather than assumed: a router
        # may serve a different model than its name suggests.
        self._dim = dim
        self.batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start : start + self.batch_size]]
            response = self._client.embeddings.create(model=self._model, input=batch)
            ordered = sorted(response.data, key=lambda d: d.index)
            chunks.append(np.array([d.embedding for d in ordered], dtype=np.float32))
        matrix = np.vstack(chunks)
        self._dim = int(matrix.shape[1])
        return l2_normalise(matrix)


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------


class SentenceTransformerProvider(EmbeddingProvider):
    """Pretrained local model. Preferred `local` implementation."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, *, batch_size: int = 256) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._encoder = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 5.x; support both rather than pinning.
        getter = getattr(self._encoder, "get_embedding_dimension", None) or (
            self._encoder.get_sentence_embedding_dimension
        )
        self._dim = int(getter())
        self.batch_size = batch_size

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class TfidfSvdProvider(EmbeddingProvider):
    """Corpus-fitted fallback for when sentence-transformers is unavailable.

    Unlike a pretrained model this is *not* a pure function of the text: the
    vectors depend on the corpus it was fitted on. The fit signature is therefore
    folded into `model`, which is the cache namespace - otherwise a cached vector
    from a different corpus would be silently reused and be meaningless.
    """

    def __init__(self, *, dim: int = FALLBACK_SVD_DIM, random_state: int = 42) -> None:
        self._target_dim = dim
        self._random_state = random_state
        self._vectorizer = None
        self._svd = None
        self._fit_signature = "unfitted"

    @property
    def model(self) -> str:
        return f"tfidf-svd{self.dim}-{self._fit_signature}"

    @property
    def dim(self) -> int:
        return int(self._svd.n_components) if self._svd is not None else self._target_dim

    @property
    def backend(self) -> str:
        return "TfidfSvdProvider"

    @property
    def supports_semantic_similarity(self) -> bool:
        # Lexical overlap only. Paraphrases score near zero here.
        return False

    def fit(self, corpus: Sequence[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = [t for t in corpus if t and t.strip()]
        if not corpus:
            raise ValueError("cannot fit TF-IDF on an empty corpus")

        def make_vectorizer(**overrides):
            params = dict(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.6,
                max_features=60_000,
                sublinear_tf=True,
                stop_words="english",
            )
            params.update(overrides)
            return TfidfVectorizer(**params)

        self._vectorizer = make_vectorizer()
        try:
            matrix = self._vectorizer.fit_transform(corpus)
        except ValueError as exc:
            # A small or homogeneous corpus can have every term pruned by
            # min_df/max_df. Retry with the filters off rather than failing the
            # whole run over a narrow app.
            log.warning("TF-IDF pruning left no terms (%s); retrying unfiltered", exc)
            self._vectorizer = make_vectorizer(min_df=1, max_df=1.0, stop_words=None)
            matrix = self._vectorizer.fit_transform(corpus)

        # TruncatedSVD requires n_components < n_features.
        n_components = min(self._target_dim, max(2, matrix.shape[1] - 1), max(2, matrix.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self._random_state)
        self._svd.fit(matrix)

        self._fit_signature = stable_hash(
            [len(corpus), matrix.shape[1], n_components, self._random_state, corpus[0][:64]]
        )
        explained = float(self._svd.explained_variance_ratio_.sum())
        log.info(
            "TF-IDF/SVD fitted: %d docs, %d features -> %d dims (%.1f%% variance)",
            len(corpus), matrix.shape[1], n_components, explained * 100,
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TfidfSvdProvider.fit() must be called before embed()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = self._vectorizer.transform(list(texts))
        return l2_normalise(self._svd.transform(matrix))


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


class FixtureEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed vectors. No network, no model, fully reproducible.

    Similar texts do *not* land near each other, so this is for exercising the
    plumbing - never for judging cluster quality.
    """

    def __init__(self, *, dim: int = DEFAULT_FIXTURE_DIM) -> None:
        self._dim = dim

    @property
    def model(self) -> str:
        return f"fixture-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def supports_semantic_similarity(self) -> bool:
        # Hashed noise: unrelated texts are as close as related ones.
        return False

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int(stable_hash(text), 16) % (2**32)
            out[i] = np.random.default_rng(seed).standard_normal(self._dim)
        return l2_normalise(out)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the configured backend, logging what was actually resolved.

    `EMBED_MODEL` / `EMBED_DIM` override the backend default when set; leaving
    them unset is the norm.
    """
    backend = settings.embed_backend

    if backend == "api":
        provider: EmbeddingProvider = ApiEmbeddingProvider(
            base_url=settings.resolved_embed_base_url(),
            api_key=settings.resolved_embed_api_key(),
            model=settings.embed_model or DEFAULT_API_MODEL,
            dim=settings.embed_dim,
            batch_size=settings.embed_batch_size,
            timeout=float(settings.llm_timeout_s),
            max_retries=settings.llm_max_retries,
            user_agent=settings.http_user_agent,
        )
    elif backend == "fixture":
        provider = FixtureEmbeddingProvider(dim=settings.embed_dim or DEFAULT_FIXTURE_DIM)
    elif backend == "local":
        provider = _build_local_provider(settings)
    else:  # pragma: no cover - Literal type makes this unreachable
        raise ValueError(f"unknown EMBED_BACKEND: {backend!r}")

    log.info("embedding backend: %s -> %s", backend, provider.describe())
    return provider


def _build_local_provider(settings: Settings) -> EmbeddingProvider:
    model_name = settings.embed_model or DEFAULT_LOCAL_MODEL
    try:
        return SentenceTransformerProvider(model_name, batch_size=settings.embed_batch_size)
    except ImportError:
        log.warning(
            "sentence-transformers not installed; falling back to TF-IDF + TruncatedSVD. "
            "Install it with: uv pip install 'sentence-transformers>=2.7'"
        )
    except Exception as exc:  # model download failed, offline, corrupt cache...
        log.warning(
            "could not load local model %s (%s: %s); falling back to TF-IDF + TruncatedSVD",
            model_name, type(exc).__name__, exc,
        )
    return TfidfSvdProvider(dim=settings.embed_dim or FALLBACK_SVD_DIM)
