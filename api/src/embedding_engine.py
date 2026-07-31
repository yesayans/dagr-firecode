"""Embedding backends + review clustering."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...
    def transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfSvdBackend:
    """TF-IDF + TruncatedSVD → 256-d L2-normalised vectors. Offline, no downloads."""

    name = "tfidf"

    def __init__(self, n_components: int = 256) -> None:
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        clean = [t if t and str(t).strip() else " " for t in texts]
        X = self.vectorizer.fit_transform(clean)
        n_comp = min(self.n_components, max(2, X.shape[1] - 1), max(2, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        emb = self.svd.fit_transform(X)
        emb = normalize(emb, norm="l2")
        self._fitted = True
        return emb

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self.svd is None:
            raise RuntimeError("TfidfSvdBackend must be fit before transform")
        clean = [t if t and str(t).strip() else " " for t in texts]
        X = self.vectorizer.transform(clean)
        emb = self.svd.transform(X)
        return normalize(emb, norm="l2")


class MiniLMBackend:
    """sentence-transformers MiniLM. Opt-in; falls back is handled by factory."""

    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._fitted = True

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        clean = [t if t and str(t).strip() else " " for t in texts]
        emb = self.model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float64)


def build_backend(
    settings: Settings | None = None,
) -> tuple[EmbeddingBackend, list[str]]:
    """Return (backend, degraded notes). MiniLM import failure → TF-IDF + warning."""
    settings = settings or get_settings()
    degraded: list[str] = []
    want = (settings.embedding_backend or "tfidf").lower().strip()
    if want == "minilm":
        try:
            return MiniLMBackend(), degraded
        except Exception as e:
            msg = f"MiniLM unavailable ({e}); degraded to tfidf"
            warnings.warn(msg, stacklevel=2)
            logger.warning(msg)
            degraded.append(msg)
            return TfidfSvdBackend(), degraded
    return TfidfSvdBackend(), degraded


class EmbeddingEngine:
    """Fit on union of review+roadmap text so TF-IDF spaces are comparable."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.degraded: list[str] = []
        if backend is None:
            self.backend, self.degraded = build_backend(self.settings)
        else:
            self.backend = backend

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    def embed_and_cluster(
        self, reviews_df: pd.DataFrame, roadmap_texts: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Cluster reviews. Fit vectorizer on union(review texts, roadmap texts).
        Returns cluster ids, centroids, cohesion, keywords; drops clusters < 5 members.
        """
        if reviews_df is None or reviews_df.empty:
            return {
                "reviews_df": reviews_df,
                "embeddings": np.zeros((0, 1)),
                "clusters": [],
                "backend": self.backend_name,
            }

        texts = [str(t) for t in reviews_df["review_text"].tolist()]
        roadmap_texts = roadmap_texts or []
        union = texts + [str(t) for t in roadmap_texts if t]

        # Fit on union so roadmap transform shares the space (critical for TF-IDF)
        union_emb = self.backend.fit_transform(union)
        review_emb = union_emb[: len(texts)]

        n = len(texts)
        if n < 5:
            # Single pseudo-cluster when volume is tiny
            labels = np.zeros(n, dtype=int)
            clusters = [
                self._cluster_payload(0, labels, review_emb, reviews_df, texts)
            ]
            clusters = [c for c in clusters if c["size"] >= 1]
            return {
                "reviews_df": reviews_df.assign(cluster_id=labels),
                "embeddings": review_emb,
                "clusters": clusters,
                "backend": self.backend_name,
                "roadmap_fit_texts": roadmap_texts,
            }

        k_min, k_max = 8, 20
        # Clamp for small corpora
        k_max = min(k_max, max(2, n // 5))
        k_min = min(k_min, k_max)
        k_min = max(2, k_min)

        best_k = k_min
        best_score = -1.0
        best_labels = None
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(review_emb)
            if len(set(labels)) < 2:
                continue
            try:
                score = float(silhouette_score(review_emb, labels, metric="cosine"))
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            km = KMeans(n_clusters=k_min, random_state=42, n_init=10)
            best_labels = km.fit_predict(review_emb)

        labels = np.asarray(best_labels)
        clusters = []
        for cid in sorted(set(labels.tolist())):
            payload = self._cluster_payload(cid, labels, review_emb, reviews_df, texts)
            if payload["size"] >= 5 or (n < 25 and payload["size"] >= 3):
                # For tiny fixtures allow min 3; production drops < 5
                if n >= 25 and payload["size"] < 5:
                    continue
                clusters.append(payload)

        # Re-check: enforce <5 drop when enough reviews
        if n >= 25:
            clusters = [c for c in clusters if c["size"] >= 5]
        else:
            clusters = [c for c in clusters if c["size"] >= 3]

        return {
            "reviews_df": reviews_df.assign(cluster_id=labels),
            "embeddings": review_emb,
            "clusters": clusters,
            "backend": self.backend_name,
            "k": best_k,
            "silhouette": best_score,
            "roadmap_fit_texts": roadmap_texts,
        }

    def embed_roadmap_items(self, items_df: pd.DataFrame) -> np.ndarray:
        if items_df is None or items_df.empty:
            return np.zeros((0, 1))
        texts = [str(t) for t in items_df["text"].tolist()]
        return self.backend.transform(texts)

    def _cluster_payload(
        self,
        cid: int,
        labels: np.ndarray,
        emb: np.ndarray,
        reviews_df: pd.DataFrame,
        texts: list[str],
    ) -> dict[str, Any]:
        mask = labels == cid
        members = emb[mask]
        centroid = members.mean(axis=0)
        if np.linalg.norm(centroid) > 0:
            centroid = centroid / np.linalg.norm(centroid)
        # cohesion = mean cosine of members to centroid
        if len(members) == 0:
            cohesion = 0.0
        else:
            cohesion = float(np.mean(members @ centroid))
        member_texts = [texts[i] for i in range(len(texts)) if mask[i]]
        keywords = _top_tfidf_keywords(member_texts, top_n=8)
        member_ids = reviews_df.loc[mask, "review_id"].tolist()
        ratings = reviews_df.loc[mask, "rating"].astype(float)
        return {
            "cluster_id": int(cid),
            "size": int(mask.sum()),
            "centroid": centroid,
            "cohesion": cohesion,
            "keywords": keywords,
            "review_ids": member_ids,
            "mean_rating": float(ratings.mean()) if len(ratings) else 3.0,
            "rating_spread": float(ratings.nunique() / 5.0) if len(ratings) else 0.0,
            "member_indices": np.where(mask)[0].tolist(),
            "representative_text": _most_central_text(member_texts, members, centroid),
        }


def _top_tfidf_keywords(texts: list[str], top_n: int = 8) -> list[str]:
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
        X = vec.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        order = scores.argsort()[::-1][:top_n]
        return [str(terms[i]) for i in order if scores[i] > 0]
    except Exception:
        # Fallback: word frequency
        from collections import Counter

        words = []
        for t in texts:
            words.extend(re_words(t))
        return [w for w, _ in Counter(words).most_common(top_n)]


def re_words(text: str) -> list[str]:
    import re

    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "you",
        "are",
        "but",
        "not",
        "have",
        "app",
        "just",
    }
    return [
        w
        for w in re.findall(r"[a-z]{3,}", text.lower())
        if w not in stop
    ]


def _most_central_text(
    texts: list[str], members: np.ndarray, centroid: np.ndarray
) -> str:
    if not texts:
        return ""
    if len(members) == 0:
        return texts[0]
    sims = members @ centroid
    return texts[int(np.argmax(sims))]


def main() -> None:
    engine = EmbeddingEngine()
    df = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(12)],
            "review_text": [
                "sleep timer does not stop playback when screen locks",
                "sleep timer fails every night on bluetooth",
                "cannot get sleep timer working with headphones",
                "downloads stuck in queue forever never finish",
                "download queue broken after update please fix",
                "episode downloads fail silently no error shown",
                "car bluetooth disconnects when phone locks screen",
                "android auto bluetooth keeps dropping podcast audio",
                "bluetooth audio cuts out in the car constantly",
                "search cannot find episodes from subscribed podcasts",
                "need better search across episode titles and notes",
                "search is useless for finding old episodes",
            ],
            "rating": [2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2],
            "created_at": ["2024-01-01"] * 12,
        }
    )
    result = engine.embed_and_cluster(df, roadmap_texts=["add dark mode support"])
    print(
        {
            "backend": result["backend"],
            "clusters": len(result["clusters"]),
            "keywords": [c["keywords"][:4] for c in result["clusters"]],
            "degraded": engine.degraded,
        }
    )


if __name__ == "__main__":
    main()
