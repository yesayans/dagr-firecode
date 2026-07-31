"""c-TF-IDF keyword extraction.

Class-based TF-IDF treats each cluster as one document, so terms are scored by
how much they *distinguish* a cluster from its neighbours rather than by raw
frequency. Free, deterministic, and it gives the LLM prompt a factual handle on
the cluster instead of asking it to infer the theme from samples alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from aipm.utils.logging import get_logger

log = get_logger(__name__)


def extract_cluster_keywords(
    texts: Sequence[str],
    clusters: Mapping[int, Sequence[int]],
    *,
    top_n: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> dict[int, list[str]]:
    """Return the most distinguishing terms per cluster.

    `clusters` maps a cluster label to the indices of its members in `texts`.
    """
    if not texts or not clusters:
        return {label: [] for label in clusters}

    from sklearn.feature_extraction.text import CountVectorizer

    labels = list(clusters.keys())
    joined = [" ".join(texts[i] for i in clusters[label]) for label in labels]

    try:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
            # min_df counts *clusters* here, and a term appearing in one cluster
            # is exactly what we are looking for, so keep it at 1 when there are
            # few clusters.
            min_df=1 if len(labels) < min_df * 2 else min_df,
            max_features=40_000,
        )
        counts = vectorizer.fit_transform(joined).toarray().astype(np.float64)
    except ValueError as exc:  # empty vocabulary after stop-word removal
        log.warning("c-TF-IDF vocabulary empty (%s); returning no keywords", exc)
        return {label: [] for label in labels}

    vocabulary = np.array(vectorizer.get_feature_names_out())

    # c-TF-IDF: tf within the class, scaled by log(average class length / term total).
    class_totals = counts.sum(axis=1, keepdims=True)
    tf = counts / np.maximum(class_totals, 1.0)
    term_totals = counts.sum(axis=0)
    average_length = float(class_totals.mean()) if len(class_totals) else 1.0
    idf = np.log(1.0 + average_length / np.maximum(term_totals, 1.0))
    scores = tf * idf

    out: dict[int, list[str]] = {}
    for row, label in enumerate(labels):
        n_terms = min(top_n, scores.shape[1])
        if n_terms == 0:
            out[label] = []
            continue
        top = np.argpartition(scores[row], -n_terms)[-n_terms:]
        top = top[np.argsort(scores[row][top])[::-1]]
        out[label] = [str(vocabulary[i]) for i in top if scores[row][i] > 0]
    return out
