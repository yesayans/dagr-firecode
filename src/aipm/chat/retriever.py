"""Hybrid retrieval over a completed analysis run.

The chat page must only answer from what the run actually contains, so the
retrieval corpus is exactly three things: the reviews, the cluster summaries and
the extracted needs. Nothing else is reachable, which is what makes "I don't know"
an honest answer rather than a failure.

Hybrid because the two halves fail differently: BM25 nails literal terms a PM
types ("refund", "face id") but misses paraphrase; embeddings catch paraphrase but
drift on rare product nouns. Scores are combined with Reciprocal Rank Fusion,
which needs no score normalisation between the two and is hard to get wrong.

BM25 is implemented here rather than pulled in as a dependency - it is ~40 lines
and the app already ships enough wheels.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from aipm.schemas import AnalysisResult, RetrievedChunk, Review
from aipm.utils.logging import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

#: Standard BM25 parameters. k1 controls term-frequency saturation, b the
#: length normalisation.
BM25_K1 = 1.5
BM25_B = 0.75

#: RRF damping. 60 is the value from the original paper and is not sensitive.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a static corpus."""

    def __init__(self, documents: Sequence[str]) -> None:
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_lengths = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avg_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 0.0
        self.n_docs = len(self.doc_tokens)

        self.term_freqs: list[Counter[str]] = [Counter(t) for t in self.doc_tokens]
        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))

        # Inverted index so scoring touches only documents containing the term.
        self.postings: dict[str, list[int]] = {}
        for index, tokens in enumerate(self.doc_tokens):
            for term in set(tokens):
                self.postings.setdefault(term, []).append(index)

        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avg_length <= 0:
            return out
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index in self.postings.get(term, ()):
                freq = self.term_freqs[index][term]
                norm = 1 - BM25_B + BM25_B * (self.doc_lengths[index] / self.avg_length)
                out[index] += idf * (freq * (BM25_K1 + 1)) / (freq + BM25_K1 * norm)
        return out


@dataclass
class CorpusEntry:
    """One retrievable item, plus what the UI needs to render its citation."""

    kind: str  # "review" | "cluster" | "need"
    ref_id: str
    text: str
    review_id: str | None = None
    score: int | None = None
    date: str | None = None
    helpful_count: int = 0


@dataclass
class RetrievalIndex:
    """Built once per run, cached by the UI. Immutable."""

    entries: list[CorpusEntry] = field(default_factory=list)
    bm25: BM25 | None = None
    vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.entries)


#: How many reviews get embedded for semantic retrieval.
#:
#: Deliberately well below the corpus size. The pipeline embedded review
#: *segments*; these are whole reviews, so they are different strings and miss
#: the disk cache - embedding 4,000 of them costs ~30s on the first visit to an
#: app's chat, which is exactly the wait the precompute step exists to avoid.
#: The cap is applied after sorting by helpful votes, so the reviews other users
#: endorsed are the ones that stay searchable by meaning. BM25 still indexes
#: every one of them, so literal terms retain full recall.
DEFAULT_MAX_INDEXED_REVIEWS = 1500


def build_index(
    result: AnalysisResult,
    reviews: Sequence[Review],
    *,
    max_reviews: int = DEFAULT_MAX_INDEXED_REVIEWS,
    embed_texts=None,
) -> RetrievalIndex:
    """Assemble the retrieval corpus from a persisted run.

    `embed_texts` is optional. Without it retrieval is BM25-only, which still
    works - the page degrades rather than breaking when embeddings are absent.
    """
    entries: list[CorpusEntry] = []

    for need in result.needs:
        # One entry per need carrying its whole reasoning, so a question about a
        # theme retrieves the theme rather than a scattering of its quotes.
        body = " ".join(
            [need.statement, need.underlying_goal, *need.surface_complaints, *need.workarounds]
        )
        entries.append(CorpusEntry(kind="need", ref_id=need.need_id, text=body))

    for cluster in result.clusters:
        if not cluster.label and not cluster.keywords:
            continue
        body = " ".join([cluster.label, cluster.summary, *cluster.keywords])
        entries.append(CorpusEntry(kind="cluster", ref_id=cluster.cluster_id, text=body))

    # Most-helpful reviews first: if the corpus has to be capped, keep the ones
    # other users endorsed.
    ordered = sorted(reviews, key=lambda r: (r.helpful_count, len(r.text)), reverse=True)
    for review in ordered[:max_reviews]:
        entries.append(
            CorpusEntry(
                kind="review",
                ref_id=review.review_id,
                text=review.text,
                review_id=review.review_id,
                score=review.score,
                date=review.review_date.isoformat() if review.review_date else None,
                helpful_count=review.helpful_count,
            )
        )

    if not entries:
        return RetrievalIndex()

    texts = [e.text for e in entries]
    vectors = None
    if embed_texts is not None:
        try:
            vectors = np.asarray(embed_texts(texts), dtype=np.float32)
        except Exception as exc:
            log.warning("could not embed the retrieval corpus (%s); BM25 only", exc)

    log.info("retrieval index: %d entries (vectors=%s)", len(entries), vectors is not None)
    return RetrievalIndex(entries=entries, bm25=BM25(texts), vectors=vectors)


#: Cosine below this means the passage is simply unrelated to the question.
#: Without a floor, the nearest neighbour of a nonsense query is still "nearest".
MIN_SEMANTIC_SIMILARITY = 0.20


def _ranked_above(scores: np.ndarray, floor: float, limit: int) -> list[int]:
    """Indices scoring strictly above `floor`, best first, capped at `limit`."""
    hits = np.flatnonzero(scores > floor)
    if hits.size == 0:
        return []
    return hits[np.argsort(scores[hits])[::-1]][:limit].tolist()


def _rrf(rankings: Sequence[Sequence[int]], n_items: int) -> np.ndarray:
    """Reciprocal Rank Fusion. Combines rankings without normalising scores."""
    fused = np.zeros(n_items, dtype=np.float32)
    for ranking in rankings:
        for rank, index in enumerate(ranking):
            fused[index] += 1.0 / (RRF_K + rank + 1)
    return fused


class Retriever:
    """Retrieves the chunks an answer is allowed to be built from."""

    def __init__(self, index: RetrievalIndex, *, embed_texts=None) -> None:
        self.index = index
        self.embed_texts = embed_texts

    def retrieve(
        self, query: str, *, k: int = 8, kinds: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        index = self.index
        if not index.entries or index.bm25 is None or not query.strip():
            return []

        n = len(index.entries)
        candidate_rankings: list[Sequence[int]] = []

        # Each ranking contributes only entries that actually matched. RRF scores
        # by *position*, so feeding it a ranking of all-zero BM25 scores would
        # hand back arbitrary documents for a term that appears nowhere - and the
        # agent would then answer from irrelevant context instead of refusing.
        lexical = index.bm25.scores(query)
        candidate_rankings.append(_ranked_above(lexical, 0.0, k * 5))

        if index.vectors is not None and self.embed_texts is not None:
            try:
                query_vector = np.asarray(self.embed_texts([query]), dtype=np.float32)[0]
                semantic = index.vectors @ query_vector
                candidate_rankings.append(
                    _ranked_above(semantic, MIN_SEMANTIC_SIMILARITY, k * 5)
                )
            except Exception as exc:
                log.warning("query embedding failed (%s); lexical only", exc)

        if not any(candidate_rankings):
            return []

        fused = _rrf(candidate_rankings, n)

        order = np.argsort(fused)[::-1]
        out: list[RetrievedChunk] = []
        for position in order:
            if fused[position] <= 0:
                break
            entry = index.entries[position]
            if kinds and entry.kind not in kinds:
                continue
            out.append(
                RetrievedChunk(
                    kind=entry.kind,
                    ref_id=entry.ref_id,
                    text=entry.text,
                    score=float(fused[position]),
                    review_id=entry.review_id,
                )
            )
            if len(out) >= k:
                break
        return out

    def entry_for(self, ref_id: str) -> CorpusEntry | None:
        return next((e for e in self.index.entries if e.ref_id == ref_id), None)
