"""Shared data contracts.

Every module in this project reads and writes these types. Define the contract
first, then let people build against stubs in parallel. If you need to change a
model here, say so out loud - it breaks other people's code.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


# --------------------------------------------------------------------------
# Raw domain objects
# --------------------------------------------------------------------------


class App(BaseModel):
    app_id: str
    name: str
    description: str = ""
    score: float | None = None
    ratings_count: int | None = None
    downloads_raw: str | None = None
    downloads_numeric: int | None = None
    categories: list[str] = Field(default_factory=list)
    source: str = "seed"  # "seed" | "upload"


class Review(BaseModel):
    review_id: str
    app_id: str
    text: str
    score: int | None = None
    review_date: date | None = None
    helpful_count: int = 0
    lang: str | None = None
    is_duplicate: bool = False
    quality_weight: float = 1.0  # 0..1, downweights spam / low-information text


class ReviewUnit(BaseModel):
    """A single clause or sentence extracted from a review.

    One review often contains several distinct complaints. Clustering whole
    reviews blurs them together, so the unit of analysis is the segment.
    """

    unit_id: str
    review_id: str
    app_id: str
    text: str
    position: int = 0
    embedding_hash: str | None = None


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


class Cluster(BaseModel):
    cluster_id: str
    run_id: str
    size: int
    keywords: list[str] = Field(default_factory=list)
    label: str = ""  # LLM-written, human readable
    summary: str = ""  # LLM-written, 1-2 sentences
    persistence: float = 0.0  # HDBSCAN cluster persistence
    cohesion: float = 0.0  # mean intra-cluster cosine similarity
    separation: float = 0.0  # normalised distance to nearest other centroid
    medoid_unit_id: str | None = None
    member_unit_ids: list[str] = Field(default_factory=list)
    representative_unit_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Needs, evidence, scoring
# --------------------------------------------------------------------------


class NeedCategory(str, Enum):
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    FEATURE_GAP = "feature_gap"
    TRUST_PRIVACY = "trust_privacy"
    PRICING = "pricing"
    SUPPORT = "support"
    CONTENT = "content"
    OTHER = "other"


class Evidence(BaseModel):
    review_id: str
    quote: str
    relevance: float = 0.0  # cosine(need_statement, review) at validation time
    review_score: int | None = None
    review_date: date | None = None
    helpful_count: int = 0
    validated: bool = False  # survived the citation guard


class ConfidenceBreakdown(BaseModel):
    """Every number here is computed in Python. The LLM never produces one.

    Each component is 0..1 and `total` is the weighted sum.

    Two explanations, deliberately separated:

    * `explanation` is assembled from the numbers above it, so the sentence under
      the meter can never contradict the bar.
    * `llm_rationale` is the model's qualitative read of *why* the evidence hangs
      together. It is reasoning, not measurement, and carries no numbers.
    """

    support: float = 0.0  # log-scaled volume of supporting units
    cohesion: float = 0.0  # how tightly the supporting cluster holds together
    separation: float = 0.0  # how distinct it is from neighbouring clusters
    temporal: float = 0.0  # sustained over time vs one-off spike
    diversity: float = 0.0  # 1 - share of near-duplicate text
    grounding: float = 0.0  # share of LLM citations that survived validation
    total: float = 0.0
    explanation: str = ""  # computed from the components above
    llm_rationale: str = ""  # qualitative, model-written, no numbers

    @computed_field  # type: ignore[misc]
    @property
    def band(self) -> str:
        if self.total >= 0.7:
            return "high"
        if self.total >= 0.45:
            return "medium"
        return "low"


class PriorityScore(BaseModel):
    reach: float = 0.0  # share of reviews touching this need
    impact: float = 0.0  # app_avg_rating - avg_rating_of_affected_reviews
    confidence: float = 0.0  # mirrors ConfidenceBreakdown.total
    value_score: float = 0.0  # reach * impact * confidence
    rank: int | None = None
    # Effort is deliberately NOT computed. The PM supplies it via a slider.
    effort: float | None = None


class Need(BaseModel):
    need_id: str
    run_id: str
    statement: str  # "Users need to trust that unsaved work survives a crash"
    underlying_goal: str  # the job-to-be-done being blocked
    category: NeedCategory = NeedCategory.OTHER
    surface_complaints: list[str] = Field(default_factory=list)
    workarounds: list[str] = Field(default_factory=list)  # strongest hidden-need signal
    cluster_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    priority: PriorityScore = Field(default_factory=PriorityScore)
    hiddenness: float = 0.0  # 1 - (explicit feature requests / total mentions)

    @computed_field  # type: ignore[misc]
    @property
    def insight_score(self) -> float:
        """Sort key for the Needs page. Hidden AND well-evidenced ranks top."""
        return self.hiddenness * self.confidence.total


# --------------------------------------------------------------------------
# Analysis runs
# --------------------------------------------------------------------------


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class AnalysisParams(BaseModel):
    """Anything that changes the output belongs here. Its hash is the cache key."""

    min_cluster_size: int | None = None  # None -> auto from corpus size
    min_samples: int | None = None
    umap_n_components: int = 8
    umap_n_neighbors: int = 15
    random_state: int = 42
    language: str = "en"
    min_segment_tokens: int = 4
    embed_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    def params_hash(self) -> str:
        from aipm.utils.hashing import stable_hash

        return stable_hash(self.model_dump())


class AnalysisRun(BaseModel):
    run_id: str
    app_id: str
    params_hash: str
    params: AnalysisParams = Field(default_factory=AnalysisParams)
    status: RunStatus = RunStatus.PENDING
    n_reviews: int = 0
    n_units: int = 0
    n_clusters: int = 0
    noise_ratio: float = 0.0
    clustering_fallback: bool = False  # True if HDBSCAN failed and KMeans was used
    citations_dropped: int = 0
    cost_usd: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class HelpfulVoteStats(BaseModel):
    """Distribution of "was this helpful" votes across an app's reviews.

    Kept separate from `OverviewStats` scalars because the distribution is
    heavily skewed - the mean alone is misleading, so the percentiles matter.
    """

    total: int = 0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    max: int = 0
    share_with_votes: float = 0.0  # fraction of reviews with at least one vote


class OverviewStats(BaseModel):
    n_reviews: int = 0
    avg_score: float = 0.0
    score_distribution: dict[int, int] = Field(default_factory=dict)
    pct_negative: float = 0.0
    trend_delta_90d: float = 0.0  # change in rolling avg score
    n_clusters: int = 0
    noise_ratio: float = 0.0
    date_range: tuple[date, date] | None = None

    helpful_votes: HelpfulVoteStats = Field(default_factory=HelpfulVoteStats)

    #: The app's rating on the store, from `apps_info.csv`. This is NOT
    #: `avg_score`: the review corpus is a quota-capped scrape, so its mean is a
    #: sampling artefact. The UI must show this as the app's real rating.
    store_score: float | None = None
    #: True when the sampled reviews look quota-balanced (near-equal counts per
    #: star) or cover only some star levels. The UI must disclose it rather than
    #: present `avg_score` as user sentiment.
    sample_is_quota_capped: bool = False
    n_star_levels: int = 0


class TrendPoint(BaseModel):
    period: date
    n_reviews: int
    avg_score: float
    rolling_avg: float | None = None


class AnalysisResult(BaseModel):
    """The single object the Streamlit layer reads. Persisted per run."""

    run: AnalysisRun
    app: App
    stats: OverviewStats
    trends: list[TrendPoint] = Field(default_factory=list)
    clusters: list[Cluster] = Field(default_factory=list)
    needs: list[Need] = Field(default_factory=list)
    projection: list[dict[str, Any]] = Field(default_factory=list)  # 2D UMAP scatter


# --------------------------------------------------------------------------
# Precomputed demo catalogue
# --------------------------------------------------------------------------


class DemoAppEntry(BaseModel):
    """One precomputed app, as advertised to the Streamlit catalogue."""

    app_id: str
    app_name: str
    category: str = "Uncategorised"
    run_id: str
    n_reviews: int = 0
    n_units: int = 0
    n_clusters: int = 0
    n_needs: int = 0
    selection_score: float = 0.0
    #: Human-readable justification for why this app made the demo set.
    selection_reasons: list[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.COMPLETE
    error: str | None = None


class DemoManifest(BaseModel):
    """Index of everything `scripts/precompute_demo.py` produced.

    The Streamlit app reads this first: it is the list of apps the demo can open
    instantly, and it records exactly how they were produced.
    """

    created_at: datetime = Field(default_factory=datetime.utcnow)
    strategy: str = "default"
    selection_config: dict[str, Any] = Field(default_factory=dict)
    embed_model: str = ""
    embed_backend: str = ""
    embed_dim: int = 0
    llm_model: str = ""
    llm_enabled: bool = False
    entries: list[DemoAppEntry] = Field(default_factory=list)
    total_reviews: int = 0
    total_needs: int = 0
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    duration_s: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def n_apps(self) -> int:
        return len([e for e in self.entries if e.status is RunStatus.COMPLETE])


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    kind: str  # "unit" | "cluster" | "need"
    ref_id: str
    text: str
    score: float
    review_id: str | None = None


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    citations: list[Evidence] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
