"""The preprocessing service: raw reviews in, analysis-ready units out.

Composes `clean`, `language`, `dedupe`, `quality` and `segment` behind one
injectable object. Each underlying step stays a pure function so it can be
tested on its own; this class only sequences them and records what it dropped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from aipm.config import Settings
from aipm.preprocess import dedupe as dedupe_mod
from aipm.preprocess.clean import clean_review_text
from aipm.preprocess.language import detect_language
from aipm.preprocess.quality import is_pure_praise, quality_weight
from aipm.preprocess.segment import segment_review
from aipm.schemas import Review, ReviewUnit
from aipm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    min_segment_tokens: int = 4
    near_dup_threshold: float = 0.95
    language: str = "en"
    drop_non_language: bool = True
    #: Praise-only segments are excluded from clustering but stay in `reviews`,
    #: so ratings and trends still see them.
    drop_pure_praise: bool = True
    min_quality_weight: float = 0.0

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: object) -> PreprocessConfig:
        base = {
            "min_segment_tokens": settings.min_segment_tokens,
            "near_dup_threshold": settings.near_dup_threshold,
        }
        base.update(overrides)  # type: ignore[arg-type]
        return cls(**base)  # type: ignore[arg-type]


@dataclass
class PreprocessDiagnostics:
    """Everything the pipeline threw away, and why. Rendered in the run report."""

    n_input: int = 0
    n_blank: int = 0
    n_wrong_language: int = 0
    n_duplicates: int = 0
    n_low_quality: int = 0
    n_reviews_kept: int = 0
    n_units_total: int = 0
    n_units_praise: int = 0
    n_units_clusterable: int = 0

    def summary_line(self) -> str:
        return (
            f"{self.n_input:,} reviews -> {self.n_reviews_kept:,} kept "
            f"(blank {self.n_blank:,}, non-language {self.n_wrong_language:,}, "
            f"dupes {self.n_duplicates:,}, low-quality {self.n_low_quality:,}) "
            f"-> {self.n_units_clusterable:,}/{self.n_units_total:,} clusterable units"
        )


@dataclass
class PreprocessResult:
    reviews: list[Review] = field(default_factory=list)
    units: list[ReviewUnit] = field(default_factory=list)
    clusterable_units: list[ReviewUnit] = field(default_factory=list)
    diagnostics: PreprocessDiagnostics = field(default_factory=PreprocessDiagnostics)

    def reviews_by_id(self) -> dict[str, Review]:
        return {r.review_id: r for r in self.reviews}


class ReviewPreprocessor:
    """Clean -> language-filter -> dedupe -> weight -> segment."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def run(self, reviews: Sequence[Review]) -> PreprocessResult:
        cfg = self.config
        diagnostics = PreprocessDiagnostics(n_input=len(reviews))

        # 1. Clean and annotate language + quality.
        cleaned: list[Review] = []
        for review in reviews:
            text = clean_review_text(review.text)
            if not text:
                diagnostics.n_blank += 1
                continue
            lang = detect_language(text)
            if cfg.drop_non_language and lang != cfg.language:
                diagnostics.n_wrong_language += 1
                continue
            weight = quality_weight(text)
            if weight < cfg.min_quality_weight:
                diagnostics.n_low_quality += 1
                continue
            cleaned.append(
                review.model_copy(update={"text": text, "lang": lang, "quality_weight": weight})
            )

        # 2. Flag near-duplicates. They are kept - the statistics should still
        #    count them - but they carry the flag into the diversity score.
        marked = dedupe_mod.mark_duplicates(cleaned, threshold=cfg.near_dup_threshold)
        diagnostics.n_duplicates = sum(1 for r in marked if r.is_duplicate)
        diagnostics.n_reviews_kept = len(marked)

        # 3. Segment. Duplicates are excluded from clustering so a copy-paste
        #    campaign cannot manufacture a cluster.
        units: list[ReviewUnit] = []
        clusterable: list[ReviewUnit] = []
        for review in marked:
            review_units = segment_review(review, min_tokens=cfg.min_segment_tokens)
            units.extend(review_units)
            if review.is_duplicate:
                continue
            for unit in review_units:
                if cfg.drop_pure_praise and is_pure_praise(unit.text):
                    diagnostics.n_units_praise += 1
                    continue
                clusterable.append(unit)

        diagnostics.n_units_total = len(units)
        diagnostics.n_units_clusterable = len(clusterable)
        log.info("preprocess: %s", diagnostics.summary_line())

        return PreprocessResult(
            reviews=marked,
            units=units,
            clusterable_units=clusterable,
            diagnostics=diagnostics,
        )
