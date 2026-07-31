"""Review -> review units.

One review routinely contains several unrelated complaints:

    "Love the design. But it logs me out every day and support never replied."

Clustering that whole string blurs three signals into one vector. Splitting on
sentence boundaries *and* contrastive connectives roughly doubles cluster
quality, which is why the unit of analysis in this project is the segment.
"""

from __future__ import annotations

import re

from aipm.preprocess.clean import clean_review_text, token_count
from aipm.schemas import Review, ReviewUnit
from aipm.utils.hashing import stable_hash

#: Sentence terminators, plus newlines which reviewers use as bullet separators.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

#: Contrastive / additive connectives that almost always introduce a *new*
#: complaint. Split on these too, keeping the connective out of the segment.
_CONNECTIVE_RE = re.compile(
    r"\s+(?:but|however|although|though|except that|the only|other than that|"
    r"unfortunately|also|plus|and then|además)\s+",
    re.IGNORECASE,
)

#: A segment longer than this is almost certainly several thoughts that dodged
#: both splitters; fall back to a comma split rather than embedding a paragraph.
_MAX_SEGMENT_CHARS = 400
_COMMA_RE = re.compile(r",\s+")


def split_text(text: str) -> list[str]:
    """Split cleaned review text into candidate segments."""
    if not text:
        return []
    segments: list[str] = []
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for part in _CONNECTIVE_RE.split(sentence):
            part = part.strip(" ,;:-")
            if not part:
                continue
            if len(part) > _MAX_SEGMENT_CHARS:
                segments.extend(p.strip() for p in _COMMA_RE.split(part) if p.strip())
            else:
                segments.append(part)
    return segments


def segment_review(review: Review, *, min_tokens: int = 4) -> list[ReviewUnit]:
    """Split one review into units, dropping fragments too short to mean anything.

    If every segment is below `min_tokens` the whole cleaned review is kept as a
    single unit - losing a short review entirely would silently drop evidence.
    """
    cleaned = clean_review_text(review.text)
    if not cleaned:
        return []

    candidates = [s for s in split_text(cleaned) if token_count(s) >= min_tokens]
    if not candidates:
        if token_count(cleaned) >= max(2, min_tokens // 2):
            candidates = [cleaned]
        else:
            return []

    units: list[ReviewUnit] = []
    for position, text in enumerate(candidates):
        units.append(
            ReviewUnit(
                unit_id=f"u_{stable_hash([review.review_id, position, text])}",
                review_id=review.review_id,
                app_id=review.app_id,
                text=text,
                position=position,
            )
        )
    return units


def segment_reviews(reviews: list[Review], *, min_tokens: int = 4) -> list[ReviewUnit]:
    units: list[ReviewUnit] = []
    for review in reviews:
        units.extend(segment_review(review, min_tokens=min_tokens))
    return units
