"""Duplicate and near-duplicate detection.

Bot reviews and copy-paste campaigns inflate cluster sizes, which would inflate
the *support* component of confidence. Flagging them here feeds the *diversity*
component, which explicitly penalises a need whose evidence is one text repeated.

Two tiers, cheapest first:

1. Exact match on aggressively normalised text - catches the bulk for free.
2. Character-shingle Jaccard within a blocking key - catches templated variants
   ("Great app!! 5 stars" / "Great app! 5 star") without an O(n^2) sweep.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from aipm.preprocess.clean import normalise_for_dedup
from aipm.schemas import Review

SHINGLE_SIZE = 5
#: Only compare texts that share this prefix; near-duplicates virtually always do.
_BLOCK_PREFIX_CHARS = 12
#: Guards the quadratic inner loop against a pathological block.
_MAX_BLOCK_SIZE = 400


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / len(a | b)


def find_duplicate_groups(
    texts: Sequence[str], *, threshold: float = 0.95
) -> dict[int, int]:
    """Map each index to the index of its group representative.

    An index mapping to itself is an original. Anything else is a near-duplicate
    of the returned representative.
    """
    normalised = [normalise_for_dedup(t) for t in texts]
    representative: dict[int, int] = {}

    # Tier 1: exact matches on normalised text.
    exact: dict[str, int] = {}
    remaining: list[int] = []
    for i, text in enumerate(normalised):
        if not text:
            representative[i] = i
            continue
        if text in exact:
            representative[i] = exact[text]
        else:
            exact[text] = i
            remaining.append(i)

    # Tier 2: shingle similarity inside prefix blocks.
    blocks: dict[str, list[int]] = defaultdict(list)
    for i in remaining:
        blocks[normalised[i][:_BLOCK_PREFIX_CHARS]].append(i)

    for members in blocks.values():
        if len(members) == 1:
            representative[members[0]] = members[0]
            continue
        if len(members) > _MAX_BLOCK_SIZE:
            # Degenerate block (e.g. thousands of "the app "). Tier 1 already
            # removed true copies; treat the rest as originals rather than
            # burning quadratic time on them.
            for i in members:
                representative[i] = i
            continue
        anchors: list[tuple[int, set[str]]] = []
        for i in members:
            sh = shingles(normalised[i])
            match = next((a for a, a_sh in anchors if jaccard(sh, a_sh) >= threshold), None)
            if match is None:
                anchors.append((i, sh))
                representative[i] = i
            else:
                representative[i] = match
    return representative


def mark_duplicates(reviews: list[Review], *, threshold: float = 0.95) -> list[Review]:
    """Return copies of `reviews` with `is_duplicate` set on non-representatives."""
    if not reviews:
        return []
    groups = find_duplicate_groups([r.text for r in reviews], threshold=threshold)
    out: list[Review] = []
    for i, review in enumerate(reviews):
        out.append(review.model_copy(update={"is_duplicate": groups.get(i, i) != i}))
    return out


def duplicate_share(reviews: Sequence[Review]) -> float:
    """Share of reviews flagged as near-duplicates. Feeds the diversity score."""
    if not reviews:
        return 0.0
    return sum(1 for r in reviews if r.is_duplicate) / len(reviews)
