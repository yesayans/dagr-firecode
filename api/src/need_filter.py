"""Per-review need-bearing classification (Silent Stakeholder signal)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Unmet want / request language — the polite 4★ "wish" signal
WANT_RE = re.compile(
    r"\b("
    r"wish|would love|would like|would be (nice|great|good|better|awesome)|"
    r"hope|please add|please (fix|make|include)|"
    r"missing|lack of|lacks?|no way to|can't|cant|cannot|doesn't let|"
    r"should (have|be able|allow|support)|if only|"
    r"need(s)? (a|an|to|the|more|better)|needs? |"
    r"feature request|suggest(ion)?|add(ed|ing)? support|"
    r"it('?d| would) (be|help)|looking for|miss(ing)? (a|an|the)|"
    r"faster .+ would|go to at least|at least \d|"
    r"leads to|would (prefer|appreciate)"
    r")\b",
    re.I,
)

# Concrete problem / defect language
PROBLEM_RE = re.compile(
    r"\b("
    r"bug|broken|crash(es|ed|ing)?|fail(s|ed|ing)?|error|freeze|frozen|"
    r"stuck|unusable|doesn'?t work|does not work|won'?t work|not work|"
    r"stopped working|keeps? (crashing|failing|stopping)|"
    r"force.?close|anr|slow network|pause when resuming|"
    r"never works|always (fails|crashes)|glitch"
    r")\b",
    re.I,
)

# Pure praise with no residual ask
PRAISE_ONLY_RE = re.compile(
    r"\b(great|love|awesome|amazing|excellent|perfect|best|thanks|thank you|"
    r"wonderful|fantastic|recommend|5 stars?|five stars?)\b",
    re.I,
)


def is_need_bearing(text: str, rating: float | None = None) -> bool:
    """
    True if the review expresses an unmet want, a problem, or a low rating.
    Pure praise with no want/problem language is False — even at 4★.
    """
    t = (text or "").strip()
    if not t:
        return False
    try:
        r = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        r = None

    if r is not None and r <= 3.0:
        return True
    if WANT_RE.search(t):
        return True
    if PROBLEM_RE.search(t):
        return True
    return False


def annotate_need_bearing(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean column need_bearing."""
    if reviews_df is None or reviews_df.empty:
        out = reviews_df.copy() if reviews_df is not None else pd.DataFrame()
        if not out.empty:
            out["need_bearing"] = False
        return out
    out = reviews_df.copy()
    flags = [
        is_need_bearing(str(row.get("review_text") or ""), row.get("rating"))
        for _, row in out.iterrows()
    ]
    out["need_bearing"] = flags
    return out


def select_need_bearing(reviews_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Return (need_bearing_df, stats) where stats has reviews_total / reviews_need_bearing.
    """
    annotated = annotate_need_bearing(reviews_df)
    total = int(len(annotated)) if annotated is not None else 0
    if annotated is None or annotated.empty:
        empty = pd.DataFrame(
            columns=["review_id", "review_text", "rating", "created_at", "need_bearing"]
        )
        return empty, {"reviews_total": 0, "reviews_need_bearing": 0}
    need = annotated[annotated["need_bearing"]].reset_index(drop=True)
    return need, {
        "reviews_total": total,
        "reviews_need_bearing": int(len(need)),
    }


def main() -> None:
    samples = [
        (
            "Faster playback would be better. It'll playback at x2 the speed... "
            "I would love it to go to at least 2.5 or maybe 3?",
            4.0,
            True,
        ),
        (
            "Lack of a persistent stream cache leads to a potentially long pause "
            "when resuming playback on slow networks.",
            4.0,
            True,
        ),
        ("Great app, love it, thanks!", 4.0, False),
        ("App crashes every time I open it", 2.0, True),
        ("Best podcast app ever", 4.0, False),
    ]
    for text, rating, expect in samples:
        got = is_need_bearing(text, rating)
        assert got is expect, (text[:40], got, expect)
    print({"ok": True, "n": len(samples)})


if __name__ == "__main__":
    main()
