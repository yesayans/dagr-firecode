"""
Null-model regression for roadmap matching.

Documents that lexical review-level matching does not separate AntennaPod from
control apps, and guards the product default: matched verdicts stay off until a
future backend clears the separation bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import GitHubScraper, ReviewScraper  # noqa: E402
from src.matching_space import build_matching_space  # noqa: E402
from src.need_filter import select_need_bearing  # noqa: E402
import null_test_validation as ntv  # noqa: E402

TARGET = "de.danoeh.antennapod"
CONTROLS = ["com.ichi2.anki", "org.isoron.uhabits", "org.wordpress.android"]
# Re-enable ROADMAP_MATCHING_ENABLED only if a backend clears this bar.
MIN_SEPARATION_TO_ENABLE = 0.35


def _have_review_cache(pkg: str) -> bool:
    settings = get_settings()
    return (settings.data_dir / "reviews" / f"{pkg}.parquet").exists()


@pytest.fixture(scope="module")
def null_rows():
    if not all(_have_review_cache(p) for p in [TARGET, *CONTROLS]):
        pytest.skip("review parquet caches missing for null-model regression")
    settings = get_settings()
    settings.resolve_github_credentials()
    if not settings.github_token_present:
        pytest.skip("GitHub token required to fetch AntennaPod roadmap")

    items_df, deg = GitHubScraper(settings).fetch_issues_and_milestones(
        "AntennaPod/AntennaPod"
    )
    if items_df is None or items_df.empty:
        pytest.skip(f"roadmap unavailable ({deg})")

    roadmap_texts = [str(t) for t in items_df["text"].tolist()]
    space = build_matching_space(roadmap_texts, settings, use_cache=True)

    rows = []
    need, _ = select_need_bearing(
        ReviewScraper(settings).fetch_reviews(TARGET, 2000).df
    )
    rows.append(ntv.score_group("AntennaPod (real)", need, space, settings))
    for pkg in CONTROLS:
        need, _ = select_need_bearing(
            ReviewScraper(settings).fetch_reviews(pkg, 2000).df
        )
        rows.append(ntv.score_group(f"{pkg} (control)", need, space, settings))
    return rows, space


def test_roadmap_matching_disabled_while_lexical_null_fails():
    """Product must not ship matched verdicts on an undiscriminating signal."""
    settings = get_settings()
    assert settings.roadmap_matching_enabled is False


def test_null_separation_documented(null_rows):
    rows, space = null_rows
    real = next(r for r in rows if r["group"].startswith("AntennaPod"))
    controls = [r for r in rows if "control" in r["group"]]
    ctrl_rate = sum(r["rate"] for r in controls) / len(controls)
    sep = real["rate"] - ctrl_rate
    # Lexical path is known-weak; if a future change clears the bar, fail here
    # so ROADMAP_MATCHING_ENABLED can be turned back on deliberately.
    if sep >= MIN_SEPARATION_TO_ENABLE:
        pytest.fail(
            f"null separation improved to {sep:.2%} (real={real['rate']:.2%} "
            f"ctrl={ctrl_rate:.2%}, thr={space.threshold:.3f}). "
            "Re-enable ROADMAP_MATCHING_ENABLED and update CONTRACT.md."
        )
    # Still assert controls are not wildly above real (would invert the story)
    assert real["rate"] + 0.15 >= ctrl_rate, (
        f"controls fire more than real+15%: real={real['rate']:.2%} "
        f"ctrl={ctrl_rate:.2%}"
    )
