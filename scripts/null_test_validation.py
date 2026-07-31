"""
Base-rate / null-model check for roadmap matching.

Scores each app's need-bearing review clusters against AntennaPod's roadmap using
the production matcher (review-level agreement + null threshold). Controls are
reviews of unrelated apps — they cannot legitimately match AntennaPod's roadmap.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import GitHubScraper, ReviewScraper  # noqa: E402
from src.embedding_engine import EmbeddingEngine  # noqa: E402
from src.gap_analyzer import GapMatrix, compute_review_window  # noqa: E402
from src.matching_space import build_matching_space  # noqa: E402
from src.need_filter import select_need_bearing  # noqa: E402

REPO = "AntennaPod/AntennaPod"
TARGET = "de.danoeh.antennapod"
CONTROLS = ["com.ichi2.anki", "org.isoron.uhabits", "org.wordpress.android"]


def need_reviews(pkg: str, settings):
    out = ReviewScraper(settings).fetch_reviews(pkg, 2000)
    df = out.df if hasattr(out, "df") else out
    need, stats = select_need_bearing(df)
    return need, stats


def score_group(name: str, df, space, settings, *, use_full_roadmap: bool = True):
    """
    Cluster need-bearing reviews, score each cluster against the roadmap with
    the production MatchingSpace. Report fire rate under the null threshold.
    """
    if df is None or df.empty:
        return {
            "group": name,
            "clusters": 0,
            "fired": 0,
            "rate": 0.0,
            "mean_top1": 0.0,
            "verdicts": {},
            "validated_rate": 0.0,
        }

    engine = EmbeddingEngine(settings)
    clustered = engine.embed_and_cluster(
        df, roadmap_texts=space.roadmap_texts
    )
    clusters = clustered["clusters"]
    reviews_df = clustered["reviews_df"]
    review_emb = clustered["embeddings"]

    # Build a synthetic items frame from roadmap texts for GapMatrix temporal rules.
    # For the null test we treat every item as open+contemporaneous so the fire
    # rate reflects matching only (not temporal filtering).
    import pandas as pd

    items = pd.DataFrame(
        [
            {
                "item_id": f"i{i}",
                "text": t,
                "title": t.split(".")[0][:120],
                "state": "open",
                "milestone_title": None,
                "closed_at": None,
                "created_at": "2010-01-01T00:00:00Z",
                "updated_at": "2010-01-01T00:00:00Z",
                "url": "",
                "labels": "",
                "kind": "issue",
            }
            for i, t in enumerate(space.roadmap_texts)
        ]
    )

    window = compute_review_window(reviews_df)
    analyzer = GapMatrix(
        settings,
        match_threshold=space.threshold,
        matching_space=space,
    )
    # Force all items contemporaneous by setting window after all created_at
    from src.gap_analyzer import ReviewWindow, parse_dt

    wide = ReviewWindow(
        start=parse_dt("2000-01-01"),
        end=parse_dt("2099-01-01"),
    )

    gaps = analyzer.analyze(
        clusters=clusters,
        review_embeddings=review_emb,
        reviews_df=reviews_df,
        roadmap_items=items,
        roadmap_embeddings=space.item_emb,
        roadmap_source="github",
        total_reviews=len(df),
        review_window=wide if use_full_roadmap else window,
    )

    # Also score raw aggregate matches (pre-verdict) for mean top1
    scores = []
    fired = 0
    for c in clusters:
        texts = [
            str(t)
            for t in reviews_df.loc[
                reviews_df["review_id"].isin(c["review_ids"]), "review_text"
            ].tolist()
        ]
        agg = space.match_cluster(texts)
        if agg is None:
            scores.append(0.0)
            continue
        scores.append(agg.score)
        if space.accepts(agg):
            fired += 1

    verdicts = Counter(g.verdict for g in gaps)
    validated = sum(
        1 for g in gaps if (g.metrics or {}).get("validated_by_later_roadmap")
    )
    n = max(len(clusters), 1)
    import numpy as np

    return {
        "group": name,
        "clusters": len(clusters),
        "fired": fired,
        "rate": fired / n,
        "mean_top1": float(np.mean(scores)) if scores else 0.0,
        "verdicts": dict(verdicts),
        "validated_rate": validated / max(len(gaps), 1) if gaps else 0.0,
        "n_gaps": len(gaps),
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute matching space (ignore null-threshold cache)",
    )
    args = parser.parse_args()
    settings = get_settings()
    settings.resolve_github_credentials()

    items_df, deg = GitHubScraper(settings).fetch_issues_and_milestones(REPO)
    if deg:
        print(f"degraded: {deg}")
    roadmap_texts = [str(t) for t in items_df["text"].tolist()]
    space = build_matching_space(
        roadmap_texts, settings, use_cache=not args.no_cache
    )
    print(
        f"roadmap items: {len(roadmap_texts)}  "
        f"null_threshold(p{space.null.percentile:g})={space.threshold:.4f}  "
        f"control_clusters={space.null.n_control_clusters}"
    )
    print()

    rows = []
    need, _ = need_reviews(TARGET, settings)
    rows.append(score_group("AntennaPod (real)", need, space, settings))
    for pkg in CONTROLS:
        try:
            need, _ = need_reviews(pkg, settings)
            rows.append(score_group(f"{pkg} (control)", need, space, settings))
        except Exception as e:
            print(f"  skip {pkg}: {type(e).__name__}: {e}")

    print(
        f"{'group':<34} {'clusters':>8} {'fired':>6} {'rate':>6} "
        f"{'mean top1':>10}  verdicts"
    )
    print("-" * 90)
    for r in rows:
        v = " ".join(f"{k}={n}" for k, n in sorted(r["verdicts"].items()))
        print(
            f"{r['group']:<34} {r['clusters']:>8} {r['fired']:>6} "
            f"{100 * r['rate']:>5.0f}% {r['mean_top1']:>10.3f}  {v}"
        )

    real = next(r for r in rows if r["group"].startswith("AntennaPod"))
    controls = [r for r in rows if "control" in r["group"]]
    if controls:
        ctrl_rate = sum(r["rate"] for r in controls) / len(controls)
        sep = real["rate"] - ctrl_rate
        print()
        print(
            f"separation (real_rate - mean_control_rate) = {sep:.3f} "
            f"(real={real['rate']:.2%} ctrl_mean={ctrl_rate:.2%})"
        )
        print(
            "If separation is near zero, roadmap-matching claims are not supportable."
        )


if __name__ == "__main__":
    main()
