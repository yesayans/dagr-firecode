"""Inspect AntennaPod cluster centroids vs live GitHub issues."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import GitHubScraper, ReviewScraper  # noqa: E402
from src.embedding_engine import EmbeddingEngine  # noqa: E402
from src.need_filter import annotate_need_bearing  # noqa: E402


def main() -> None:
    settings = get_settings()
    settings.resolve_github_credentials()
    reviews = ReviewScraper(settings).fetch_reviews("de.danoeh.antennapod", max_reviews=500)
    df = annotate_need_bearing(reviews.df)
    gh, deg = GitHubScraper(settings).fetch_issues_and_milestones("AntennaPod/AntennaPod")
    if deg:
        print("degraded", deg)
    items = gh.rename(columns={"issue_id": "item_id"})
    engine = EmbeddingEngine(settings)
    clustered = engine.embed_and_cluster(df, roadmap_texts=items["text"].tolist())
    road_emb = engine.embed_roadmap_items(items)
    titles = items["title"].tolist()
    thr = settings.match_threshold_tfidf
    margin = settings.match_margin_tfidf
    print(
        f"clusters={len(clustered['clusters'])} need_bearing={int(df['need_bearing'].sum())} "
        f"thr={thr} margin={margin}"
    )
    for c in clustered["clusters"]:
        cent = c["centroid"]
        sims = road_emb @ cent
        order = np.argsort(-sims)
        top = float(sims[order[0]])
        runner = float(sims[order[1]])
        m = top - runner
        accept = top >= thr and m >= margin
        print(
            f"\ncluster {c['cluster_id']} size={c['size']} "
            f"mean_r={c['mean_rating']:.2f} keywords={c['keywords'][:5]}"
        )
        print(f"  repr: {c['representative_text'][:140]!r}")
        print(f"  top1={top:.3f} margin={m:.3f} accept={accept}")
        for rank, j in enumerate(order[:4], 1):
            st = items.iloc[j]["state"]
            print(f"    {rank}. {sims[j]:.3f} [{st}] {titles[j][:75]}")


if __name__ == "__main__":
    main()
