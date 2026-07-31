"""
Base-rate check for retrospective validation.

`validated_by_later_roadmap` fired on every AntennaPod gap. A signal that always
fires carries no information, so this measures how often it fires for clusters
that CANNOT legitimately validate: reviews of unrelated apps scored against
AntennaPod's roadmap. If the control rate matches the real rate, the signal is
an artifact of a permissive threshold rather than evidence of anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import GitHubScraper, ReviewScraper  # noqa: E402
from src.embedding_engine import EmbeddingEngine  # noqa: E402
from src.need_filter import select_need_bearing  # noqa: E402

REPO = "AntennaPod/AntennaPod"
TARGET = "de.danoeh.antennapod"
CONTROLS = ["com.ichi2.anki", "org.isoron.uhabits", "org.wordpress.android"]


def reviews_for(pkg: str):
    out = ReviewScraper().fetch_reviews(pkg, 2000)
    df = out.df if hasattr(out, "df") else out
    try:
        res = select_need_bearing(df)
        df = res[0] if isinstance(res, tuple) else res
    except Exception:
        pass
    return df


def cluster_texts(df) -> list[str]:
    res = EmbeddingEngine().embed_and_cluster(df)
    clusters = res["clusters"] if isinstance(res, dict) else res
    out = []
    for c in clusters:
        ids = c.get("review_ids") or []
        sub = df[df.review_id.isin(ids)]
        out.append(" ".join(sub.review_text.astype(str).tolist()))
    return out


def main() -> None:
    settings = get_settings()
    threshold = float(getattr(settings, "match_threshold_tfidf", 0.16))
    print(f"threshold: {threshold}")

    items_df, _ = GitHubScraper(settings).fetch_issues_and_milestones(REPO)
    item_texts = [str(t) for t in items_df["text"].tolist()]
    print(f"roadmap items: {len(item_texts)}\n")

    groups = {"AntennaPod (real)": cluster_texts(reviews_for(TARGET))}
    for pkg in CONTROLS:
        try:
            groups[f"{pkg} (control)"] = cluster_texts(reviews_for(pkg))
        except Exception as e:
            print(f"  skip {pkg}: {type(e).__name__}: {e}")

    all_cluster_texts = [t for g in groups.values() for t in g]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit(item_texts + all_cluster_texts)
    item_mat = vec.transform(item_texts)

    print(f"{'group':<34} {'clusters':>8} {'fired':>6} {'rate':>6} {'mean top1':>10}")
    print("-" * 70)
    for name, texts in groups.items():
        if not texts:
            continue
        sims = cosine_similarity(vec.transform(texts), item_mat)
        tops = sims.max(axis=1)
        fired = int((tops >= threshold).sum())
        print(f"{name:<34} {len(texts):>8} {fired:>6} {100*fired/len(texts):>5.0f}% {tops.mean():>10.3f}")

    print("\nIf the control rates match the real rate, the signal is not evidence.")


if __name__ == "__main__":
    main()
