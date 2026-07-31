"""Inspect need-bearing filter + clustering on a real review corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.data_ingestion import ReviewScraper  # noqa: E402
from src.embedding_engine import EmbeddingEngine  # noqa: E402
from src.need_filter import WANT_RE, select_need_bearing  # noqa: E402

PKG = sys.argv[1] if len(sys.argv) > 1 else "de.danoeh.antennapod"


def main() -> None:
    out = ReviewScraper().fetch_reviews(PKG, 2000)
    df = out.df
    need_df, stats = select_need_bearing(df)
    print(f"package: {PKG}")
    print(f"reviews_total: {stats['reviews_total']}")
    print(f"reviews_need_bearing: {stats['reviews_need_bearing']}")
    print(f"provenance: {out.provenance}")

    res = EmbeddingEngine().embed_and_cluster(need_df)
    clusters = res["clusters"]
    print(f"clusters: {len(clusters)}  k={res.get('k')} range={res.get('k_range')}")
    print(f"min_cluster_size: {res.get('min_cluster_size')}\n")

    rows = []
    for c in clusters:
        ids = c.get("review_ids") or []
        sub = need_df[need_df.review_id.isin(ids)]
        texts = list(sub.review_text)
        n_want = sum(1 for t in texts if WANT_RE.search(str(t)))
        pct = round(100 * n_want / max(len(texts), 1))
        rows.append(
            {
                "mean": float(c.get("mean_rating") or 0),
                "n": int(c.get("size") or len(texts)),
                "want": n_want,
                "pct": pct,
                "nb": float(c.get("need_bearing_share") or 0),
                "kw": ", ".join((c.get("keywords") or [])[:6]),
                "sample": next((t for t in texts if WANT_RE.search(str(t))), texts[0] if texts else ""),
            }
        )

    rows.sort(key=lambda r: -r["mean"])
    header = f"{'mean*':>6} {'n':>4} {'want':>5} {'want%':>6}  keywords"
    print(header)
    print("-" * (len(header) + 20))
    for r in rows:
        print(f"{r['mean']:6.2f} {r['n']:4d} {r['want']:5d} {r['pct']:5d}%  {r['kw']}")

    print("\nSample want-language / need-bearing reviews from clusters:")
    for r in rows:
        if r["sample"]:
            print(f"  [{r['mean']:.2f}* n={r['n']}] {str(r['sample'])[:200]}")


if __name__ == "__main__":
    main()
