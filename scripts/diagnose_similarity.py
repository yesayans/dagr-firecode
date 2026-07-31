"""
Why does review-to-roadmap similarity never clear the threshold?

Probes real AntennaPod review phrases against the real GitHub roadmap under
several lexical representations, and reports whether the known-correct issue is
retrievable at all. Retrieval quality is the question here, not the threshold:
if the true match is not ranked first, no threshold can rescue it.
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
from src.data_ingestion import GitHubScraper  # noqa: E402

REPO = "AntennaPod/AntennaPod"

# Verbatim fragments from the real corpus, paired with the roadmap topic a human
# would expect them to match.
PROBES = [
    ("If there was Chromecast support it would be perfect.", "chromecast/cast"),
    ("only bug is the auto download doesn't seem to work", "auto download"),
    ("Lack of a persistent stream cache leads to a long pause when resuming playback on slow networks", "streaming/cache"),
    ("I would love it to go to at least 2.5 or maybe 3 speed", "playback speed"),
    ("sleep timer does not stop playback", "sleep timer"),
    ("Podcasts won't start playing", "playback"),
]


def build(items_text: list[str], probes: list[str], **kw) -> np.ndarray:
    vec = TfidfVectorizer(**kw)
    vec.fit(items_text + probes)
    return cosine_similarity(vec.transform(probes), vec.transform(items_text))


def report(name: str, sims: np.ndarray, titles: list[str]) -> None:
    print(f"\n=== {name} ===")
    for i, (probe, expected) in enumerate(PROBES):
        order = np.argsort(-sims[i])[:3]
        top = sims[i][order[0]]
        print(f"\n  probe: {probe[:70]!r}")
        print(f"  expect: {expected}   best score: {top:.3f}")
        for rank, j in enumerate(order, 1):
            print(f"    {rank}. {sims[i][j]:.3f}  {titles[j][:78]}")


def main() -> None:
    settings = get_settings()
    print(f"github token source: {getattr(settings, 'github_token_source', 'unknown')}")

    scraper = GitHubScraper(settings)
    out = scraper.fetch_issues_and_milestones(REPO)
    items = out.items if hasattr(out, "items") else (out[0] if isinstance(out, tuple) else out)
    if hasattr(items, "to_dict"):
        items = items.to_dict("records")
    print(f"roadmap items: {len(items)}")

    titles = [str(it.get("title") or it.get("text") or "")[:120] for it in items]
    texts = [str(it.get("text") or it.get("title") or "") for it in items]
    probes = [p for p, _ in PROBES]

    report("word unigram+bigram (current-style)", build(texts, probes, ngram_range=(1, 2), stop_words="english", sublinear_tf=True), titles)
    report("word, title-only corpus", build(titles, probes, ngram_range=(1, 2), stop_words="english", sublinear_tf=True), titles)
    report("char_wb 3-5 grams", build(texts, probes, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), titles)


if __name__ == "__main__":
    main()
