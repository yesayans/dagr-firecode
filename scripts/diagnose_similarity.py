"""
Probe review fragments against the AntennaPod GitHub roadmap under word /
char_wb / union lexical representations (same backends as production).

Use scripts/calibrate_retrieval.py to refresh the offline calibration fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import GitHubScraper  # noqa: E402
from src.embedding_engine import TfidfSvdBackend  # noqa: E402

REPO = "AntennaPod/AntennaPod"

PROBES = [
    ("If there was Chromecast support it would be perfect.", "chromecast"),
    ("only bug is the auto download doesn't seem to work", "automatic download"),
    (
        "Lack of a persistent stream cache leads to a long pause when resuming playback on slow networks",
        "re-buffer",
    ),
    ("I would love it to go to at least 2.5 or maybe 3 speed", "playback speed"),
    ("sleep timer does not stop playback", "sleep timer"),
    ("Podcasts won't start playing", "(ambiguous)"),
]


def report(mode: str, texts: list[str], titles: list[str]) -> None:
    probes = [p for p, _ in PROBES]
    backend = TfidfSvdBackend(vectorizer_mode=mode)
    emb = backend.fit_transform(texts + probes)
    sims = emb[len(texts) :] @ emb[: len(texts)].T
    thr = get_settings().match_threshold_tfidf
    margin_req = get_settings().match_margin_tfidf
    print(f"\n=== {mode} (threshold={thr} margin={margin_req}) ===")
    for i, (probe, expected) in enumerate(PROBES):
        order = np.argsort(-sims[i])
        top = float(sims[i][order[0]])
        runner = float(sims[i][order[1]]) if len(order) > 1 else 0.0
        margin = top - runner
        accept = top >= thr and margin >= margin_req
        hit = expected != "(ambiguous)" and expected.lower() in titles[order[0]].lower()
        flag = "CORRECT" if hit else ("ACCEPT" if accept else "below")
        print(f"\n  probe: {probe[:70]!r}")
        print(
            f"  expect: {expected}   best={top:.3f} margin={margin:.3f} "
            f"→ {flag}"
        )
        for rank, j in enumerate(order[:3], 1):
            print(f"    {rank}. {sims[i][j]:.3f}  {titles[j][:78]}")


def main() -> None:
    settings = get_settings()
    print(f"github token source: {getattr(settings, 'github_token_source', 'unknown')}")
    scraper = GitHubScraper(settings)
    df, degraded = scraper.fetch_issues_and_milestones(REPO)
    if degraded:
        print(f"degraded: {degraded}")
    items = df.to_dict("records")
    print(f"roadmap items: {len(items)}")
    titles = [str(it.get("title") or it.get("text") or "")[:120] for it in items]
    texts = [str(it.get("text") or it.get("title") or "") for it in items]
    for mode in ("word", "char_wb", "union"):
        report(mode, texts, titles)


if __name__ == "__main__":
    main()
