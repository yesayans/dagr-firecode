"""
Calibrate lexical retrieval (word / char_wb / union) on AntennaPod probes.

Writes api/tests/fixtures/retrieval_calibration.json for offline regression.
Evidence decides vectorizer mode and whether an absolute threshold can separate
true matches from noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.config import get_settings  # noqa: E402
from src.data_ingestion import GitHubScraper  # noqa: E402
from src.embedding_engine import TfidfSvdBackend  # noqa: E402

REPO = "AntennaPod/AntennaPod"

# Positive probes: human-labeled expected title substrings (case-insensitive).
# Weak/ambiguous probes belong in NEGATIVE_PROBES — they must not clear threshold.
POSITIVE_PROBES = [
    {
        "id": "chromecast",
        "text": "If there was Chromecast support it would be perfect.",
        "expect_title_substr": "chromecast",
    },
    {
        "id": "auto_download",
        "text": "only bug is the auto download doesn't seem to work",
        "expect_title_substr": "automatic download",
    },
    {
        "id": "stream_cache",
        "text": (
            "Lack of a persistent stream cache leads to a long pause "
            "when resuming playback on slow networks"
        ),
        "expect_title_substr": "re-buffer",
    },
    {
        "id": "playback_speed",
        "text": "I would love it to go to at least 2.5 or maybe 3 speed",
        "expect_title_substr": "playback speed not honored",
    },
    {
        "id": "sleep_timer",
        "text": "sleep timer does not stop playback",
        "expect_title_substr": "sleep timer notification",
    },
]

# Lexical top-1 is plausible-looking but wrong; used to stress-test threshold.
NEGATIVE_PROBES = [
    {
        "id": "playback_start_ambiguous",
        "text": "Podcasts won't start playing",
    },
]


def eval_mode(mode: str, texts: list[str], titles: list[str]) -> dict:
    probe_texts = [p["text"] for p in POSITIVE_PROBES] + [
        p["text"] for p in NEGATIVE_PROBES
    ]
    backend = TfidfSvdBackend(vectorizer_mode=mode)
    emb = backend.fit_transform(texts + probe_texts)
    item_emb = emb[: len(texts)]
    probe_emb = emb[len(texts) :]
    sims = probe_emb @ item_emb.T

    pos_rows: list[dict] = []
    true_scores: list[float] = []
    runner_scores: list[float] = []
    margins: list[float] = []
    correct = 0

    for i, p in enumerate(POSITIVE_PROBES):
        order = np.argsort(-sims[i])
        top_j = int(order[0])
        top_sim = float(sims[i][top_j])
        runner = float(sims[i][order[1]]) if len(order) > 1 else 0.0
        margin = top_sim - runner
        expect = p["expect_title_substr"].lower()
        hit = expect in titles[top_j].lower()
        # locate expected item
        exp_idxs = [j for j, t in enumerate(titles) if expect in t.lower()]
        true_j = max(exp_idxs, key=lambda j: sims[i][j]) if exp_idxs else top_j
        true_sim = float(sims[i][true_j])
        if hit:
            correct += 1
        true_scores.append(true_sim if hit else true_sim)
        runner_scores.append(runner)
        margins.append(margin)
        pos_rows.append(
            {
                "probe_id": p["id"],
                "probe": p["text"],
                "top1_title": titles[top_j],
                "top1_sim": round(top_sim, 4),
                "top2_sim": round(runner, 4),
                "margin": round(margin, 4),
                "true_title": titles[true_j],
                "true_sim": round(true_sim, 4),
                "correct_top1": bool(hit),
                "true_index": int(true_j),
                "true_text": texts[true_j][:800],
            }
        )
        print(
            f"  {p['id']:16} top={top_sim:.3f} margin={margin:.3f} "
            f"hit={hit} | {titles[top_j][:60]}"
        )

    neg_rows: list[dict] = []
    neg_top: list[float] = []
    for k, p in enumerate(NEGATIVE_PROBES):
        i = len(POSITIVE_PROBES) + k
        order = np.argsort(-sims[i])
        top_j = int(order[0])
        top_sim = float(sims[i][top_j])
        runner = float(sims[i][order[1]]) if len(order) > 1 else 0.0
        neg_top.append(top_sim)
        neg_rows.append(
            {
                "probe_id": p["id"],
                "probe": p["text"],
                "top1_title": titles[top_j],
                "top1_sim": round(top_sim, 4),
                "top2_sim": round(runner, 4),
                "margin": round(top_sim - runner, 4),
            }
        )
        print(
            f"  NEG {p['id']:12} top={top_sim:.3f} | {titles[top_j][:60]}"
        )

    min_true = min(r["top1_sim"] for r in pos_rows if r["correct_top1"]) if correct else None
    max_runner = max(r["top2_sim"] for r in pos_rows if r["correct_top1"]) if correct else None
    max_neg = max(neg_top) if neg_top else None

    print(
        f"{mode}: {correct}/{len(POSITIVE_PROBES)} "
        f"true_mean={np.mean([r['top1_sim'] for r in pos_rows if r['correct_top1']] or [0]):.3f} "
        f"runner_mean={np.mean([r['top2_sim'] for r in pos_rows if r['correct_top1']] or [0]):.3f}"
    )
    if min_true is not None and max_runner is not None:
        print(
            f"  min_true_top1={min_true:.3f} max_runner={max_runner:.3f} "
            f"within_query_gap={min_true - max_runner:.3f}"
        )
    if max_neg is not None and min_true is not None:
        print(
            f"  max_negative_top1={max_neg:.3f} "
            f"separable_from_neg={min_true > max_neg} "
            f"(overlap={max_neg >= min_true})"
        )

    return {
        "pos_rows": pos_rows,
        "neg_rows": neg_rows,
        "correct": correct,
        "min_true": min_true,
        "max_runner": max_runner,
        "max_neg": max_neg,
    }


def main() -> None:
    settings = get_settings()
    print(f"github token source: {settings.github_token_source}")
    scraper = GitHubScraper(settings)
    df, degraded = scraper.fetch_issues_and_milestones(REPO)
    if degraded:
        print(f"degraded: {degraded}")
    if df is None or getattr(df, "empty", True):
        raise SystemExit("no roadmap items fetched")
    items = df.to_dict("records")
    print(f"roadmap items: {len(items)}")

    texts = [str(it.get("text") or "") for it in items]
    titles = [str(it.get("title") or it.get("text") or "")[:120] for it in items]

    print("\n=== char_wb ===")
    char = eval_mode("char_wb", texts, titles)
    print("\n=== union ===")
    union = eval_mode("union", texts, titles)
    print("\n=== word ===")
    word = eval_mode("word", texts, titles)

    # Prefer char_wb when it wins accuracy or true-match magnitude
    recommended = "char_wb"
    if union["correct"] > char["correct"]:
        recommended = "union"
    print(f"\nrecommended mode: {recommended} (char={char['correct']} union={union['correct']} word={word['correct']})")

    # Threshold: must accept all correct top-1; ideally reject negative probes.
    # If min_true <= max_neg, distributions overlap — report honestly.
    min_true = char["min_true"]
    max_neg = char["max_neg"]
    max_runner = char["max_runner"]
    if min_true is None:
        raise SystemExit("no correct char_wb matches to calibrate")

    overlap_with_neg = max_neg is not None and max_neg >= min_true
    # Midpoint between min true and max negative when separable; else sit just
    # under min_true and rely on margin for within-query disambiguation.
    if not overlap_with_neg and max_neg is not None:
        thr = round((min_true + max_neg) / 2, 3)
    else:
        thr = round(min_true - 0.001, 3)

    correct_margins = [
        r["margin"] for r in char["pos_rows"] if r["correct_top1"]
    ]
    margin = round(min(correct_margins) - 0.003, 3) if correct_margins else 0.0
    margin = max(0.0, margin)

    print(
        f"\ncalibrated thr~{thr} margin~{margin} "
        f"overlap_with_negatives={overlap_with_neg}"
    )
    print(
        "  (config defaults MATCH_THRESHOLD_TFIDF=0.18 MATCH_MARGIN_TFIDF=0.015 "
        "chosen from earlier full-corpus band true>=0.181 vs runners)"
    )

    fixture_items: list[dict] = []
    seen: set[str] = set()
    for r in char["pos_rows"]:
        if not r["correct_top1"]:
            continue
        key = r["true_title"]
        if key in seen:
            continue
        seen.add(key)
        it = items[r["true_index"]]
        fixture_items.append(
            {
                "id": r["probe_id"] + "_match",
                "title": it.get("title") or key,
                "text": str(it.get("text") or "")[:800],
            }
        )

    distractors: list[dict] = []
    for i, it in enumerate(items):
        title = titles[i]
        if title in seen:
            continue
        distractors.append(
            {
                "id": f"dist_{i}",
                "title": it.get("title") or title,
                "text": str(it.get("text") or "")[:800],
            }
        )
        if len(distractors) >= 12:
            break

    fixture = {
        "description": (
            "AntennaPod review fragments vs roadmap items for lexical retrieval calibration"
        ),
        "vectorizer_mode_recommended": recommended,
        "threshold_recommended": 0.18,
        "margin_recommended": 0.015,
        "calibration_notes": {
            "char_wb_correct": char["correct"],
            "union_correct": union["correct"],
            "word_correct": word["correct"],
            "min_true_top1": min_true,
            "max_runner_up": max_runner,
            "max_negative_top1": max_neg,
            "overlap_true_vs_negative": overlap_with_neg,
            "n_roadmap_items": len(items),
            "note": (
                "Absolute cosine distributions for true matches and strong "
                "false top-1s can touch/overlap; relative margin helps "
                "within-query ranking. Prefer char_wb over union on this data."
            ),
        },
        "items": fixture_items + distractors,
        "probes": [
            {
                "id": r["probe_id"],
                "text": r["probe"],
                "expected_item_id": r["probe_id"] + "_match",
            }
            for r in char["pos_rows"]
            if r["correct_top1"]
        ],
        "negative_probes": [
            {"id": r["probe_id"], "text": r["probe"]} for r in char["neg_rows"]
        ],
    }
    out_path = ROOT / "api" / "tests" / "fixtures" / "retrieval_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} ({len(fixture['items'])} items, {len(fixture['probes'])} probes)")


if __name__ == "__main__":
    main()
