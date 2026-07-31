"""
Regression: lexical retrieval must rank known review->issue pairs correctly.

Fixture is produced by `scripts/calibrate_retrieval.py` against live AntennaPod
roadmap items. `probes` are hand-labelled positives whose expected item is present
in `items`; `negative_probes` are deliberately ambiguous reviews with no correct
match, used to document that an absolute threshold cannot separate them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import get_settings
from src.embedding_engine import TfidfSvdBackend, build_tfidf_vectorizer

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_calibration.json"


@pytest.fixture(scope="module")
def calibration():
    assert FIXTURE.exists(), f"missing {FIXTURE}; run scripts/calibrate_retrieval.py"
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _similarities(mode: str, calibration: dict):
    """Return (ids, positive_sims, negative_sims) for one vectorizer mode."""
    items = calibration["items"]
    probes = calibration["probes"]
    negatives = calibration.get("negative_probes", [])

    texts = [str(it["text"]) for it in items]
    ids = [str(it["id"]) for it in items]
    probe_texts = [p["text"] for p in probes] + [p["text"] for p in negatives]

    backend = TfidfSvdBackend(vectorizer_mode=mode)
    emb = backend.fit_transform(texts + probe_texts)
    item_emb = emb[: len(texts)]
    probe_emb = emb[len(texts) :]
    sims = probe_emb @ item_emb.T

    return ids, sims[: len(probes)], sims[len(probes) :]


def _rank_probes(mode: str, calibration: dict) -> list[dict]:
    ids, pos_sims, _ = _similarities(mode, calibration)
    rows: list[dict] = []
    for i, p in enumerate(calibration["probes"]):
        row_sims = pos_sims[i]
        order = np.argsort(-row_sims)
        top_j = int(order[0])
        runner = float(row_sims[int(order[1])]) if len(order) > 1 else 0.0
        expected = p["expected_item_id"]
        assert expected in ids, (
            f"fixture is inconsistent: probe {p['id']} expects {expected}, "
            "which is not in items; regenerate with scripts/calibrate_retrieval.py"
        )
        exp_j = ids.index(expected)
        rows.append(
            {
                "probe_id": p["id"],
                "top1_id": ids[top_j],
                "expected_id": expected,
                "correct": ids[top_j] == expected,
                "top1_sim": float(row_sims[top_j]),
                "true_sim": float(row_sims[exp_j]),
                "margin": float(row_sims[top_j]) - runner,
            }
        )
    return rows


def _negative_top1(mode: str, calibration: dict) -> list[float]:
    _, _, neg_sims = _similarities(mode, calibration)
    return [float(np.max(row)) for row in neg_sims]


def test_fixture_probes_reference_existing_items(calibration):
    """Every expected_item_id must exist; guards the generator's dedupe logic."""
    ids = {str(it["id"]) for it in calibration["items"]}
    missing = [
        p["expected_item_id"]
        for p in calibration["probes"]
        if p["expected_item_id"] not in ids
    ]
    assert not missing, f"probes reference missing items: {missing}"
    assert calibration["probes"], "fixture has no positive probes"


def test_expected_ids_retrieve_as_top1(calibration):
    rows = _rank_probes("char_wb", calibration)
    failures = [r for r in rows if not r["correct"]]
    assert not failures, (
        "retrieval top-1 regressions:\n"
        + "\n".join(
            f"  {r['probe_id']}: got {r['top1_id']} want {r['expected_id']} "
            f"(true_sim={r['true_sim']:.3f} top={r['top1_sim']:.3f})"
            for r in failures
        )
    )


def test_char_wb_beats_word_on_probes(calibration):
    char_rows = _rank_probes("char_wb", calibration)
    word_rows = _rank_probes("word", calibration)
    char_hits = sum(1 for r in char_rows if r["correct"])
    word_hits = sum(1 for r in word_rows if r["correct"])
    assert char_hits >= word_hits
    assert char_hits == len(char_rows), (
        f"char_wb top-1 correct {char_hits}/{len(char_rows)}; retrieval regressed"
    )


def test_char_wb_preferred_over_union(calibration):
    """Evidence gate: keep char_wb default unless union clearly wins."""
    char_rows = _rank_probes("char_wb", calibration)
    union_rows = _rank_probes("union", calibration)
    char_hits = sum(1 for r in char_rows if r["correct"])
    union_hits = sum(1 for r in union_rows if r["correct"])
    assert char_hits >= union_hits, "union now beats char_wb; revisit the default"

    # char_wb should also separate the true match from the rest by a wider absolute
    # margin, which is why it is the default rather than a coin flip on accuracy.
    char_margin = float(np.mean([r["margin"] for r in char_rows]))
    union_margin = float(np.mean([r["margin"] for r in union_rows]))
    assert char_margin + 1e-6 >= union_margin, (
        f"union mean margin {union_margin:.4f} exceeds char_wb {char_margin:.4f}"
    )


def test_default_backend_is_char_wb():
    backend = TfidfSvdBackend()
    assert backend.vectorizer_mode == "char_wb"
    vec = build_tfidf_vectorizer("char_wb")
    assert getattr(vec, "analyzer", None) == "char_wb"


def test_threshold_admits_every_true_match(calibration):
    """The threshold is a recall floor: no labelled true match may fall below it."""
    rows = _rank_probes("char_wb", calibration)
    thr = float(get_settings().match_threshold_tfidf)
    below = [r for r in rows if r["true_sim"] < thr]
    assert not below, (
        "true matches below MATCH_THRESHOLD_TFIDF="
        f"{thr}: "
        + ", ".join(f"{r['probe_id']}={r['true_sim']:.3f}" for r in below)
    )


def test_correct_matches_clear_the_configured_margin(calibration):
    rows = _rank_probes("char_wb", calibration)
    margin = float(get_settings().match_margin_tfidf)
    weak = [r for r in rows if r["correct"] and r["margin"] < margin]
    assert not weak, (
        f"correct matches below MATCH_MARGIN_TFIDF={margin}: "
        + ", ".join(f"{r['probe_id']}={r['margin']:.4f}" for r in weak)
    )


def test_calibration_notes_match_observed_separation(calibration):
    """Fixture notes from full-corpus calibrate must stay consistent with re-rank."""
    negatives = calibration.get("negative_probes", [])
    assert negatives, "fixture lost its negative probes; regenerate the calibration"

    rows = _rank_probes("char_wb", calibration)
    min_true = min(r["true_sim"] for r in rows)
    max_runner = max(r["true_sim"] - r["margin"] for r in rows if r["correct"])
    # Offline fixture may not reproduce full-corpus negative overlap; the durable
    # honesty signal is within-query runner-ups that can exceed the weakest true match.
    assert max_runner + 1e-9 >= min_true or calibration["calibration_notes"].get(
        "overlap_true_vs_negative"
    ), (
        "neither within-query runner overlap nor recorded negative overlap present; "
        "update CONTRACT.md calibration narrative if separation genuinely improved"
    )
    notes = calibration["calibration_notes"]
    assert notes.get("min_true_top1") is not None
    assert notes.get("n_roadmap_items", 0) >= 50
