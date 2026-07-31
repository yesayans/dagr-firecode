"""Hard rule: no gap without evidence."""

from __future__ import annotations

from src.config import get_settings
from src.llm_extractor import ExtractedGap
from src.pipeline import AnalysisPipeline
from src.store import LocalJsonStore


def test_write_gaps_drops_empty_evidence(tmp_path):
    store = LocalJsonStore(path=tmp_path / "store.json")
    app = store.upsert_app(
        {
            "package_name": "com.example.app",
            "display_name": "Example",
            "review_count": 10,
            "avg_stars": 3.0,
            "github_repo": None,
            "roadmap_source": "none",
            "roadmap_item_count": 0,
            "sample_review": "bad",
        }
    )
    job = store.create_job(
        {"app_id": app["id"], "status": "running", "stage": "persisting"}
    )
    gaps = [
        {
            "rank": 1,
            "need": "Should be dropped",
            "one_sentence_summary": "no evidence",
            "verdict": "UNVERIFIED",
            "confidence": 50,
            "confidence_rationale": "test",
            "latent_reasoning": "test",
            "metrics": {},
            "evidence": [],
        },
        {
            "rank": 2,
            "need": "Keep me",
            "one_sentence_summary": "has evidence",
            "verdict": "UNVERIFIED",
            "confidence": 70,
            "confidence_rationale": "test",
            "latent_reasoning": "test",
            "metrics": {"cluster_size": 5},
            "evidence": [
                {
                    "evidence_id": "r1",
                    "source_type": "review",
                    "title": "r1",
                    "snippet": "something bad happened with sync",
                    "url": None,
                    "payload": {"components": {}, "weights": {}},
                }
            ],
        },
    ]
    written = store.write_gaps_with_evidence(job["id"], gaps)
    assert len(written) == 1
    assert written[0]["need"] == "Keep me"
    job2 = store.get_job(job["id"])
    assert len(job2["gaps"]) == 1
    assert job2["gaps"][0]["evidence"]


def test_pipeline_hard_rule_rerank(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store = LocalJsonStore(path=tmp_path / "store.json")
    pipe = AnalysisPipeline(store)

    extracted = [
        ExtractedGap(
            latent_need="No evidence gap",
            verdict="UNVERIFIED",
            confidence=90,
            confidence_rationale="x",
            one_sentence_summary="x",
            latent_reasoning="x",
            cited_review_ids=["missing"],
            llm_used=False,
            need_source="representative_review",
            llm_confidence=None,
            metrics={"cluster_size": 5, "deterministic_confidence": 90},
            review_ids=["missing"],
            matched_item=None,
            keywords=["x"],
        ),
        ExtractedGap(
            latent_need="Real gap",
            verdict="UNVERIFIED",
            confidence=80,
            confidence_rationale="x",
            one_sentence_summary="x",
            latent_reasoning="x",
            cited_review_ids=["r1"],
            llm_used=False,
            need_source="representative_review",
            llm_confidence=None,
            metrics={"cluster_size": 6, "deterministic_confidence": 80},
            review_ids=["r1", "r2"],
            matched_item=None,
            keywords=["download"],
        ),
    ]
    reviews_by_id = {
        "r1": {
            "review_id": "r1",
            "review_text": "downloads fail often",
            "rating": 1,
        },
        "r2": {"review_id": "r2", "review_text": "queue stuck", "rating": 2},
    }
    rows = pipe._to_gap_rows(extracted, reviews_by_id, "none")
    rows = [g for g in rows if g.get("evidence")]
    for i, g in enumerate(rows, start=1):
        g["rank"] = i
    assert len(rows) == 1
    assert rows[0]["rank"] == 1
    assert rows[0]["need"] == "Real gap"
