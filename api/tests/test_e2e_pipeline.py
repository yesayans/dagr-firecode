"""End-to-end pipeline on synthetic fixture, none mode, no network/LLM."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import get_settings
from src.gap_analyzer import reconstruct_confidence
from src.pipeline import AnalysisPipeline, config_hash
from src.resolver import ResolveResult, _empty_items
from src.store import LocalJsonStore

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_reviews.parquet"


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    data = tmp_path / "data"
    reviews = data / "reviews"
    roadmaps = data / "roadmaps"
    reviews.mkdir(parents=True)
    roadmaps.mkdir(parents=True)

    assert FIXTURE.exists(), "run scripts/seed_synthetic_reviews.py first"
    df = pd.read_parquet(FIXTURE)
    pkg = "com.dagr.synthetic"
    df.to_parquet(reviews / f"{pkg}.parquet", index=False)

    store = LocalJsonStore(path=tmp_path / "store.json")
    app = store.upsert_app(
        {
            "package_name": pkg,
            "display_name": "SyntheticApp",
            "review_count": len(df),
            "avg_stars": 2.0,
            "github_repo": None,
            "roadmap_source": "none",
            "roadmap_item_count": 0,
            "sample_review": df.iloc[0]["review_text"],
        }
    )

    settings = get_settings()
    pipe = AnalysisPipeline(store, settings)
    pipe.reviews.cache_dir = reviews
    pipe.resolver.cache_dir = roadmaps

    # Force none-mode resolve — no network
    def _none_resolve(*_a, **_k):
        return ResolveResult(
            roadmap_source="none",
            github_repo=None,
            web_urls=None,
            roadmap_items=_empty_items(),
            degraded=[],
            notes="e2e forced none",
        )

    monkeypatch.setattr(pipe.resolver, "resolve", _none_resolve)

    job = store.create_job(
        {
            "app_id": app["id"],
            "config_hash": config_hash(app["id"], len(df), settings),
            "status": "queued",
            "stage": "queued",
        }
    )
    return store, pipe, app, job, df


def test_e2e_none_mode_offline(seeded):
    _store, pipe, app, job, df = seeded
    result = pipe.run(job["id"], app, max_reviews=len(df))
    assert result["status"] == "completed", result.get("error")
    assert result["stage"] == "done"
    assert result["roadmap_source"] == "none"
    assert result["stats"]["llm_used"] is False
    assert result["stats"]["embedding_backend"] == "tfidf"
    assert result["gaps"], "expected at least one gap"
    for g in result["gaps"]:
        assert g["verdict"] == "UNVERIFIED"
        assert g["evidence"], "hard rule: evidence required"
        assert g["metrics"]["components"]
        assert g["metrics"]["weights"]
        det = g["metrics"]["deterministic_confidence"]
        assert reconstruct_confidence(g["metrics"]) == det
        assert g["confidence"] == det
        assert "deterministic only" in g["confidence_rationale"]
