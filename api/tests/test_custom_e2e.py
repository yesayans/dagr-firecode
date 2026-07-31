"""CSV + external roadmap through the analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings, get_settings
from src.data_ingestion import ReviewScraper
from src.pipeline import AnalysisPipeline
from src.store import LocalJsonStore, reset_store_singleton


@pytest.fixture()
def local_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    reset_store_singleton()
    data = tmp_path / "data"
    (data / "reviews").mkdir(parents=True)
    (data / "cache").mkdir(parents=True)
    (data / "roadmaps").mkdir(parents=True)
    monkeypatch.setenv("ROADMAP_MATCHING_ENABLED", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        ROADMAP_MATCHING_ENABLED=False,
        OPENROUTER_API_KEY=None,
    )
    # Point repo data_dir
    monkeypatch.setattr(type(settings), "data_dir", property(lambda self: data))
    store = LocalJsonStore(data / "cache" / "store.json")
    yield settings, store, data
    get_settings.cache_clear()
    reset_store_singleton()


def test_csv_and_paste_roadmap_pipeline(local_env, monkeypatch: pytest.MonkeyPatch):
    settings, store, data = local_env
    scraper = ReviewScraper(settings)
    monkeypatch.setattr(
        scraper, "cache_path", lambda pkg: data / "reviews" / f"{pkg}.parquet"
    )
    monkeypatch.setattr(
        scraper,
        "meta_path",
        lambda pkg: data / "reviews" / f"{pkg}.meta.json",
    )

    csv = (
        "review,stars,date\n"
        '"The offline sync never finishes and I lose notes every commute home.",1,2024-01-01\n'
        '"Please add export to CSV so I can leave this app when needed.",2,2024-02-01\n'
        '"Dark mode is missing and night reading hurts my eyes badly now.",2,2024-03-01\n'
        '"Search cannot find old notes even when I type the exact title words.",1,2024-04-01\n'
        '"Notifications for shared folders never arrive on my second device.",2,2024-05-01\n'
        '"Backup restore wiped half my notebooks after the last update shipped.",1,2024-06-01\n'
        '"Wish collaboration comments worked without forcing a cloud account.",3,2024-07-01\n'
        '"Tags are broken after import and I cannot organize research folders.",2,2024-08-01\n'
        '"The editor crashes when pasting long markdown from my research notes.",1,2024-09-01\n'
        '"Would love widgets for pinned notes on the home screen for quick access.",3,2024-10-01\n'
        '"Attachment uploads fail silently on mobile data and waste my time.",2,2024-11-01\n'
        '"Version history disappeared so I cannot undo destructive edits anymore.",1,2024-12-01\n'
    ).encode("utf-8")

    pkg = "custom.acme.notes"
    ingested = scraper.ingest_csv(pkg, csv, max_reviews=500)
    assert ingested.rows_kept >= 8

    app = store.upsert_app(
        {
            "package_name": pkg,
            "display_name": "Acme Notes",
            "dataset": "csv_upload",
            "review_count": ingested.rows_kept,
            "avg_stars": 2.0,
            "github_repo": None,
            "roadmap_source": "web",
            "roadmap_item_count": 3,
            "sample_review": "offline sync",
            "metadata": {
                "external_roadmap_urls": [],
                "external_roadmap_text": (
                    "Offline sync for teams\n"
                    "Export to CSV and Markdown\n"
                    "Home screen widgets\n"
                ),
                "force_roadmap_refresh": True,
                "review_provenance": "csv_upload",
            },
        }
    )
    job = store.create_job(
        {
            "app_id": app["id"],
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "config_hash": "test-csv",
        }
    )

    # Pipeline constructs its own ReviewScraper — point settings.data_dir
    pipe = AnalysisPipeline(store, settings)
    monkeypatch.setattr(pipe.reviews, "cache_dir", data / "reviews")
    monkeypatch.setattr(
        pipe.reviews,
        "cache_path",
        lambda pkg: data / "reviews" / f"{pkg}.parquet",
    )
    monkeypatch.setattr(
        pipe.reviews,
        "meta_path",
        lambda pkg: data / "reviews" / f"{pkg}.meta.json",
    )

    pipe.run(job["id"], app, max_reviews=500)
    done = store.get_job(job["id"])
    assert done["status"] == "completed", done.get("error")
    assert done["roadmap_source"] == "web"
    assert (done.get("stats") or {}).get("review_provenance") == "csv_upload"
    gaps = done.get("gaps") or []
    assert len(gaps) >= 1
    assert all(g["verdict"] == "UNVERIFIED" for g in gaps)
    assert all(g.get("evidence") for g in gaps)
