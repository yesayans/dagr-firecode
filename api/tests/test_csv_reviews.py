"""Flexible CSV review ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data_ingestion import ReviewScraper, parse_reviews_csv


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


def test_parse_aliases_review_stars_date():
    raw = _csv(
        "Review,Stars,Date\n"
        '"The sync never finishes when I am offline and I lose work.",2,2024-01-01\n'
        '"I wish there was a dark mode for night reading sessions.",4,2024-02-01\n'
    )
    parsed = parse_reviews_csv(raw, filename="t.csv")
    assert parsed.column_mapping["review_text"] == "Review"
    assert parsed.column_mapping["rating"] == "Stars"
    assert parsed.column_mapping["created_at"] == "Date"
    assert parsed.rows_raw == 2


def test_parse_requires_text_column():
    with pytest.raises(ValueError, match="review text column"):
        parse_reviews_csv(_csv("foo,bar\n1,2\n"))


def test_ingest_drops_five_star_and_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scraper = ReviewScraper()
    monkeypatch.setattr(
        scraper, "cache_path", lambda pkg: tmp_path / f"{pkg}.parquet"
    )
    monkeypatch.setattr(
        scraper, "meta_path", lambda pkg: tmp_path / f"{pkg}.meta.json"
    )

    raw = _csv(
        "text,rating\n"
        '"short",1\n'
        '"This is a long enough complaint about missing offline sync forever.",5\n'
        '"This is a long enough complaint about missing offline sync forever and it hurts.",2\n'
        '"Please add export to CSV so I can move my notes elsewhere easily.",3\n'
    )
    result = scraper.ingest_csv("custom.testapp", raw, max_reviews=100)
    assert result.rows_kept == 2
    assert float(result.df["rating"].max()) < 5
    assert (tmp_path / "custom.testapp.parquet").exists()
    meta = (tmp_path / "custom.testapp.meta.json").read_text(encoding="utf-8")
    assert "csv_upload" in meta


def test_body_column_alias():
    raw = _csv(
        "body,score\n"
        '"Cannot find the search bar anymore after the latest update broke navigation.",1\n'
    )
    parsed = parse_reviews_csv(raw)
    assert parsed.column_mapping["review_text"] == "body"
    assert parsed.column_mapping["rating"] == "score"
