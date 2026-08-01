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


def test_ingest_drops_short_but_keeps_five_star(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Ingest filters by length and duplication only — never by rating.

    Rating is a statistic; `need_filter.is_need_bearing` is the analysis filter,
    and it is strictly better at this job: it keeps a 5-star review that voices a
    want and rejects pure praise even at 4 stars. Dropping by rating first threw
    away the polite unmet-want signal ("love it, but I wish...") that this
    product exists to surface, and left the rating histogram with a permanently
    empty 5-star bar.
    """
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

    # Only "short" is dropped — 3 of 4 rows survive, including the 5-star one.
    assert result.rows_kept == 3
    assert float(result.df["rating"].max()) == 5.0
    assert (tmp_path / "custom.testapp.parquet").exists()
    meta = (tmp_path / "custom.testapp.meta.json").read_text(encoding="utf-8")
    assert "csv_upload" in meta


def test_five_star_feature_request_reaches_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The regression this guards: a 5-star review asking for a feature.

    Ingest used to delete it before `need_filter` ever saw it, discarding the
    exact signal the need filter is built to catch.
    """
    from src.need_filter import select_need_bearing

    scraper = ReviewScraper()
    monkeypatch.setattr(scraper, "cache_path", lambda pkg: tmp_path / f"{pkg}.parquet")
    monkeypatch.setattr(scraper, "meta_path", lambda pkg: tmp_path / f"{pkg}.meta.json")

    raw = _csv(
        "text,rating\n"
        '"Love this app, but I really wish it could export my data to CSV.",5\n'
        '"Perfect application, exactly what I needed, no complaints at all here.",5\n'
    )
    result = scraper.ingest_csv("custom.testapp", raw, max_reviews=100)
    assert result.rows_kept == 2

    need_df, stats = select_need_bearing(result.df)
    kept = need_df["review_text"].tolist()

    # The feature request survives; the pure praise does not.
    assert any("wish it could export" in t for t in kept)
    assert not any("no complaints at all" in t for t in kept)
    assert stats["reviews_need_bearing"] == 1


def test_body_column_alias():
    raw = _csv(
        "body,score\n"
        '"Cannot find the search bar anymore after the latest update broke navigation.",1\n'
    )
    parsed = parse_reviews_csv(raw)
    assert parsed.column_mapping["review_text"] == "body"
    assert parsed.column_mapping["rating"] == "score"
