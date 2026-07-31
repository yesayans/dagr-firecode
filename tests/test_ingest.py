"""Ingest: normalisation and validation."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from aipm.ingest import normalize as norm
from aipm.ingest.validators import Severity, validate_apps, validate_reviews


class TestNormalizeCategories:
    def test_strips_chart_rank_prefix(self):
        """The real dataset prefixes categories with a store chart placement."""
        assert norm.normalize_categories(
            "#9 top free news & magazines, News & Magazines"
        ) == ["News & Magazines"]

    def test_plain_category_untouched(self):
        assert norm.normalize_categories("Food & Drink") == ["Food & Drink"]

    def test_multiple_categories(self):
        assert norm.normalize_categories("Shopping, Lifestyle") == ["Shopping", "Lifestyle"]

    def test_deduplicates(self):
        assert norm.normalize_categories("Games, Games") == ["Games"]

    @pytest.mark.parametrize("value", ["", None, float("nan")])
    def test_empty_inputs(self, value):
        assert norm.normalize_categories(value) == []

    def test_primary_category_falls_back(self):
        assert norm.primary_category("#1 top grossing productivity") == "Uncategorised"


class TestNormalizeAppId:
    def test_float_ids_do_not_drift(self):
        """pandas may read the same id as int in one file and float in another."""
        assert norm.normalize_app_id(1.0) == norm.normalize_app_id(1) == "1"

    def test_package_names_preserved(self):
        assert norm.normalize_app_id("com.example.app") == "com.example.app"


class TestParsers:
    @pytest.mark.parametrize(
        "raw,expected",
        [("10000000", 10_000_000), ("1,000,000+", 1_000_000), ("1.5M", 1_500_000),
         ("500K", 500_000), ("", None), ("n/a", None)],
    )
    def test_parse_downloads(self, raw, expected):
        assert norm.parse_downloads(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("2025-04-03", date(2025, 4, 3)), ("2025/04/03", date(2025, 4, 3)),
         ("", None), ("not a date", None)],
    )
    def test_parse_date(self, raw, expected):
        assert norm.parse_date(raw) == expected

    def test_parse_score_clamps(self):
        assert norm.parse_score("9") == 5.0
        assert norm.parse_score("-2") == 1.0

    def test_normalize_text_strips_control_chars(self):
        assert norm.normalize_text("a\x00b\tc") == "ab c"


class TestValidators:
    def test_accepts_the_real_schema(self):
        frame = pd.DataFrame(
            {"app_id": [1], "app_name": ["A"], "description": ["d"], "score": [4.5],
             "ratings_count": [10], "downloads": [100], "categories": ["Finance"],
             "section": ["Popular apps"]}
        )
        assert validate_apps(frame).ok

    def test_missing_required_column_is_an_error(self):
        report = validate_apps(pd.DataFrame({"app_id": [1]}))
        assert not report.ok
        assert any(i.column == "app_name" for i in report.errors)

    def test_resolves_column_aliases(self):
        """A re-scrape with different headers must not require a code change."""
        frame = pd.DataFrame({"appId": [1], "title": ["A"]})
        report = validate_apps(frame)
        assert report.ok
        diag = next(c for c in report.columns if c.column == "app_name")
        assert diag.resolved_from == "title"

    def test_blank_review_text_warns_with_row_count(self):
        frame = pd.DataFrame({"app_id": [1, 2], "review_text": ["good", "  "]})
        report = validate_reviews(frame)
        assert report.ok  # a warning, not an error
        issue = next(i for i in report.issues if i.column == "review_text")
        assert issue.severity is Severity.WARNING
        assert issue.n_rows_affected == 1

    def test_unparseable_dates_are_counted(self):
        frame = pd.DataFrame(
            {"app_id": [1, 2], "review_text": ["a", "b"],
             "review_date": ["2025-01-01", "banana"]}
        )
        issue = next(i for i in validate_reviews(frame).issues if i.column == "review_date")
        assert issue.n_rows_affected == 1

    def test_raise_for_errors(self):
        from aipm.ingest.validators import DatasetValidationError

        with pytest.raises(DatasetValidationError):
            validate_apps(pd.DataFrame({"nothing": [1]})).raise_for_errors()


class TestFrameMapping:
    """The batch loader and the upload page must share one mapping.

    Two copies is how the paths drift and start deriving different review ids
    for the same row, which silently breaks evidence links.
    """

    def _reviews_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "app_id": [1, 1, 2],
            "review_text": ["it crashes on login", "payment failed twice", "other app"],
            "review_score": [1, 2, 5],
            "review_date": ["2025-01-02", "2025-03-04", "2025-01-01"],
            "helpful_count": [3, 9, 0],
        })

    def test_frame_to_apps_maps_and_normalises(self):
        from aipm.ingest.loaders import frame_to_apps

        frame = pd.DataFrame({
            "app_id": [1], "app_name": ["A"], "categories": ["#3 top free finance, Finance"],
            "score": [4.5], "downloads": ["1,000,000+"],
        })
        app = frame_to_apps(frame)["1"]
        assert app.categories == ["Finance"]
        assert app.downloads_numeric == 1_000_000

    def test_frame_to_apps_records_the_source(self):
        from aipm.ingest.loaders import frame_to_apps

        frame = pd.DataFrame({"app_id": [1], "app_name": ["A"]})
        assert frame_to_apps(frame, source="upload")["1"].source == "upload"

    def test_frame_to_reviews_filters_to_one_app(self):
        from aipm.ingest.loaders import frame_to_reviews

        assert len(frame_to_reviews(self._reviews_frame(), "1")) == 2

    def test_frame_to_reviews_orders_most_recent_first(self):
        from aipm.ingest.loaders import frame_to_reviews

        reviews = frame_to_reviews(self._reviews_frame(), "1")
        assert reviews[0].review_date > reviews[1].review_date

    def test_limit_keeps_the_most_recent(self):
        from aipm.ingest.loaders import frame_to_reviews

        reviews = frame_to_reviews(self._reviews_frame(), "1", limit=1)
        assert len(reviews) == 1 and reviews[0].review_date == date(2025, 3, 4)

    def test_review_ids_are_stable_across_calls(self):
        from aipm.ingest.loaders import frame_to_reviews

        first = [r.review_id for r in frame_to_reviews(self._reviews_frame(), "1")]
        second = [r.review_id for r in frame_to_reviews(self._reviews_frame(), "1")]
        assert first == second

    def test_blank_text_is_dropped(self):
        from aipm.ingest.loaders import frame_to_reviews

        frame = pd.DataFrame({"app_id": [1, 1], "review_text": ["real text here", "   "]})
        assert len(frame_to_reviews(frame, "1")) == 1
