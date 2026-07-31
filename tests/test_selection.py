"""Demo selection: scoring, filtering and category diversity."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aipm.demo.selection import (
    DEFAULT_WEIGHTS,
    DemoAppSelector,
    DemoSelectionConfig,
)
from aipm.schemas import App


def make_stats(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).set_index("app_id")
    for star in range(1, 6):
        column = f"n_score_{star}"
        if column not in frame:
            frame[column] = frame["n_reviews"] // 5
    return frame


def row(app_id, n_reviews=5000, share_substantive=0.8, n_months=40,
        share_helpful=0.5, avg_score=3.0, n_star_levels=5) -> dict:
    return {
        "app_id": app_id, "n_reviews": n_reviews, "avg_score": avg_score,
        "share_substantive": share_substantive, "n_months": n_months,
        "share_helpful": share_helpful, "n_star_levels": n_star_levels,
    }


def make_apps(spec: dict[str, str]) -> dict[str, App]:
    return {
        app_id: App(app_id=app_id, name=f"App {app_id}", categories=[category])
        for app_id, category in spec.items()
    }


class TestConfig:
    def test_rejects_n_apps_outside_bounds(self):
        with pytest.raises(ValueError, match="between"):
            DemoSelectionConfig(n_apps=25)

    def test_rejects_unknown_weight(self):
        with pytest.raises(ValueError, match="unknown selection weight"):
            DemoSelectionConfig(weights={"vibes": 1.0})

    def test_rejects_zero_weights(self):
        with pytest.raises(ValueError, match="positive"):
            DemoSelectionConfig(weights={k: 0.0 for k in DEFAULT_WEIGHTS})

    def test_loads_from_json(self, tmp_path):
        path = tmp_path / "strategy.json"
        path.write_text(json.dumps({
            "strategy_name": "recency-first", "n_apps": 6,
            "weights": {"volume": 0.1, "quality": 0.2, "recency": 0.5,
                        "coverage": 0.1, "engagement": 0.1},
            "exclude_app_ids": ["9"],
        }))
        config = DemoSelectionConfig.from_json(path)
        assert config.strategy_name == "recency-first"
        assert config.weights["recency"] == 0.5
        assert config.exclude_app_ids == ("9",)

    def test_cli_overrides_beat_the_file(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"n_apps": 6}))
        assert DemoSelectionConfig.from_json(path, n_apps=9).n_apps == 9

    def test_none_overrides_are_ignored(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text(json.dumps({"n_apps": 6}))
        assert DemoSelectionConfig.from_json(path, n_apps=None).n_apps == 6

    def test_as_dict_is_serialisable(self):
        json.dumps(DemoSelectionConfig().as_dict())


class TestFilters:
    def _selector(self, **kwargs) -> DemoAppSelector:
        return DemoAppSelector(DemoSelectionConfig(**kwargs))

    def test_low_volume_rejected(self):
        stats = make_stats([row("1", n_reviews=50), row("2")])
        result = self._selector(min_reviews=800).select(stats, make_apps({"1": "A", "2": "B"}))
        assert "1" not in [c.app_id for c in result.selected]

    def test_degenerate_star_coverage_rejected(self):
        """An app whose scrape holds only 1-star reviews makes a misleading demo."""
        stats = make_stats([row("1", n_star_levels=1), row("2")])
        result = self._selector(min_star_levels=3).select(
            stats, make_apps({"1": "A", "2": "B"})
        )
        rejected = {c.app_id: c.rejected_because for c in result.rejected}
        assert "star level" in rejected["1"]

    def test_thin_text_rejected(self):
        stats = make_stats([row("1", share_substantive=0.02), row("2")])
        result = self._selector(min_share_substantive=0.1).select(
            stats, make_apps({"1": "A", "2": "B"})
        )
        assert "1" in {c.app_id for c in result.rejected}

    def test_short_history_rejected(self):
        stats = make_stats([row("1", n_months=2), row("2")])
        result = self._selector(min_months_covered=6).select(
            stats, make_apps({"1": "A", "2": "B"})
        )
        assert "1" in {c.app_id for c in result.rejected}

    def test_explicit_exclusion(self):
        stats = make_stats([row(str(i)) for i in range(1, 8)])
        apps = make_apps({str(i): f"Cat{i}" for i in range(1, 8)})
        result = DemoAppSelector(
            DemoSelectionConfig(n_apps=5, exclude_app_ids=("3",))
        ).select(stats, apps)
        assert "3" not in {c.app_id for c in result.selected}

    def test_pinned_app_bypasses_quality_filters(self):
        stats = make_stats([row("1", n_reviews=10), *[row(str(i)) for i in range(2, 9)]])
        apps = make_apps({str(i): f"Cat{i}" for i in range(1, 9)})
        result = DemoAppSelector(
            DemoSelectionConfig(n_apps=5, include_app_ids=("1",))
        ).select(stats, apps)
        assert "1" in {c.app_id for c in result.selected}


class TestScoring:
    def test_higher_volume_scores_higher_all_else_equal(self):
        stats = make_stats([row("1", n_reviews=20000), row("2", n_reviews=1000)])
        selector = DemoAppSelector()
        scored = selector.score(
            selector.build_candidates(stats, make_apps({"1": "A", "2": "B"}))
        )
        assert scored[0].app_id == "1"

    def test_weights_change_the_ranking(self):
        stats = make_stats([
            row("1", n_reviews=20000, share_substantive=0.2),
            row("2", n_reviews=2000, share_substantive=0.99),
        ])
        apps = make_apps({"1": "A", "2": "B"})

        volume_first = DemoAppSelector(DemoSelectionConfig(
            weights={"volume": 0.9, "quality": 0.025, "recency": 0.025,
                     "coverage": 0.025, "engagement": 0.025}))
        quality_first = DemoAppSelector(DemoSelectionConfig(
            weights={"volume": 0.025, "quality": 0.9, "recency": 0.025,
                     "coverage": 0.025, "engagement": 0.025}))

        top_by_volume = volume_first.score(
            volume_first.build_candidates(stats, apps))[0].app_id
        top_by_quality = quality_first.score(
            quality_first.build_candidates(stats, apps))[0].app_id
        assert top_by_volume == "1" and top_by_quality == "2"

    def test_reasons_are_populated(self):
        stats = make_stats([row("1")])
        selector = DemoAppSelector()
        scored = selector.score(selector.build_candidates(stats, make_apps({"1": "A"})))
        assert any("reviews" in r for r in scored[0].reasons)

    def test_quota_capped_sample_is_disclosed(self):
        frame = make_stats([row("1", n_reviews=15000)])
        for star in range(1, 6):
            frame.loc["1", f"n_score_{star}"] = 3000
        selector = DemoAppSelector()
        candidate = selector.build_candidates(frame, make_apps({"1": "A"}))[0]
        assert candidate.quota_capped
        assert any("quota-capped" in r for r in selector.score([candidate])[0].reasons)


class TestDiversity:
    def test_prefers_distinct_categories(self):
        stats = make_stats([row(str(i), n_reviews=10000 - i * 100) for i in range(1, 9)])
        # First four apps share a category; the rest are distinct.
        apps = make_apps({"1": "Food", "2": "Food", "3": "Food", "4": "Food",
                          "5": "Games", "6": "Finance", "7": "Social", "8": "Travel"})
        result = DemoAppSelector(DemoSelectionConfig(n_apps=5)).select(stats, apps)
        categories = [c.category for c in result.selected]
        assert len(set(categories)) == 5

    def test_max_per_category_enforced(self):
        stats = make_stats([row(str(i)) for i in range(1, 9)])
        apps = make_apps({str(i): ("Food" if i <= 6 else "Games") for i in range(1, 9)})
        result = DemoAppSelector(
            DemoSelectionConfig(n_apps=5, max_per_category=2)
        ).select(stats, apps)
        assert sum(1 for c in result.selected if c.category == "Food") <= 2

    def test_respects_n_apps(self):
        stats = make_stats([row(str(i)) for i in range(1, 21)])
        apps = make_apps({str(i): f"Cat{i}" for i in range(1, 21)})
        assert len(DemoAppSelector(DemoSelectionConfig(n_apps=7)).select(stats, apps).selected) == 7

    def test_warns_when_too_few_apps_qualify(self):
        stats = make_stats([row("1"), row("2")])
        result = DemoAppSelector(DemoSelectionConfig(n_apps=8, min_apps=5)).select(
            stats, make_apps({"1": "A", "2": "B"})
        )
        assert result.warnings

    def test_empty_dataset(self):
        result = DemoAppSelector().select(make_stats([row("1")]), {})
        assert result.selected == [] and result.warnings
