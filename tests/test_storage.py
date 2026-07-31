"""Storage: SQLite repository round-trips."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from aipm.schemas import (
    AnalysisParams,
    AnalysisResult,
    AnalysisRun,
    App,
    Cluster,
    ConfidenceBreakdown,
    DemoAppEntry,
    DemoManifest,
    Evidence,
    Need,
    NeedCategory,
    OverviewStats,
    PriorityScore,
    Review,
    RunStatus,
)
from aipm.storage.sqlite_repo import SqliteRepository


@pytest.fixture
def repo(tmp_path):
    with SqliteRepository(tmp_path / "test.db") as repository:
        yield repository


def make_result(app: App, *, run_id: str = "run1", params_hash: str = "h1") -> AnalysisResult:
    return AnalysisResult(
        run=AnalysisRun(
            run_id=run_id, app_id=app.app_id, params_hash=params_hash,
            params=AnalysisParams(), status=RunStatus.COMPLETE,
            n_reviews=10, n_units=20, n_clusters=2,
            started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        ),
        app=app,
        stats=OverviewStats(n_reviews=10, avg_score=3.2, store_score=4.5),
        clusters=[Cluster(cluster_id="c1", run_id=run_id, size=5, keywords=["crash"])],
        needs=[
            Need(
                need_id="n1", run_id=run_id,
                statement="Users need reliable order confirmation",
                underlying_goal="place an order once",
                category=NeedCategory.RELIABILITY,
                cluster_ids=["c1"],
                evidence=[Evidence(review_id="r1", quote="it failed", validated=True)],
                confidence=ConfidenceBreakdown(support=0.8, total=0.72),
                priority=PriorityScore(reach=0.3, impact=1.2, value_score=0.26, rank=1),
                hiddenness=0.6,
            )
        ],
    )


class TestApps:
    def test_round_trip(self, repo, app):
        repo.save_apps([app])
        got = repo.get_app("app1")
        assert got.name == "Test App" and got.categories == ["Food & Drink"]

    def test_upsert_is_idempotent(self, repo, app):
        repo.save_apps([app])
        repo.save_apps([app.model_copy(update={"name": "Renamed"})])
        assert len(repo.list_apps()) == 1
        assert repo.get_app("app1").name == "Renamed"

    def test_missing_app(self, repo):
        assert repo.get_app("nope") is None


class TestReviews:
    def test_round_trip_preserves_fields(self, repo, app):
        repo.save_apps([app])
        repo.save_reviews([
            Review(review_id="r1", app_id="app1", text="full review text here",
                   score=2, review_date=date(2024, 5, 1), helpful_count=7,
                   lang="en", quality_weight=0.8)
        ])
        got = repo.get_reviews("app1")[0]
        assert got.text == "full review text here"
        assert got.review_date == date(2024, 5, 1)
        assert got.helpful_count == 7
        assert got.quality_weight == 0.8

    def test_count(self, repo, app):
        repo.save_apps([app])
        repo.save_reviews([
            Review(review_id=f"r{i}", app_id="app1", text=f"t{i}") for i in range(5)
        ])
        assert repo.count_reviews("app1") == 5

    def test_empty_save_is_a_noop(self, repo):
        repo.save_reviews([])


class TestRuns:
    def test_result_round_trip(self, repo, app):
        repo.save_apps([app])
        repo.save_result(make_result(app))
        got = repo.get_result("run1")
        assert got.needs[0].statement == "Users need reliable order confirmation"
        assert got.stats.store_score == 4.5

    def test_latest_result_for_app(self, repo, app):
        repo.save_apps([app])
        repo.save_result(make_result(app))
        assert repo.get_latest_result("app1").run.run_id == "run1"

    def test_find_by_params_enables_free_reruns(self, repo, app):
        repo.save_apps([app])
        repo.save_result(make_result(app))
        assert repo.find_run_by_params("app1", "h1").run_id == "run1"
        assert repo.find_run_by_params("app1", "other") is None

    def test_resaving_a_run_does_not_duplicate_children(self, repo, app):
        repo.save_apps([app])
        repo.save_result(make_result(app))
        repo.save_result(make_result(app))
        rows = repo._conn.execute(
            "SELECT COUNT(*) AS n FROM need_evidence"
        ).fetchone()["n"]
        assert rows == 1

    def test_needs_are_queryable_not_only_in_the_blob(self, repo, app):
        repo.save_apps([app])
        repo.save_result(make_result(app))
        row = repo._conn.execute(
            "SELECT statement, confidence_total FROM needs WHERE run_id = 'run1'"
        ).fetchone()
        assert row["confidence_total"] == pytest.approx(0.72)

    def test_computed_fields_survive_serialisation(self, repo, app):
        repo.save_apps([app])
        repo.save_result(make_result(app))
        need = repo.get_result("run1").needs[0]
        assert need.insight_score == pytest.approx(0.6 * 0.72)
        assert need.confidence.band == "high"

    def test_missing_run(self, repo):
        assert repo.get_result("nope") is None


class TestDemoManifest:
    def test_round_trip(self, repo):
        manifest = DemoManifest(
            strategy="default", embed_model="m", embed_dim=384, llm_model="x",
            entries=[DemoAppEntry(app_id="app1", app_name="A", run_id="run1", n_needs=3)],
        )
        repo.save_demo_manifest(manifest)
        got = repo.get_demo_manifest()
        assert got.n_apps == 1 and got.entries[0].n_needs == 3

    def test_overwrites_previous(self, repo):
        repo.save_demo_manifest(DemoManifest(strategy="a"))
        repo.save_demo_manifest(DemoManifest(strategy="b"))
        assert repo.get_demo_manifest().strategy == "b"

    def test_absent_manifest(self, repo):
        assert repo.get_demo_manifest() is None

    def test_failed_entries_excluded_from_n_apps(self, repo):
        manifest = DemoManifest(entries=[
            DemoAppEntry(app_id="a", app_name="A", run_id="r1"),
            DemoAppEntry(app_id="b", app_name="B", run_id="", status=RunStatus.FAILED),
        ])
        repo.save_demo_manifest(manifest)
        assert repo.get_demo_manifest().n_apps == 1


class TestThreadSafety:
    """Streamlit caches one repository across script-runner threads.

    A single shared sqlite3 connection raises `ProgrammingError` the moment a
    second thread touches it, which is how this surfaced: every page load after
    the first died.
    """

    def test_usable_from_another_thread(self, tmp_path, app):
        import threading

        repository = SqliteRepository(tmp_path / "threads.db")
        repository.init_schema()
        repository.save_apps([app])

        errors: list[Exception] = []
        results: list[int] = []

        def read() -> None:
            try:
                results.append(len(repository.list_apps()))
            except Exception as exc:  # noqa: BLE001 - the assertion is the point
                errors.append(exc)

        thread = threading.Thread(target=read)
        thread.start()
        thread.join()

        assert not errors, f"cross-thread read failed: {errors[0]}"
        assert results == [1]
        repository.close()

    def test_concurrent_writes_from_many_threads(self, tmp_path, app):
        import threading

        repository = SqliteRepository(tmp_path / "threads.db")
        repository.init_schema()
        repository.save_apps([app])

        errors: list[Exception] = []

        def write(index: int) -> None:
            try:
                repository.save_reviews([
                    Review(review_id=f"r{index}", app_id=app.app_id, text=f"review {index}")
                ])
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"concurrent write failed: {errors[0]}"
        assert repository.count_reviews(app.app_id) == 8
        repository.close()
