"""SQLite implementation of `Repository`.

Postgres-shaped DDL so the same statements port over. Two deliberate choices:

* Clusters, needs and evidence are written to real tables *and* the whole
  `AnalysisResult` is stored as one JSON blob on `analysis_runs`. The tables make
  the data queryable; the blob makes a dashboard load a single row read, which is
  the entire point of precomputing.
* `UNIQUE (app_id, params_hash)` means re-running identical parameters is free.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from aipm.schemas import (
    AnalysisResult,
    AnalysisRun,
    App,
    DemoManifest,
    Review,
    RunStatus,
)
from aipm.storage.repository import Repository
from aipm.utils.logging import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS apps (
    app_id            TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT DEFAULT '',
    score             REAL,
    ratings_count     INTEGER,
    downloads_raw     TEXT,
    downloads_numeric INTEGER,
    categories        TEXT DEFAULT '[]',
    source            TEXT DEFAULT 'seed',
    created_at        TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id      TEXT PRIMARY KEY,
    app_id         TEXT NOT NULL REFERENCES apps(app_id),
    text           TEXT NOT NULL,
    score          INTEGER,
    review_date    TEXT,
    helpful_count  INTEGER DEFAULT 0,
    lang           TEXT,
    is_duplicate   INTEGER DEFAULT 0,
    quality_weight REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_reviews_app_date ON reviews(app_id, review_date);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id              TEXT PRIMARY KEY,
    app_id              TEXT NOT NULL REFERENCES apps(app_id),
    params_hash         TEXT NOT NULL,
    params              TEXT DEFAULT '{}',
    status              TEXT NOT NULL,
    n_reviews           INTEGER DEFAULT 0,
    n_units             INTEGER DEFAULT 0,
    n_clusters          INTEGER DEFAULT 0,
    noise_ratio         REAL DEFAULT 0.0,
    clustering_fallback INTEGER DEFAULT 0,
    citations_dropped   INTEGER DEFAULT 0,
    cost_usd            REAL DEFAULT 0.0,
    started_at          TEXT,
    finished_at         TEXT,
    error               TEXT,
    result_blob         TEXT,
    UNIQUE (app_id, params_hash)
);
CREATE INDEX IF NOT EXISTS idx_runs_app ON analysis_runs(app_id, finished_at);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id     TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    label          TEXT DEFAULT '',
    summary        TEXT DEFAULT '',
    keywords       TEXT DEFAULT '[]',
    size           INTEGER DEFAULT 0,
    persistence    REAL DEFAULT 0.0,
    cohesion       REAL DEFAULT 0.0,
    separation     REAL DEFAULT 0.0,
    medoid_unit_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_clusters_run ON clusters(run_id);

CREATE TABLE IF NOT EXISTS needs (
    need_id               TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    statement             TEXT NOT NULL,
    underlying_goal       TEXT DEFAULT '',
    category              TEXT DEFAULT 'other',
    surface_complaints    TEXT DEFAULT '[]',
    workarounds           TEXT DEFAULT '[]',
    cluster_ids           TEXT DEFAULT '[]',
    hiddenness            REAL DEFAULT 0.0,
    confidence_total      REAL DEFAULT 0.0,
    confidence_components TEXT DEFAULT '{}',
    reach                 REAL DEFAULT 0.0,
    impact                REAL DEFAULT 0.0,
    value_score           REAL DEFAULT 0.0,
    rank                  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_needs_run ON needs(run_id);

CREATE TABLE IF NOT EXISTS need_evidence (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    need_id   TEXT NOT NULL REFERENCES needs(need_id) ON DELETE CASCADE,
    review_id TEXT NOT NULL,
    quote     TEXT DEFAULT '',
    relevance REAL DEFAULT 0.0,
    validated INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_evidence_need ON need_evidence(need_id);

CREATE TABLE IF NOT EXISTS demo_manifest (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    payload    TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class SqliteRepository(Repository):
    """Thread-safe by construction: one connection per thread.

    A single shared connection is not an option. Streamlit runs each rerun on a
    script-runner thread and caches this object across all of them, and SQLite
    refuses a connection used off its creating thread. `check_same_thread=False`
    would silence the error while leaving concurrent writes genuinely unsafe, so
    each thread gets its own connection instead. WAL mode lets readers proceed
    while a writer holds the database.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connect()  # fail fast if the path is unusable

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "conn", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.conn = connection
        return connection

    @property
    def _conn(self) -> sqlite3.Connection:
        """This thread's connection. Keeps every call site below unchanged."""
        return self._connect()

    # -- lifecycle ---------------------------------------------------------

    def init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        connection = getattr(self._local, "conn", None)
        if connection is not None:
            connection.close()
            self._local.conn = None

    # -- apps --------------------------------------------------------------

    def save_apps(self, apps: Sequence[App]) -> None:
        if not apps:
            return
        rows = [
            (
                a.app_id, a.name, a.description, a.score, a.ratings_count,
                a.downloads_raw, a.downloads_numeric, json.dumps(a.categories), a.source,
            )
            for a in apps
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO apps (app_id, name, description, score, "
                "ratings_count, downloads_raw, downloads_numeric, categories, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def get_app(self, app_id: str) -> App | None:
        row = self._conn.execute("SELECT * FROM apps WHERE app_id = ?", (app_id,)).fetchone()
        return self._row_to_app(row) if row else None

    def list_apps(self) -> list[App]:
        rows = self._conn.execute("SELECT * FROM apps ORDER BY name").fetchall()
        return [self._row_to_app(r) for r in rows]

    @staticmethod
    def _row_to_app(row: sqlite3.Row) -> App:
        return App(
            app_id=row["app_id"],
            name=row["name"],
            description=row["description"] or "",
            score=row["score"],
            ratings_count=row["ratings_count"],
            downloads_raw=row["downloads_raw"],
            downloads_numeric=row["downloads_numeric"],
            categories=json.loads(row["categories"] or "[]"),
            source=row["source"] or "seed",
        )

    # -- reviews -----------------------------------------------------------

    def save_reviews(self, reviews: Sequence[Review]) -> None:
        if not reviews:
            return
        rows = [
            (
                r.review_id, r.app_id, r.text, r.score,
                r.review_date.isoformat() if r.review_date else None,
                r.helpful_count, r.lang, int(r.is_duplicate), r.quality_weight,
            )
            for r in reviews
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO reviews (review_id, app_id, text, score, "
                "review_date, helpful_count, lang, is_duplicate, quality_weight) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def get_reviews(self, app_id: str, *, limit: int | None = None) -> list[Review]:
        sql = "SELECT * FROM reviews WHERE app_id = ? ORDER BY review_date DESC"
        params: tuple[object, ...] = (app_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (app_id, limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            Review(
                review_id=r["review_id"],
                app_id=r["app_id"],
                text=r["text"],
                score=r["score"],
                review_date=_parse_date(r["review_date"]),
                helpful_count=r["helpful_count"] or 0,
                lang=r["lang"],
                is_duplicate=bool(r["is_duplicate"]),
                quality_weight=r["quality_weight"] if r["quality_weight"] is not None else 1.0,
            )
            for r in rows
        ]

    def count_reviews(self, app_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM reviews WHERE app_id = ?", (app_id,)
        ).fetchone()
        return int(row["n"])

    # -- analysis runs -----------------------------------------------------

    def save_result(self, result: AnalysisResult) -> None:
        run = result.run
        with self._conn:
            # Replacing a run must not leave the previous run's children behind.
            self._conn.execute("DELETE FROM clusters WHERE run_id = ?", (run.run_id,))
            self._conn.execute(
                "DELETE FROM need_evidence WHERE need_id IN "
                "(SELECT need_id FROM needs WHERE run_id = ?)",
                (run.run_id,),
            )
            self._conn.execute("DELETE FROM needs WHERE run_id = ?", (run.run_id,))

            self._conn.execute(
                "INSERT OR REPLACE INTO analysis_runs (run_id, app_id, params_hash, params, "
                "status, n_reviews, n_units, n_clusters, noise_ratio, clustering_fallback, "
                "citations_dropped, cost_usd, started_at, finished_at, error, result_blob) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.run_id, run.app_id, run.params_hash, run.params.model_dump_json(),
                    run.status.value, run.n_reviews, run.n_units, run.n_clusters,
                    run.noise_ratio, int(run.clustering_fallback), run.citations_dropped,
                    run.cost_usd,
                    run.started_at.isoformat() if run.started_at else None,
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.error,
                    result.model_dump_json(),
                ),
            )

            self._conn.executemany(
                "INSERT OR REPLACE INTO clusters (cluster_id, run_id, label, summary, keywords, "
                "size, persistence, cohesion, separation, medoid_unit_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.cluster_id, c.run_id, c.label, c.summary, json.dumps(c.keywords),
                        c.size, c.persistence, c.cohesion, c.separation, c.medoid_unit_id,
                    )
                    for c in result.clusters
                ],
            )

            self._conn.executemany(
                "INSERT OR REPLACE INTO needs (need_id, run_id, statement, underlying_goal, "
                "category, surface_complaints, workarounds, cluster_ids, hiddenness, "
                "confidence_total, confidence_components, reach, impact, value_score, rank) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        n.need_id, n.run_id, n.statement, n.underlying_goal, n.category.value,
                        json.dumps(n.surface_complaints), json.dumps(n.workarounds),
                        json.dumps(n.cluster_ids), n.hiddenness, n.confidence.total,
                        n.confidence.model_dump_json(), n.priority.reach, n.priority.impact,
                        n.priority.value_score, n.priority.rank,
                    )
                    for n in result.needs
                ],
            )

            self._conn.executemany(
                "INSERT INTO need_evidence (need_id, review_id, quote, relevance, validated) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (n.need_id, e.review_id, e.quote, e.relevance, int(e.validated))
                    for n in result.needs
                    for e in n.evidence
                ],
            )
        log.info(
            "persisted run %s (app=%s, %d clusters, %d needs)",
            run.run_id, run.app_id, len(result.clusters), len(result.needs),
        )

    def get_result(self, run_id: str) -> AnalysisResult | None:
        row = self._conn.execute(
            "SELECT result_blob FROM analysis_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None or not row["result_blob"]:
            return None
        return AnalysisResult.model_validate_json(row["result_blob"])

    def get_latest_result(self, app_id: str) -> AnalysisResult | None:
        row = self._conn.execute(
            "SELECT result_blob FROM analysis_runs WHERE app_id = ? AND status = ? "
            "AND result_blob IS NOT NULL ORDER BY finished_at DESC LIMIT 1",
            (app_id, RunStatus.COMPLETE.value),
        ).fetchone()
        if row is None:
            return None
        return AnalysisResult.model_validate_json(row["result_blob"])

    def list_runs(self, app_id: str | None = None) -> list[AnalysisRun]:
        if app_id:
            rows = self._conn.execute(
                "SELECT * FROM analysis_runs WHERE app_id = ? ORDER BY finished_at DESC",
                (app_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM analysis_runs ORDER BY finished_at DESC"
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def find_run_by_params(self, app_id: str, params_hash: str) -> AnalysisRun | None:
        row = self._conn.execute(
            "SELECT * FROM analysis_runs WHERE app_id = ? AND params_hash = ?",
            (app_id, params_hash),
        ).fetchone()
        return self._row_to_run(row) if row else None

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> AnalysisRun:
        from aipm.schemas import AnalysisParams

        return AnalysisRun(
            run_id=row["run_id"],
            app_id=row["app_id"],
            params_hash=row["params_hash"],
            params=AnalysisParams.model_validate_json(row["params"] or "{}"),
            status=RunStatus(row["status"]),
            n_reviews=row["n_reviews"] or 0,
            n_units=row["n_units"] or 0,
            n_clusters=row["n_clusters"] or 0,
            noise_ratio=row["noise_ratio"] or 0.0,
            clustering_fallback=bool(row["clustering_fallback"]),
            citations_dropped=row["citations_dropped"] or 0,
            cost_usd=row["cost_usd"] or 0.0,
            started_at=_parse_datetime(row["started_at"]),
            finished_at=_parse_datetime(row["finished_at"]),
            error=row["error"],
        )

    # -- demo catalogue ----------------------------------------------------

    def save_demo_manifest(self, manifest: DemoManifest) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO demo_manifest (id, payload, updated_at) "
                "VALUES (1, ?, CURRENT_TIMESTAMP)",
                (manifest.model_dump_json(),),
            )

    def get_demo_manifest(self) -> DemoManifest | None:
        row = self._conn.execute("SELECT payload FROM demo_manifest WHERE id = 1").fetchone()
        return DemoManifest.model_validate_json(row["payload"]) if row else None


def _parse_date(value: str | None):
    from datetime import date

    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_datetime(value: str | None):
    from datetime import datetime

    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
