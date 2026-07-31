"""Storage interface.

Everything above this line codes against the ABC, never against SQLite. That is
what lets the demo run on a single file today and move to Postgres later without
touching the pipeline or the UI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from aipm.schemas import (
    AnalysisResult,
    AnalysisRun,
    App,
    DemoAppEntry,
    DemoManifest,
    Review,
)


class Repository(ABC):
    """Persistence contract for apps, reviews, analysis runs and the demo index."""

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def init_schema(self) -> None:
        """Create tables if absent. Must be idempotent."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Repository:
        self.init_schema()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- apps --------------------------------------------------------------

    @abstractmethod
    def save_apps(self, apps: Sequence[App]) -> None: ...

    @abstractmethod
    def get_app(self, app_id: str) -> App | None: ...

    @abstractmethod
    def list_apps(self) -> list[App]: ...

    # -- reviews -----------------------------------------------------------

    @abstractmethod
    def save_reviews(self, reviews: Sequence[Review]) -> None: ...

    @abstractmethod
    def get_reviews(self, app_id: str, *, limit: int | None = None) -> list[Review]: ...

    @abstractmethod
    def count_reviews(self, app_id: str) -> int: ...

    # -- analysis runs -----------------------------------------------------

    @abstractmethod
    def save_result(self, result: AnalysisResult) -> None:
        """Persist a completed run and everything the UI needs to render it."""

    @abstractmethod
    def get_result(self, run_id: str) -> AnalysisResult | None: ...

    @abstractmethod
    def get_latest_result(self, app_id: str) -> AnalysisResult | None: ...

    @abstractmethod
    def list_runs(self, app_id: str | None = None) -> list[AnalysisRun]: ...

    @abstractmethod
    def find_run_by_params(self, app_id: str, params_hash: str) -> AnalysisRun | None:
        """Used to skip work that has already been done. Makes re-runs free."""

    @abstractmethod
    def list_catalog(self) -> list[DemoAppEntry]:
        """Every app that has a completed run, newest run per app.

        This is what the catalogue lists, deliberately *not* the demo manifest.
        The manifest records what one script produced; an app analysed through
        the upload page is equally real and must show up too.
        """

    # -- demo catalogue ----------------------------------------------------

    @abstractmethod
    def save_demo_manifest(self, manifest: DemoManifest) -> None: ...

    @abstractmethod
    def get_demo_manifest(self) -> DemoManifest | None: ...
