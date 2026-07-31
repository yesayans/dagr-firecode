"""Demo precomputation: run the pipeline per app and persist the catalogue.

Holds the *business* logic of precomputing - running an app, deciding whether it
can be skipped, converting outcomes into catalogue entries, assembling the
manifest. `scripts/precompute_demo.py` only wires dependencies and drives this.

One app failing must never abort the batch: a demo with seven apps beats a demo
with none, so failures are captured as entries with `status=FAILED` and surfaced
in the manifest.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from aipm.analysis.pipeline import AnalysisPipeline
from aipm.demo.selection import AppCandidate, DemoSelectionConfig
from aipm.schemas import (
    AnalysisResult,
    App,
    DemoAppEntry,
    DemoManifest,
    Review,
    RunStatus,
)
from aipm.storage.repository import Repository
from aipm.utils.logging import get_logger

log = get_logger(__name__)

ProgressFn = Callable[[str, str], None]


@dataclass
class AppOutcome:
    """What happened for one app. Carries the result only on success."""

    entry: DemoAppEntry
    result: AnalysisResult | None = None
    skipped: bool = False
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.entry.status is RunStatus.COMPLETE


@dataclass
class PrecomputeReport:
    outcomes: list[AppOutcome] = field(default_factory=list)
    manifest: DemoManifest | None = None
    duration_s: float = 0.0

    @property
    def n_ok(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def n_failed(self) -> int:
        return sum(1 for o in self.outcomes if not o.ok)

    @property
    def n_skipped(self) -> int:
        return sum(1 for o in self.outcomes if o.skipped)


class DemoPrecomputeService:
    """Runs the analysis pipeline for each selected app and stores the output."""

    def __init__(
        self,
        *,
        repository: Repository,
        pipeline: AnalysisPipeline,
        store_reviews: bool = True,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self.store_reviews = store_reviews

    # -- single app --------------------------------------------------------

    def precompute_app(
        self,
        app: App,
        reviews: Sequence[Review],
        candidate: AppCandidate | None = None,
        *,
        force: bool = False,
        progress: ProgressFn | None = None,
    ) -> AppOutcome:
        """Analyse one app and persist the result. Never raises."""
        started = time.perf_counter()
        base_entry = DemoAppEntry(
            app_id=app.app_id,
            app_name=app.name,
            category=(candidate.category if candidate else
                      (app.categories[0] if app.categories else "Uncategorised")),
            run_id="",
            selection_score=candidate.score if candidate else 0.0,
            selection_reasons=list(candidate.reasons) if candidate else [],
        )

        if not reviews:
            return AppOutcome(
                entry=base_entry.model_copy(
                    update={"status": RunStatus.FAILED, "error": "no reviews available"}
                ),
                duration_s=time.perf_counter() - started,
            )

        # Identical parameters over identical input produce identical output, so
        # a completed matching run can be reused outright.
        if not force:
            cached = self._reusable_run(app, reviews)
            if cached is not None:
                log.info("app %s (%s): reusing run %s", app.app_id, app.name, cached.run.run_id)
                return AppOutcome(
                    entry=self._entry_from_result(base_entry, cached),
                    result=cached,
                    skipped=True,
                    duration_s=time.perf_counter() - started,
                )

        try:
            # `run_with_reviews` hands back the *preprocessed* reviews, which are
            # the ones worth storing: they carry quality weights, language and
            # duplicate flags that the evidence drill-down displays.
            result, prepared_reviews = self.pipeline.run_with_reviews(
                app, reviews, progress=progress
            )
        except Exception as exc:
            log.error(
                "app %s (%s) failed: %s: %s\n%s",
                app.app_id, app.name, type(exc).__name__, exc, traceback.format_exc(),
            )
            return AppOutcome(
                entry=base_entry.model_copy(
                    update={
                        "status": RunStatus.FAILED,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                duration_s=time.perf_counter() - started,
            )

        try:
            self._persist(app, result, prepared_reviews)
        except Exception as exc:
            log.error("app %s: analysis succeeded but persistence failed: %s", app.app_id, exc)
            return AppOutcome(
                entry=base_entry.model_copy(
                    update={
                        "status": RunStatus.FAILED,
                        "error": f"persistence failed: {type(exc).__name__}: {exc}",
                    }
                ),
                result=result,
                duration_s=time.perf_counter() - started,
            )

        return AppOutcome(
            entry=self._entry_from_result(base_entry, result),
            result=result,
            duration_s=time.perf_counter() - started,
        )

    def _reusable_run(self, app: App, reviews: Sequence[Review]) -> AnalysisResult | None:
        params_hash = self.pipeline.params.params_hash()
        existing = self.repository.find_run_by_params(app.app_id, params_hash)
        if existing is None or existing.status is not RunStatus.COMPLETE:
            return None
        result = self.repository.get_result(existing.run_id)
        if result is None or not result.needs:
            # A stored run with no needs is not worth reusing - it is exactly the
            # case a re-run is meant to fix.
            return None
        return result

    def _persist(
        self, app: App, result: AnalysisResult, reviews: Sequence[Review]
    ) -> None:
        self.repository.save_apps([app])
        if self.store_reviews:
            # The full review text, not the truncated quotes on the evidence
            # objects: the Evidence page promises the whole review, and it
            # resolves ids against this table. Written before the run that
            # cites them so the foreign key always resolves.
            self.repository.save_reviews(reviews)
        self.repository.save_result(result)

    @staticmethod
    def _entry_from_result(entry: DemoAppEntry, result: AnalysisResult) -> DemoAppEntry:
        return entry.model_copy(
            update={
                "run_id": result.run.run_id,
                "n_reviews": result.run.n_reviews,
                "n_units": result.run.n_units,
                "n_clusters": result.run.n_clusters,
                "n_needs": len(result.needs),
                "status": result.run.status,
                "error": result.run.error,
            }
        )

    # -- batch -------------------------------------------------------------

    def build_manifest(
        self,
        outcomes: Sequence[AppOutcome],
        *,
        selection_config: DemoSelectionConfig,
        embed_backend: str,
        embed_model: str,
        embed_dim: int,
        llm_model: str,
        llm_enabled: bool,
        duration_s: float,
        warnings: Sequence[str] = (),
    ) -> DemoManifest:
        """Assemble the index the Streamlit catalogue reads."""
        entries = [o.entry for o in outcomes]
        completed = [o for o in outcomes if o.ok and o.result is not None]

        manifest = DemoManifest(
            created_at=datetime.now(timezone.utc),
            strategy=selection_config.strategy_name,
            selection_config=selection_config.as_dict(),
            embed_backend=embed_backend,
            embed_model=embed_model,
            embed_dim=embed_dim,
            llm_model=llm_model,
            llm_enabled=llm_enabled,
            entries=entries,
            total_reviews=sum(e.n_reviews for e in entries),
            total_needs=sum(e.n_needs for e in entries),
            total_cost_usd=round(sum(o.result.run.cost_usd for o in completed), 6),
            duration_s=round(duration_s, 2),
            warnings=list(warnings),
        )
        for outcome in outcomes:
            if not outcome.ok and outcome.entry.error:
                manifest.warnings.append(
                    f"{outcome.entry.app_name} ({outcome.entry.app_id}): {outcome.entry.error}"
                )
        return manifest

    def save_manifest(self, manifest: DemoManifest) -> None:
        self.repository.save_demo_manifest(manifest)
        log.info(
            "demo manifest saved: %d app(s), %d need(s), $%.4f",
            manifest.n_apps, manifest.total_needs, manifest.total_cost_usd,
        )


