#!/usr/bin/env python3
"""Precompute demo data for the Streamlit app.

Selects a representative set of apps, runs the full analysis pipeline for each,
and persists the results so the UI can open a dashboard instantly instead of
doing expensive AI work during startup.

    python scripts/precompute_demo.py
    python scripts/precompute_demo.py --n-apps 6 --max-reviews 2000
    python scripts/precompute_demo.py --strategy config/demo_strategy.json
    python scripts/precompute_demo.py --dry-run          # selection only
    python scripts/precompute_demo.py --force            # ignore cached runs

This file is orchestration only: argument parsing, dependency wiring, iteration
and reporting. Every processing step lives in a service under `src/aipm/` and is
testable without going near this script.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

# Make `src/` importable when run as a plain script (no editable install needed).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aipm.analysis.needs import NeedService, build_need_extractor  # noqa: E402
from aipm.analysis.pipeline import AnalysisPipeline, PipelineConfig  # noqa: E402
from aipm.clustering.cluster import ClusteringConfig  # noqa: E402
from aipm.config import Settings  # noqa: E402
from aipm.demo.precompute import AppOutcome, DemoPrecomputeService, PrecomputeReport  # noqa: E402
from aipm.demo.selection import DemoAppSelector, DemoSelectionConfig  # noqa: E402
from aipm.embeddings.cache import EmbeddingCache  # noqa: E402
from aipm.embeddings.provider import build_embedding_provider  # noqa: E402
from aipm.embeddings.store import EmbeddingService  # noqa: E402
from aipm.ingest.loaders import CsvReviewDataset  # noqa: E402
from aipm.ingest.validators import DatasetValidationError  # noqa: E402
from aipm.llm.client import NullLlmClient, build_llm_client  # noqa: E402
from aipm.preprocess.pipeline import PreprocessConfig, ReviewPreprocessor  # noqa: E402
from aipm.schemas import AnalysisParams  # noqa: E402
from aipm.storage.sqlite_repo import SqliteRepository  # noqa: E402
from aipm.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("precompute_demo")

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FAILED = 2
EXIT_BAD_INPUT = 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    selection = parser.add_argument_group("selection")
    selection.add_argument("--n-apps", type=int, help="how many apps to precompute (5-10)")
    selection.add_argument(
        "--strategy", type=Path, help="JSON file with a DemoSelectionConfig override"
    )
    selection.add_argument(
        "--include", nargs="*", metavar="APP_ID",
        help="always include these app ids, bypassing quality filters",
    )
    selection.add_argument(
        "--exclude", nargs="*", metavar="APP_ID", help="never include these app ids"
    )
    selection.add_argument(
        "--max-per-category", type=int, help="cap on apps sharing a primary category"
    )

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument(
        "--max-reviews", type=int,
        help="cap reviews analysed per app (most recent win). Controls runtime.",
    )
    pipeline.add_argument(
        "--max-clusters", type=int, help="cap clusters sent to the LLM per app"
    )
    pipeline.add_argument(
        "--embed-backend", choices=("api", "local", "fixture"),
        help="override EMBED_BACKEND",
    )
    pipeline.add_argument(
        "--require-llm", action="store_true",
        help="fail instead of falling back to heuristic need extraction",
    )
    pipeline.add_argument(
        "--skip-llm-healthcheck", action="store_true",
        help="skip the preflight probe (useful if the endpoint rejects tiny requests)",
    )
    pipeline.add_argument(
        "--random-state", type=int, default=None, help="seed for reproducible runs"
    )

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument(
        "--force", action="store_true", help="recompute even if a matching run exists"
    )
    behaviour.add_argument(
        "--dry-run", action="store_true",
        help="show the selection and exit without running the pipeline",
    )
    behaviour.add_argument(
        "--fail-fast", action="store_true", help="stop at the first app that fails"
    )
    behaviour.add_argument("--db", type=Path, help="override the SQLite path")
    behaviour.add_argument(
        "-v", "--verbose", action="store_true", help="debug logging"
    )
    behaviour.add_argument(
        "-q", "--quiet", action="store_true", help="warnings and errors only"
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


def build_selection_config(
    settings: Settings, args: argparse.Namespace
) -> DemoSelectionConfig:
    """Layer the strategy: defaults <- settings <- JSON file <- CLI flags."""
    overrides = {
        "n_apps": args.n_apps,
        "max_per_category": args.max_per_category,
        "include_app_ids": tuple(args.include) if args.include else None,
        "exclude_app_ids": tuple(args.exclude) if args.exclude else None,
    }
    if args.strategy:
        if not args.strategy.exists():
            raise DatasetValidationError(f"strategy file not found: {args.strategy}")
        return DemoSelectionConfig.from_json(args.strategy, **overrides)
    return DemoSelectionConfig.from_settings(settings, **overrides)


def build_pipeline(
    settings: Settings, args: argparse.Namespace
) -> tuple[AnalysisPipeline, EmbeddingService, object]:
    """Construct the pipeline and hand back the pieces the report needs to name."""
    params = AnalysisParams(
        random_state=args.random_state if args.random_state is not None else 42,
        min_segment_tokens=settings.min_segment_tokens,
        llm_model=settings.llm_model,
    )

    embedding_provider = build_embedding_provider(settings)
    embedding_service = EmbeddingService(
        embedding_provider,
        EmbeddingCache(settings.embedding_cache_dir / "vectors.sqlite"),
    )

    llm_client = build_llm_client(settings)

    # Probe once, before analysing anything. A wrong key would otherwise only
    # surface per cluster, after full retry backoff, for every app in the batch.
    if llm_client.available and not args.skip_llm_healthcheck:
        healthy, reason = llm_client.healthcheck()
        if not healthy:
            log.error("LLM healthcheck failed: %s", reason)
            if args.require_llm:
                raise DatasetValidationError(
                    f"--require-llm was passed but the endpoint is unreachable.\n"
                    f"  base_url: {settings.llm_base_url}\n"
                    f"  model:    {settings.llm_model}\n"
                    f"  error:    {reason}"
                )
            llm_client = NullLlmClient(f"healthcheck failed - {reason}")
        else:
            log.info("LLM healthcheck passed: %s", llm_client.model)

    if args.require_llm and not llm_client.available:
        raise DatasetValidationError(
            f"--require-llm was passed but no LLM is available: "
            f"{getattr(llm_client, 'reason', 'unknown')}"
        )

    need_service = NeedService(
        build_need_extractor(llm_client, allow_heuristic=not args.require_llm),
        confidence_weights=settings.confidence_weights(),
        citation_threshold=settings.citation_relevance_threshold,
    )

    pipeline = AnalysisPipeline(
        preprocessor=ReviewPreprocessor(PreprocessConfig.from_settings(settings)),
        embedding_service=embedding_service,
        need_service=need_service,
        params=params.model_copy(update={"embed_model": embedding_provider.model}),
        config=PipelineConfig(
            n_representatives=settings.n_representatives,
            max_clusters_to_label=args.max_clusters or settings.max_clusters_to_label,
            citation_threshold=settings.citation_relevance_threshold,
        ),
        clustering_config=ClusteringConfig(
            min_cluster_size_floor=settings.min_cluster_size_floor,
            min_cluster_size_ratio=settings.min_cluster_size_ratio,
            min_clusters_before_fallback=settings.min_clusters_before_fallback,
            random_state=params.random_state,
        ),
    )
    return pipeline, embedding_service, llm_client


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Single-line-per-stage progress. Readable in a terminal and in a log file."""

    STAGES = ("preprocess", "embed", "reduce", "cluster", "characterise", "stats", "needs")

    def __init__(self, total_apps: int, *, enabled: bool = True) -> None:
        self.total_apps = total_apps
        self.enabled = enabled
        self.index = 0
        self.app_name = ""
        self._started = time.perf_counter()

    def start_app(self, index: int, app_name: str, app_id: str) -> None:
        self.index = index
        self.app_name = app_name
        if self.enabled:
            print(
                f"\n[{index}/{self.total_apps}] {app_name} (id={app_id})",
                flush=True,
            )

    def __call__(self, stage: str, detail: str) -> None:
        if not self.enabled:
            return
        marker = "*" if stage != "done" else "="
        print(f"    {marker} {stage:<13} {detail}", flush=True)

    def finish_app(self, outcome: AppOutcome) -> None:
        if not self.enabled:
            return
        entry = outcome.entry
        if outcome.skipped:
            print(f"    = cached        run {entry.run_id} reused", flush=True)
        elif outcome.ok:
            print(
                f"    = complete      {entry.n_clusters} clusters, {entry.n_needs} needs "
                f"in {outcome.duration_s:.1f}s",
                flush=True,
            )
        else:
            print(f"    ! FAILED        {entry.error}", flush=True)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_selection(result, quiet: bool = False) -> None:
    if quiet:
        return
    print("\n" + "=" * 78)
    print(f"SELECTION  -  {result.summary_line()}")
    print("=" * 78)
    header = f"{'score':>6}  {'id':>5}  {'app':<32} {'category':<18} {'reviews':>8}"
    print(header)
    print("-" * 78)
    for candidate in result.selected:
        print(
            f"{candidate.score:>6.3f}  {candidate.app_id:>5}  {candidate.name[:32]:<32} "
            f"{candidate.category[:18]:<18} {candidate.n_reviews:>8,}"
        )
        for reason in candidate.reasons:
            print(f"{'':>8}- {reason}")
    if result.rejected:
        print(f"\nRejected ({len(result.rejected)}):")
        for candidate in result.rejected[:10]:
            print(f"  {candidate.app_id:>5}  {candidate.name[:34]:<34} {candidate.rejected_because}")
    for warning in result.warnings:
        print(f"\n  ! {warning}")


def print_summary(report: PrecomputeReport, settings: Settings) -> None:
    manifest = report.manifest
    print("\n" + "=" * 78)
    print("PRECOMPUTE SUMMARY")
    print("=" * 78)
    print(f"  apps succeeded : {report.n_ok}")
    print(f"  apps failed    : {report.n_failed}")
    print(f"  apps reused    : {report.n_skipped}")
    if manifest:
        print(f"  reviews        : {manifest.total_reviews:,}")
        print(f"  needs          : {manifest.total_needs}")
        print(f"  embeddings     : {manifest.embed_backend} / {manifest.embed_model} "
              f"({manifest.embed_dim}d)")
        print(f"  LLM            : {manifest.llm_model} "
              f"({'enabled' if manifest.llm_enabled else 'DISABLED - heuristic needs'})")
        print(f"  cost           : ${manifest.total_cost_usd:.4f}")
    print(f"  duration       : {report.duration_s:.1f}s")
    print(f"  database       : {settings.sqlite_path}")

    for outcome in report.outcomes:
        if not outcome.ok:
            print(f"\n  ! {outcome.entry.app_name}: {outcome.entry.error}")

    if manifest and manifest.warnings:
        print(f"\n  warnings ({len(manifest.warnings)}):")
        for warning in manifest.warnings[:10]:
            print(f"    - {warning}")

    print("\nThe Streamlit app can now read these runs without calling any AI service.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(
        logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    )

    settings = Settings()
    if args.db:
        settings = settings.model_copy(update={"sqlite_path": args.db})
    if args.embed_backend:
        settings = settings.model_copy(update={"embed_backend": args.embed_backend})
    if args.max_reviews:
        settings = settings.model_copy(update={"demo_max_reviews_per_app": args.max_reviews})
    settings.ensure_dirs()

    started = time.perf_counter()

    # 1. Load and validate the datasets --------------------------------------
    try:
        dataset = CsvReviewDataset.from_settings(settings)
        apps_report, reviews_report = dataset.validate()
        log.info("%s", apps_report.summary_line())
        log.info("%s", reviews_report.summary_line())
        for issue in (*apps_report.issues, *reviews_report.issues):
            log.warning("%s.%s: %s (%d rows)",
                        issue.severity.value, issue.column, issue.message, issue.n_rows_affected)
        apps_report.raise_for_errors()
        reviews_report.raise_for_errors()

        apps = dataset.load_apps()
        stats = dataset.app_review_stats()
        recent_shares = dataset.recent_review_shares(
            window_days=DemoSelectionConfig().recency_window_days
        ).to_dict()
    except DatasetValidationError as exc:
        log.error("dataset unusable: %s", exc)
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            f"Expected inputs:\n  {settings.apps_csv}\n  {settings.reviews_csv}",
            file=sys.stderr,
        )
        return EXIT_BAD_INPUT
    except Exception as exc:
        log.exception("failed to load datasets")
        print(f"\nERROR loading datasets: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    # 2. Select the demo apps ------------------------------------------------
    try:
        selection_config = build_selection_config(settings, args)
    except (DatasetValidationError, ValueError) as exc:
        print(f"\nERROR in selection config: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    selection = DemoAppSelector(selection_config).select(
        stats, apps, recent_shares=recent_shares
    )
    print_selection(selection, quiet=args.quiet)

    if not selection.selected:
        print("\nERROR: no apps passed selection. Loosen the filters.", file=sys.stderr)
        return EXIT_FAILED
    if args.dry_run:
        print("\nDry run - nothing was computed or written.")
        return EXIT_OK

    # 3. Wire the pipeline ---------------------------------------------------
    try:
        pipeline, embedding_service, llm_client = build_pipeline(settings, args)
    except DatasetValidationError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except Exception as exc:
        log.exception("failed to build the pipeline")
        print(f"\nERROR building pipeline: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    if not llm_client.available:
        log.warning(
            "no LLM configured (%s) - needs will be templated by the heuristic extractor",
            getattr(llm_client, "reason", "unknown"),
        )

    # 4. Load reviews for the selected apps in one pass -----------------------
    selected_ids = [c.app_id for c in selection.selected]
    log.info(
        "loading reviews for %d app(s), capped at %d each",
        len(selected_ids), settings.demo_max_reviews_per_app,
    )
    try:
        reviews_by_app = dataset.load_reviews_for(
            selected_ids, limit_per_app=settings.demo_max_reviews_per_app
        )
    except Exception as exc:
        log.exception("failed to load reviews")
        print(f"\nERROR loading reviews: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    # 5. Run the pipeline per app and persist --------------------------------
    report = PrecomputeReport()
    progress = ProgressReporter(len(selection.selected), enabled=not args.quiet)

    with SqliteRepository(settings.sqlite_path) as repository:
        service = DemoPrecomputeService(repository=repository, pipeline=pipeline)

        for index, candidate in enumerate(selection.selected, start=1):
            app = apps[candidate.app_id]
            progress.start_app(index, app.name, app.app_id)

            outcome = service.precompute_app(
                app,
                reviews_by_app.get(candidate.app_id, []),
                candidate,
                force=args.force,
                progress=progress,
            )
            report.outcomes.append(outcome)
            progress.finish_app(outcome)

            if not outcome.ok and args.fail_fast:
                log.error("stopping after first failure (--fail-fast)")
                break

        report.duration_s = time.perf_counter() - started
        provider = embedding_service.provider
        report.manifest = service.build_manifest(
            report.outcomes,
            selection_config=selection_config,
            embed_backend=settings.embed_backend,
            embed_model=provider.model,
            embed_dim=provider.dim,
            llm_model=llm_client.model,
            llm_enabled=llm_client.available,
            duration_s=report.duration_s,
            warnings=selection.warnings,
        )
        service.save_manifest(report.manifest)

    print_summary(report, settings)

    if report.n_ok == 0:
        return EXIT_FAILED
    return EXIT_PARTIAL if report.n_failed else EXIT_OK


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
