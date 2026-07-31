"""Upload Dataset - bring your own reviews.

Deliberately a four-step flow rather than one button: validate, preview, estimate,
then run. A PM who uploads the wrong export should find out before spending
minutes and money, and should be told *which column* is wrong.

The pipeline is never executed during a normal page render — only inside the
explicit run action, which persists its result and then re-reads it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aipm.analysis.needs import NeedService, build_need_extractor  # noqa: E402
from aipm.analysis.pipeline import AnalysisPipeline, PipelineConfig  # noqa: E402
from aipm.clustering.cluster import ClusteringConfig  # noqa: E402
from aipm.ingest import normalize as norm  # noqa: E402
from aipm.ingest.loaders import frame_to_apps, frame_to_reviews  # noqa: E402
from aipm.ingest.validators import (  # noqa: E402
    Severity, resolve_columns, validate_apps, validate_reviews,
)
from aipm.preprocess.pipeline import PreprocessConfig, ReviewPreprocessor  # noqa: E402
from aipm.schemas import AnalysisParams, App, Review  # noqa: E402
from app.components.tiles import kicker, rule, section, stat  # noqa: E402
from app.state import (  # noqa: E402
    Selection, clear_caches, get_embedding_service, get_llm_client,
    get_repository, get_settings, switch_to,
)
from app.theme import active_palette, compact_number  # noqa: E402

palette = active_palette()

MAX_PREVIEW_ROWS = 8


def render_sidebar() -> None:
    with st.sidebar:
        st.caption(
            "Two CSVs, the same shape as the bundled demo data. Column names are "
            "resolved by alias, so common export variations work unchanged."
        )
        rule()
        st.markdown("**apps_info.csv**")
        st.caption("`app_id`, `app_name`, and optionally `description`, `score`, "
                   "`ratings_count`, `downloads`, `categories`")
        st.markdown("**apps_reviews.csv**")
        st.caption("`app_id`, `review_text`, and optionally `review_score`, "
                   "`review_date`, `helpful_count`")


# ---------------------------------------------------------------------------
# Step 1 - validation
# ---------------------------------------------------------------------------


def render_report(report, frame: pd.DataFrame) -> None:
    """Per-column diagnostics. The point is to name the broken column."""
    icon = "✅" if report.ok else "🚫"
    label = "passes validation" if report.ok else "cannot be used"
    st.markdown(f"{icon} **{report.dataset}** — {len(frame):,} rows, {label}")

    st.dataframe(
        [
            {
                "column": c.column,
                "found": "yes" if c.present else "MISSING",
                "read from": c.resolved_from or ("—" if c.present else ""),
                "empty rows": c.n_null,
                "sample": " | ".join(c.sample_values)[:90],
            }
            for c in report.columns
        ],
        hide_index=True, width="stretch",
        column_config={"sample": st.column_config.TextColumn(width="large")},
    )

    for issue in report.issues:
        message = f"**{issue.column}** — {issue.message}"
        if issue.n_rows_affected:
            message += f" ({issue.n_rows_affected:,} rows)"
        if issue.severity is Severity.ERROR:
            st.error(message, icon="🚫")
        else:
            st.warning(message, icon="⚠️")


# ---------------------------------------------------------------------------
# Step 3 - the run
# ---------------------------------------------------------------------------


def build_pipeline(max_clusters: int) -> AnalysisPipeline:
    settings = get_settings()
    embedding_service = get_embedding_service()
    llm_client = get_llm_client()

    params = AnalysisParams(
        embed_model=embedding_service.provider.model,
        llm_model=llm_client.model,
        min_segment_tokens=settings.min_segment_tokens,
    )
    return AnalysisPipeline(
        preprocessor=ReviewPreprocessor(PreprocessConfig.from_settings(settings)),
        embedding_service=embedding_service,
        need_service=NeedService(
            build_need_extractor(llm_client),
            confidence_weights=settings.confidence_weights(),
            citation_threshold=settings.citation_relevance_threshold,
        ),
        params=params,
        config=PipelineConfig(
            n_representatives=settings.n_representatives,
            max_clusters_to_label=max_clusters,
        ),
        clustering_config=ClusteringConfig(
            min_cluster_size_floor=settings.min_cluster_size_floor,
            min_cluster_size_ratio=settings.min_cluster_size_ratio,
            min_clusters_before_fallback=settings.min_clusters_before_fallback,
        ),
    )


def run_analysis(app: App, reviews: list[Review], max_clusters: int) -> None:
    """Execute, persist, then invalidate the cached reads. Never inline in render."""
    repository = get_repository()
    pipeline = build_pipeline(max_clusters)
    started = time.perf_counter()

    stage_labels = {
        "preprocess": "Cleaning, deduplicating and segmenting",
        "embed": "Embedding segments",
        "reduce": "Reducing dimensions",
        "cluster": "Clustering",
        "characterise": "Extracting keywords and representatives",
        "stats": "Computing statistics and trends",
        "needs": "Reasoning over themes",
        "done": "Finished",
    }

    with st.status("Running the analysis…", expanded=True) as status:
        def progress(stage: str, detail: str) -> None:
            status.write(f"**{stage_labels.get(stage, stage)}** — {detail}")

        try:
            result, prepared_reviews = pipeline.run_with_reviews(
                app, reviews, progress=progress
            )
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.exception(exc)
            return

        status.write("**Persisting** — writing the run to storage")
        repository.save_apps([app])
        # The preprocessed reviews, so the evidence table shows real quality
        # weights and duplicate flags rather than defaults.
        repository.save_reviews(prepared_reviews)
        repository.save_result(result)
        clear_caches()

        elapsed = time.perf_counter() - started
        status.update(
            label=f"Done in {elapsed:.0f}s — {len(result.needs)} needs from "
                  f"{result.run.n_clusters} themes",
            state="complete", expanded=False,
        )

    st.success(
        f"**{app.name}** analysed: {result.run.n_reviews:,} reviews → "
        f"{result.run.n_units:,} segments → {result.run.n_clusters} themes → "
        f"{len(result.needs)} needs.",
        icon="✅",
    )
    # Same tab: the upload is finished, so this is a hand-off rather than a
    # second context. The new run travels in the URL, so the result is
    # shareable the moment it lands.
    target = Selection(app.app_id, result.run.run_id)
    columns = st.columns(2)
    with columns[0]:
        if st.button("Open the dashboard", type="primary", width="stretch"):
            switch_to("views/dashboard.py", target)
    with columns[1]:
        if st.button("See the needs", width="stretch"):
            switch_to("views/details.py", target)


# ---------------------------------------------------------------------------


def main() -> None:
    render_sidebar()

    st.title("Upload a dataset")
    st.markdown(
        '<div class="aipm-muted" style="margin-top:-0.35rem;margin-bottom:0.9rem">'
        "Validate first, then estimate, then run. Nothing is analysed until you "
        "press the button.</div>",
        unsafe_allow_html=True,
    )

    section("1 · Files")
    columns = st.columns(2)
    with columns[0]:
        apps_file = st.file_uploader("Applications CSV", type=["csv"], key="apps_csv")
    with columns[1]:
        reviews_file = st.file_uploader("Reviews CSV", type=["csv"], key="reviews_csv")

    if not apps_file or not reviews_file:
        st.info(
            "Upload both files to continue. Column names are matched by alias — "
            "`appId`, `content`, `thumbsUpCount` and similar all resolve.",
            icon="📄",
        )
        return

    try:
        apps_frame = pd.read_csv(apps_file)
        reviews_frame = pd.read_csv(reviews_file)
    except Exception as exc:
        st.error(f"Could not parse the CSVs: {type(exc).__name__}: {exc}",
                 icon="🚫")
        return

    rule()
    section("2 · Validation", "Per-column diagnostics, so a bad export names itself.")
    apps_report = validate_apps(apps_frame)
    reviews_report = validate_reviews(reviews_frame)
    render_report(apps_report, apps_frame)
    st.markdown("")
    render_report(reviews_report, reviews_frame)

    if not (apps_report.ok and reviews_report.ok):
        st.error("Fix the errors above before running an analysis.",
                 icon="🚫")
        return

    apps = frame_to_apps(apps_frame, source="upload")
    if not apps:
        st.error("No usable application rows found.", icon="🚫")
        return

    review_columns = resolve_columns(reviews_frame, ("app_id",))
    counts = (
        reviews_frame[review_columns["app_id"]]
        .map(norm.normalize_app_id).value_counts().to_dict()
    )
    selectable = {a: apps[a] for a in apps if counts.get(a, 0) > 0}
    if not selectable:
        st.error(
            "No application in the first file has any reviews in the second. "
            "Check that `app_id` matches across both.",
            icon="🔗",
        )
        return

    rule()
    section("3 · Preview and scope")

    controls = st.columns([0.45, 0.28, 0.27])
    with controls[0]:
        app_id = st.selectbox(
            "Application to analyse", list(selectable),
            format_func=lambda a: f"{selectable[a].name} — {counts.get(a, 0):,} reviews",
        )
    with controls[1]:
        available = counts.get(app_id, 0)
        max_reviews = st.slider(
            "Reviews to analyse", 200, max(200, min(available, 10_000)),
            min(available, 3_000), step=200,
            help="Most recent first. Controls runtime and cost.",
        )
    with controls[2]:
        max_clusters = st.slider(
            "Themes to reason over", 3, 20, 12,
            help="Each theme costs one model call.",
        )

    st.dataframe(
        reviews_frame[
            reviews_frame[review_columns["app_id"]].map(norm.normalize_app_id) == app_id
        ].head(MAX_PREVIEW_ROWS),
        hide_index=True, width="stretch",
    )

    # --- estimate ----------------------------------------------------------
    llm_client = get_llm_client()
    # ~2 segments per review is what the demo corpus averages after cleaning.
    estimated_units = int(max_reviews * 2.0)
    estimated_seconds = 6 + estimated_units / 900 + max_clusters * 3.5
    # ~900 prompt + ~250 completion tokens per theme, at the configured model.
    estimated_cost = 0.0
    if llm_client.available and hasattr(llm_client, "_price"):
        estimated_cost = llm_client._price(900 * max_clusters, 250 * max_clusters)

    estimate = st.columns(4)
    with estimate[0]:
        stat("Reviews", compact_number(max_reviews), sub="most recent first")
    with estimate[1]:
        stat("Segments", f"≈{compact_number(estimated_units)}", sub="after cleaning")
    with estimate[2]:
        stat("Model calls", str(max_clusters), sub="one per theme")
    with estimate[3]:
        stat("Estimated", f"${estimated_cost:.3f}",
             sub=f"≈{estimated_seconds / 60:.1f} min")

    if not llm_client.available:
        st.warning(
            "No language model is configured, so needs will be generated by the "
            "heuristic fallback and clearly labelled. Statistics and clustering "
            "are unaffected.",
            icon="⚠️",
        )

    rule()
    section("4 · Run")
    kicker("This writes a new run to storage and does not overwrite the demo data")

    if st.button("Run the analysis", type="primary", width="stretch"):
        reviews = frame_to_reviews(reviews_frame, app_id, limit=max_reviews)
        if not reviews:
            st.error("No usable review text for that application.",
                     icon="🚫")
            return
        run_analysis(selectable[app_id], reviews, max_clusters)


main()
