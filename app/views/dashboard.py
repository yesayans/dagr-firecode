"""Dashboard - the quantitative view of one application.

Nothing on this page comes from the language model. It renders from statistics
computed in Python, so it stays correct and complete even when no model endpoint
is reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.charts import (  # noqa: E402
    confidence_histogram, helpful_votes_curve, rating_distribution,
    rating_trend, value_map, volume_trend,
)
from app.components.tiles import (  # noqa: E402
    confidence_pill, kicker, rule, sampling_caveat, section, stat,
)
from app.state import load_reviews, require_selection, switch_to  # noqa: E402
from app.theme import active_palette, chart_config, compact_number  # noqa: E402

palette = active_palette()


def render_sidebar(result) -> None:
    with st.sidebar:
        st.caption(" · ".join(result.app.categories) or "Uncategorised")
        run = result.run
        st.markdown("**This run**")
        st.caption(f"`{run.run_id}`")
        st.caption(f"{run.n_reviews:,} reviews → {run.n_units:,} segments")
        st.caption(f"{run.n_clusters} themes · {run.noise_ratio:.0%} unclustered")
        if run.clustering_fallback:
            st.caption("ℹ️ density clustering fell back — themes are approximate")
        if run.citations_dropped:
            st.caption(f"⚖️ {run.citations_dropped} citations dropped by the guard")
        st.caption(f"Embeddings `{run.params.embed_model.split('/')[-1]}`")


def main() -> None:
    result = require_selection("The dashboard", route="Dashboard")
    if result is None:
        return

    render_sidebar(result)
    stats = result.stats

    st.title(result.app.name)
    st.markdown(
        f'<div class="aipm-muted" style="margin-top:-0.35rem;margin-bottom:0.9rem">'
        f"{' · '.join(result.app.categories)}"
        f"{' · ' + compact_number(result.app.downloads_numeric) + ' downloads' if result.app.downloads_numeric else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- KPI row -----------------------------------------------------------
    columns = st.columns(5)
    with columns[0]:
        stat("Store rating", f"{stats.store_score or '—'}★", sub="as published")
    with columns[1]:
        stat(
            "Sample mean", f"{stats.avg_score:.2f}★",
            sub="of the analysed reviews",
        )
    with columns[2]:
        stat("Reviews analysed", compact_number(stats.n_reviews),
             sub=f"{result.run.n_units:,} segments")
    with columns[3]:
        stat("Negative share", f"{stats.pct_negative:.0%}", sub="1–2 star reviews")
    with columns[4]:
        stat(
            "90-day trend", f"{stats.trend_delta_90d:+.2f}★",
            delta=stats.trend_delta_90d, palette=palette,
            sub="vs the prior quarter",
        )

    sampling_caveat(stats)
    rule()

    # --- distribution + volume --------------------------------------------
    left, right = st.columns([0.42, 0.58])

    with left:
        section("Rating distribution", "Ordered scale, centred on the neutral 3★.")
        # Height matched to the two stacked charts opposite so the columns end
        # level instead of leaving a wedge of empty page.
        st.plotly_chart(
            rating_distribution(stats.score_distribution, palette, height=430),
            width="stretch", config=chart_config("ratings"),
        )

    with right:
        section(
            "Volume and rating over time",
            "Two measures, two charts — a shared axis would need two scales.",
        )
        st.plotly_chart(
            volume_trend(result.trends, palette),
            width="stretch", config=chart_config("volume"),
        )
        st.plotly_chart(
            rating_trend(result.trends, palette),
            width="stretch", config=chart_config("rating_trend"),
        )

    if stats.date_range:
        st.caption(
            f"Reviews span {stats.date_range[0]:%b %Y} to {stats.date_range[1]:%b %Y}."
        )

    rule()

    # --- needs overview ----------------------------------------------------
    section(
        "What the analysis found",
        "Every number here is computed from the data; the model only names things.",
    )

    # Two short charts side by side. The needs list goes full width underneath -
    # putting a five-item list beside two small plots leaves most of a column
    # empty, because Streamlit columns do not equalise height.
    columns = st.columns(2)

    with columns[0]:
        with st.container(border=True):
            kicker("Needs by confidence")
            st.plotly_chart(
                confidence_histogram(result.needs, palette),
                width="stretch", config=chart_config("bands"),
            )
            high = sum(1 for n in result.needs if n.confidence.band == "high")
            st.caption(
                f"{high} of {len(result.needs)} needs clear the high-confidence "
                f"threshold (0.70)."
            )

    with columns[1]:
        with st.container(border=True):
            kicker("Helpful-vote concentration")
            reviews = load_reviews(result.app.app_id, result.run.run_id)
            st.plotly_chart(
                helpful_votes_curve(reviews, palette),
                width="stretch", config=chart_config("helpful"),
            )
            votes = stats.helpful_votes
            st.caption(
                f"{votes.total:,} helpful votes across {votes.share_with_votes:.0%} "
                f"of reviews · median {votes.median:.0f}, p90 {votes.p90:.0f}, "
                f"max {votes.max:,}."
            )

    top = sorted(result.needs, key=lambda n: n.priority.value_score, reverse=True)[:5]
    if top:
        with st.container(border=True):
            kicker("Top needs by value")
            for need in top:
                row = st.columns([0.06, 0.62, 0.32])
                with row[0]:
                    st.markdown(
                        f'<div style="font-size:1.35rem;font-weight:650;'
                        f'color:{palette.ink_muted}">{need.priority.rank}</div>',
                        unsafe_allow_html=True,
                    )
                with row[1]:
                    st.markdown(need.statement)
                    st.caption(need.underlying_goal)
                with row[2]:
                    st.markdown(
                        f'<div style="text-align:right">'
                        f'{confidence_pill(need.confidence.band, need.confidence.total)}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"reach {need.priority.reach:.0%} · "
                        f"impact {need.priority.impact:.2f}★"
                    )
            # Same tab: this is navigation within one app, not a new context.
            if st.button("Open all needs", type="primary"):
                switch_to("views/details.py")

    rule()

    # --- value map ---------------------------------------------------------
    section(
        "Reach against impact",
        "Bubble size is computed confidence. Effort is deliberately not plotted — "
        "no engineering context exists here, and a fabricated estimate is what "
        "gets a tool distrusted in a planning meeting.",
    )
    st.plotly_chart(
        value_map(result.needs, palette),
        width="stretch", config=chart_config("value_map"),
    )

    with st.expander("Table view — every need, sorted by value"):
        st.dataframe(
            [
                {
                    "rank": n.priority.rank,
                    "need": n.statement,
                    "category": n.category.value,
                    "reach": n.priority.reach,
                    "impact": n.priority.impact,
                    "confidence": n.confidence.total,
                    "band": n.confidence.band,
                    "hiddenness": n.hiddenness,
                    "value": n.priority.value_score,
                }
                for n in sorted(result.needs, key=lambda n: n.priority.rank or 999)
            ],
            hide_index=True, width="stretch",
            column_config={
                "need": st.column_config.TextColumn(width="large"),
                "reach": st.column_config.NumberColumn(format="%.1f%%"),
                "impact": st.column_config.NumberColumn(format="%.2f★"),
                "confidence": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "hiddenness": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )


main()
