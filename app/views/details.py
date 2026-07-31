"""Application Details - needs, themes and the evidence behind them.

This is the page that has to survive scepticism, so every claim is one click from
the reviews that produced it. Three tabs:

* **Hidden needs** - the ranked cards with confidence breakdowns.
* **Themes** - the clusters, their keywords, and where they sit in the map.
* **Evidence** - every analysed review, filterable and exportable.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.charts import cluster_scatter  # noqa: E402
from app.components.needs import need_card  # noqa: E402
from app.components.tiles import (  # noqa: E402
    kicker, pill, pill_row, rule, sampling_caveat, section, stat,
)
from app.state import (  # noqa: E402
    get_settings, load_reviews, require_selection,
)
from app.theme import active_palette, chart_config  # noqa: E402

palette = active_palette()


def render_sidebar(result) -> None:
    with st.sidebar:
        st.caption(" · ".join(result.app.categories) or "Uncategorised")
        st.markdown("**Reading this page**")
        st.caption(
            "A need is *hidden* when users describe symptoms and workarounds "
            "rather than asking for a feature. Sorting by hiddenness × confidence "
            "surfaces what a PM would not get from reading reviews top to bottom."
        )


# ---------------------------------------------------------------------------
# Tab 1 - needs
# ---------------------------------------------------------------------------


def tab_needs(result, weights: dict[str, float]) -> None:
    needs = result.needs
    if not needs:
        st.info(
            "No needs were extracted for this run. The dashboard statistics are "
            "still complete.",
            icon="ℹ️",
        )
        return

    controls = st.columns([0.34, 0.30, 0.36])
    with controls[0]:
        order = st.selectbox(
            "Sort by",
            ["Hiddenness × confidence", "Value", "Confidence", "Reach"],
            help="Hiddenness × confidence is the ordering the product argues for: "
                 "well-evidenced needs that are not already obvious.",
        )
    with controls[1]:
        categories = sorted({n.category.value for n in needs})
        chosen = st.multiselect("Category", categories, placeholder="All categories")
    with controls[2]:
        min_confidence = st.slider(
            "Minimum confidence", 0.0, 1.0, 0.0, 0.05,
            help="Low-confidence needs are shown as hypotheses, not hidden.",
        )

    sort_keys = {
        "Hiddenness × confidence": lambda n: -n.insight_score,
        "Value": lambda n: -n.priority.value_score,
        "Confidence": lambda n: -n.confidence.total,
        "Reach": lambda n: -n.priority.reach,
    }
    visible = [
        n for n in needs
        if (not chosen or n.category.value in chosen)
        and n.confidence.total >= min_confidence
    ]
    visible.sort(key=sort_keys[order])

    strong = [n for n in visible if n.confidence.band != "low"]
    weak = [n for n in visible if n.confidence.band == "low"]

    st.caption(
        f"Showing {len(visible)} of {len(needs)} needs · "
        f"{len(strong)} evidenced, {len(weak)} hypotheses."
    )

    for index, need in enumerate(strong, start=1):
        need_card(need, weights, palette, rank=index)

    if weak:
        rule()
        with st.expander(f"Hypotheses — weak evidence ({len(weak)})"):
            st.caption(
                "These cleared extraction but not the evidence bar. They are shown "
                "rather than dropped, labelled for what they are."
            )
            for need in weak:
                need_card(need, weights, palette)


# ---------------------------------------------------------------------------
# Tab 2 - themes
# ---------------------------------------------------------------------------


def tab_themes(result) -> None:
    clusters = sorted(result.clusters, key=lambda c: c.size, reverse=True)
    if not clusters:
        st.info("No themes were formed for this run.", icon="ℹ️")
        return

    # The projection stores integer cluster labels; the Cluster objects carry
    # hashed ids. Recover the mapping by descending size, which is the order
    # `_characterise` used.
    projection = result.projection
    label_counts: dict[int, int] = {}
    for point in projection:
        label = point.get("cluster")
        if label is not None and label != -1:
            label_counts[label] = label_counts.get(label, 0) + 1
    ordered_labels = sorted(label_counts, key=lambda k: label_counts[k], reverse=True)
    cluster_by_label = {
        label: clusters[i] for i, label in enumerate(ordered_labels) if i < len(clusters)
    }

    left, right = st.columns([0.44, 0.56])

    with left:
        section("Themes", "Ordered by how many review segments they hold.")
        options = list(cluster_by_label.keys())
        if not options:
            st.caption("No projected clusters to explore.")
            selected_label = None
        else:
            selected_label = st.selectbox(
                "Highlight a theme",
                options,
                format_func=lambda label: (
                    f"{cluster_by_label[label].label or 'Unlabelled'} "
                    f"({cluster_by_label[label].size:,})"
                ),
            )

        if selected_label is not None:
            cluster = cluster_by_label[selected_label]
            with st.container(border=True):
                st.markdown(f"**{escape(cluster.label or 'Unlabelled theme')}**")
                if cluster.summary:
                    st.markdown(
                        f'<div class="aipm-muted">{escape(cluster.summary)}</div>',
                        unsafe_allow_html=True,
                    )
                pill_row([
                    pill(f"{cluster.size:,} segments"),
                    pill(f"cohesion {cluster.cohesion:.2f}"),
                    pill(f"separation {cluster.separation:.2f}"),
                ])
                if cluster.keywords:
                    kicker("Distinguishing keywords (c-TF-IDF, computed)")
                    pill_row([pill(k) for k in cluster.keywords[:10]])

    with right:
        section(
            "Segment map",
            "One point per review segment. The selected theme is lit; the rest is "
            "context — twenty hues would be confetti, not a palette.",
        )
        st.plotly_chart(
            cluster_scatter(
                projection, palette,
                highlight_cluster=selected_label,
                cluster_labels={
                    label: (c.label or f"theme {label}")
                    for label, c in cluster_by_label.items()
                },
            ),
            width="stretch", config=chart_config("cluster_map"),
        )

    rule()
    with st.expander("All themes — table view"):
        st.dataframe(
            [
                {
                    "theme": c.label or "(unlabelled)",
                    "segments": c.size,
                    "cohesion": c.cohesion,
                    "separation": c.separation,
                    "keywords": ", ".join(c.keywords[:6]),
                }
                for c in clusters
            ],
            hide_index=True, width="stretch",
            column_config={
                "theme": st.column_config.TextColumn(width="medium"),
                "keywords": st.column_config.TextColumn(width="large"),
                "cohesion": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "separation": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )


# ---------------------------------------------------------------------------
# Tab 3 - evidence
# ---------------------------------------------------------------------------


def tab_evidence(result) -> None:
    reviews = load_reviews(result.app.app_id, result.run.run_id)
    if not reviews:
        st.info("No reviews were stored for this run.", icon="ℹ️")
        return

    cited: dict[str, list[str]] = {}
    for need in result.needs:
        for item in need.evidence:
            if item.validated:
                cited.setdefault(item.review_id, []).append(need.statement)

    section(
        "Every analysed review",
        "This is the drill-down that turns scepticism into trust: the raw text "
        "behind every claim, filterable and exportable.",
    )

    controls = st.columns([0.34, 0.22, 0.22, 0.22])
    with controls[0]:
        query = st.text_input("Search text", placeholder="e.g. refund, face id…")
    with controls[1]:
        stars = st.multiselect("Rating", [1, 2, 3, 4, 5], placeholder="Any")
    with controls[2]:
        min_helpful = st.number_input("Min helpful votes", min_value=0, value=0, step=1)
    with controls[3]:
        only_cited = st.toggle("Cited as evidence only", value=False)

    rows = []
    for review in reviews:
        if query and query.lower() not in review.text.lower():
            continue
        if stars and review.score not in stars:
            continue
        if review.helpful_count < min_helpful:
            continue
        is_cited = review.review_id in cited
        if only_cited and not is_cited:
            continue
        rows.append(
            {
                "review_id": review.review_id,
                "date": review.review_date,
                "rating": review.score,
                "helpful": review.helpful_count,
                "cited for": " | ".join(cited.get(review.review_id, []))[:120],
                "quality": review.quality_weight,
                "duplicate": review.is_duplicate,
                "text": review.text,
            }
        )

    st.caption(f"{len(rows):,} of {len(reviews):,} reviews match.")
    if not rows:
        st.info("No reviews match those filters.", icon="🔍")
        return

    frame = pd.DataFrame(rows)
    st.dataframe(
        frame, hide_index=True, width="stretch", height=520,
        column_config={
            "review_id": st.column_config.TextColumn("id", width="small"),
            "date": st.column_config.DateColumn(width="small"),
            "rating": st.column_config.NumberColumn(format="%d★", width="small"),
            "helpful": st.column_config.NumberColumn(width="small"),
            "quality": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f", width="small"
            ),
            "cited for": st.column_config.TextColumn(width="medium"),
            "text": st.column_config.TextColumn(width="large"),
        },
    )

    st.download_button(
        "Download as CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{result.app.app_id}_evidence.csv",
        mime="text/csv",
    )


def main() -> None:
    result = require_selection("Application details", route="Application_Details")
    if result is None:
        return

    render_sidebar(result)
    weights = get_settings().confidence_weights()

    st.title(result.app.name)
    st.markdown(
        f'<div class="aipm-muted" style="margin-top:-0.35rem">'
        f"{result.run.n_reviews:,} reviews · {result.run.n_units:,} segments · "
        f"{result.run.n_clusters} themes · {len(result.needs)} needs</div>",
        unsafe_allow_html=True,
    )
    sampling_caveat(result.stats)

    columns = st.columns(4)
    with columns[0]:
        stat("Needs", str(len(result.needs)), sub="cited and scored")
    with columns[1]:
        high = sum(1 for n in result.needs if n.confidence.band == "high")
        stat("High confidence", str(high), sub="of the extracted needs")
    with columns[2]:
        stat("Citations dropped", str(result.run.citations_dropped),
             sub="rejected by the guard")
    with columns[3]:
        stat("Themes", str(result.run.n_clusters),
             sub=f"{result.run.noise_ratio:.0%} unclustered")

    rule()

    needs_tab, themes_tab, evidence_tab = st.tabs(
        ["Hidden needs", "Themes", "Evidence"]
    )
    with needs_tab:
        tab_needs(result, weights)
    with themes_tab:
        tab_themes(result)
    with evidence_tab:
        tab_evidence(result)


main()
