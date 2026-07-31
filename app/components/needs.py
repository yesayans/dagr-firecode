"""Need cards, the confidence meter and its explanation.

The confidence meter is the product's central claim, so it is rendered as a
*stacked* meter of the six weighted contributions rather than a single bar: a
reader can see which component earned the score. The sentence underneath is the
one computed in Python, so bar and sentence can never disagree.

The model's qualitative rationale sits beside it, clearly attributed, because it
is reasoning rather than measurement.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from app.components.charts import confidence_breakdown
from app.components.tiles import confidence_pill, pill, pill_row
from app.theme import Palette, band_color, chart_config

COMPONENT_ORDER = (
    "support", "cohesion", "separation", "temporal", "diversity", "grounding",
)

COMPONENT_HELP = {
    "support": "How many review segments back this need, log-scaled against the "
               "largest theme in the run.",
    "cohesion": "Mean similarity between the supporting segments. Low means the "
                "theme is loose.",
    "separation": "Distance to the nearest other theme. Low means it overlaps a "
                  "neighbour.",
    "temporal": "Share of active months in which the theme appears. Low means a "
                "one-off spike.",
    "diversity": "1 − the share of near-duplicate reviews. Low means the evidence "
                 "is one text repeated.",
    "grounding": "Share of the model's citations that survived validation.",
}


def confidence_meter(need, weights: dict[str, float], palette: Palette) -> None:
    """Stacked contributions, scaled so the filled width equals the total."""
    breakdown = need.confidence
    contributions = [
        (name, getattr(breakdown, name, 0.0) * weights.get(name, 0.0))
        for name in COMPONENT_ORDER
    ]
    total = sum(c for _, c in contributions) or 1.0

    segments = []
    for index, (name, contribution) in enumerate(contributions):
        width = contribution / total * breakdown.total * 100
        if width <= 0:
            continue
        # One hue, stepped by position, so the meter reads as one quantity split
        # into parts rather than six competing series.
        shade = 0.45 + 0.55 * (1 - index / max(1, len(contributions) - 1))
        segments.append(
            f'<div class="aipm-meter__seg" title="{escape(name)}: '
            f'{getattr(breakdown, name, 0.0):.2f} × weight {weights.get(name, 0):.2f}" '
            f'style="width:{width:.2f}%;background:{_tint(palette.series[0], shade)}"></div>'
        )

    st.markdown(
        '<div class="aipm-meter">' + "".join(segments) + "</div>"
        '<div class="aipm-meter__scale"><span>0.00</span>'
        f"<span>{breakdown.total:.2f} of 1.00</span></div>",
        unsafe_allow_html=True,
    )


def _tint(hex_color: str, weight: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r * weight + 255 * (1 - weight))
    g = int(g * weight + 255 * (1 - weight))
    b = int(b * weight + 255 * (1 - weight))
    return f"#{r:02x}{g:02x}{b:02x}"


def confidence_detail(need, weights: dict[str, float], palette: Palette) -> None:
    """The full breakdown: chart, computed sentence, model rationale, table view."""
    components = {name: getattr(need.confidence, name, 0.0) for name in COMPONENT_ORDER}

    st.plotly_chart(
        confidence_breakdown(components, weights, palette),
        width="stretch",
        config=chart_config("confidence"),
        key=f"conf_{need.need_id}",
    )

    st.markdown(
        f'<div class="aipm-muted"><strong style="color:{band_color(palette, need.confidence.band)}">'
        f"Computed:</strong> {escape(need.confidence.explanation)}</div>",
        unsafe_allow_html=True,
    )
    if need.confidence.llm_rationale:
        st.markdown(
            f'<div class="aipm-muted" style="margin-top:0.4rem">'
            f"<strong>Model's reading:</strong> {escape(need.confidence.llm_rationale)}</div>",
            unsafe_allow_html=True,
        )

    # Table view - the relief path for the contrast warning and for anyone who
    # would rather read numbers than a chart.
    with st.expander("Component table"):
        st.dataframe(
            [
                {
                    "component": name,
                    "score": round(components[name], 3),
                    "weight": weights.get(name, 0.0),
                    "contribution": round(components[name] * weights.get(name, 0.0), 4),
                    "what it measures": COMPONENT_HELP[name],
                }
                for name in COMPONENT_ORDER
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "score": st.column_config.ProgressColumn(
                    "score", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "what it measures": st.column_config.TextColumn(width="large"),
            },
        )


def need_card(need, weights: dict[str, float], palette: Palette, *, rank: int | None = None) -> None:
    """One need, with statement, goal, workaround, meter and evidence."""
    with st.container(border=True):
        header = f"#{rank} · " if rank else ""
        st.markdown(f"#### {header}{escape(need.statement)}")

        badges = [
            confidence_pill(need.confidence.band, need.confidence.total),
            pill(f"hiddenness {need.hiddenness:.2f}", "aipm-pill--accent"),
            pill(f"reach {need.priority.reach:.1%}"),
            pill(f"impact {need.priority.impact:.2f}★"),
            pill(need.category.value.replace("_", " ")),
        ]
        pill_row(badges)

        st.markdown(
            f'<div class="aipm-kicker">Job to be done</div>'
            f'<div style="margin-bottom:0.5rem">{escape(need.underlying_goal)}</div>',
            unsafe_allow_html=True,
        )

        if need.workarounds and need.workarounds[0].strip():
            # The workaround is the strongest hidden-need signal in the corpus,
            # so it gets its own emphasis rather than being buried in a list.
            st.markdown(
                f'<div class="aipm-kicker">Workaround users describe</div>'
                f'<div class="aipm-quote">{escape(need.workarounds[0])}</div>',
                unsafe_allow_html=True,
            )

        if need.surface_complaints and need.surface_complaints[0].strip():
            st.markdown(
                f'<div class="aipm-kicker">Surface complaint</div>'
                f'<div class="aipm-muted" style="margin-bottom:0.6rem">'
                f"{escape(need.surface_complaints[0])}</div>",
                unsafe_allow_html=True,
            )

        confidence_meter(need, weights, palette)

        left, right = st.columns(2)
        with left:
            with st.expander("Confidence breakdown"):
                confidence_detail(need, weights, palette)
        with right:
            n_validated = sum(1 for e in need.evidence if e.validated)
            with st.expander(f"Evidence · {n_validated} verified of {len(need.evidence)}"):
                evidence_list(need.evidence, palette)


def evidence_list(evidence, palette: Palette) -> None:
    """Quotes with provenance. Verified and contextual are labelled, not just tinted."""
    if not evidence:
        st.caption("No evidence survived validation for this need.")
        return
    for item in evidence:
        mark = "✓ verified citation" if item.validated else "◦ supporting context"
        color = palette.good if item.validated else palette.ink_muted
        stars = f"{item.review_score}★" if item.review_score else "—"
        date = item.review_date.isoformat() if item.review_date else "undated"
        st.markdown(
            f'<div class="aipm-quote" style="border-left-color:{color}">'
            f"{escape(item.quote)}"
            f'<div class="aipm-quote__meta">'
            f"<span style='color:{color}'>{mark}</span> · {stars} · {date} · "
            f"{item.helpful_count} helpful · relevance {item.relevance:.2f} · "
            f"<code>{escape(item.review_id)}</code>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
