"""Plotly figures.

Form is chosen by the data's job, not by variety:

* **1-5 star distribution** is an ordered scale with a neutral middle, so it gets
  a *diverging* encoding centred on 3 stars - not five arbitrary hues.
* **Volume and rating over time** are two measures on different scales, so they
  are two stacked charts sharing an x-axis, never a dual-axis chart.
* **The cluster map** has 12-20 clusters. Twenty hues is not a palette, it is
  confetti, so it uses the *emphasis* form: one theme lit, the rest context grey.
* **Confidence components** compare magnitudes within one series, so one hue with
  direct labels.

Every figure carries a hover layer; single-series figures carry no legend box
because the title already names the series.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from app.theme import TEMPLATE_NAME, Palette, band_color

BAR_CORNER = 4  # rounded data-end, anchored to the baseline
MARK_GAP = 2  # surface-coloured gap between adjacent fills


def _base(palette: Palette, height: int, title: str | None = None) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template=TEMPLATE_NAME,
        height=height,
        showlegend=False,
        hovermode="closest",
    )
    if title:
        figure.update_layout(title=dict(text=title))
    return figure


def empty_figure(palette: Palette, message: str, height: int = 220) -> go.Figure:
    figure = _base(palette, height)
    figure.add_annotation(
        text=message, showarrow=False,
        font=dict(color=palette.ink_muted, size=13), xref="paper", yref="paper",
        x=0.5, y=0.5,
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


def rating_distribution(
    distribution: dict[int, int], palette: Palette, *, height: int = 240
) -> go.Figure:
    """Horizontal diverging bars over the ordered 1-5 star scale.

    Diverging rather than sequential because the scale has a real neutral: 1-2
    stars is a complaint, 3 is ambivalence, 4-5 is praise. Counts are direct-
    labelled so the chart is readable without reading the axis.
    """
    stars = [1, 2, 3, 4, 5]
    counts = [distribution.get(s, 0) for s in stars]
    total = sum(counts) or 1

    # Two poles plus a neutral grey midpoint - equal step count per arm, and
    # never a hue at the middle.
    colors = [
        palette.diverging_low,
        _blend(palette.diverging_low, 0.62),
        palette.diverging_mid,
        _blend(palette.diverging_high, 0.62),
        palette.diverging_high,
    ]

    figure = _base(palette, height)
    figure.add_bar(
        x=counts,
        y=[f"{s}★" for s in stars],
        orientation="h",
        marker=dict(
            color=colors,
            cornerradius=BAR_CORNER,
            line=dict(color=palette.surface, width=MARK_GAP),
        ),
        text=[f"{c:,}  ({c / total:.0%})" for c in counts],
        textposition="outside",
        textfont=dict(color=palette.ink_secondary, size=12),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,} reviews<extra></extra>",
    )
    figure.update_layout(bargap=0.28, margin=dict(l=8, r=64, t=8, b=8))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(
        showgrid=False, autorange="reversed",
        tickfont=dict(color=palette.ink_secondary, size=13),
    )
    return figure


def _blend(hex_color: str, weight: float) -> str:
    """Lighten toward white by `1 - weight`. Keeps the diverging arms one hue each."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r * weight + 255 * (1 - weight))
    g = int(g * weight + 255 * (1 - weight))
    b = int(b * weight + 255 * (1 - weight))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Trends - two charts, never a dual axis
# ---------------------------------------------------------------------------


def volume_trend(trends: Sequence, palette: Palette, *, height: int = 190) -> go.Figure:
    """Review volume per month. Single series, so no legend."""
    if not trends:
        return empty_figure(palette, "No dated reviews", height)

    figure = _base(palette, height)
    figure.add_bar(
        x=[t.period for t in trends],
        y=[t.n_reviews for t in trends],
        marker=dict(
            color=palette.series[0],
            cornerradius=BAR_CORNER,
            line=dict(color=palette.surface, width=MARK_GAP),
        ),
        hovertemplate="%{x|%b %Y}<br>%{y:,} reviews<extra></extra>",
    )
    figure.update_layout(bargap=0.25, margin=dict(l=8, r=8, t=8, b=8))
    figure.update_yaxes(title_text="reviews", rangemode="tozero")
    return figure


def rating_trend(trends: Sequence, palette: Palette, *, height: int = 190) -> go.Figure:
    """Rolling average rating. Separate figure from volume - different scale.

    Months with no reviews are broken rather than interpolated, so a gap in the
    data reads as a gap.
    """
    if not trends:
        return empty_figure(palette, "No dated reviews", height)

    periods = [t.period for t in trends]
    rolling = [t.rolling_avg if t.n_reviews > 0 else None for t in trends]
    monthly = [t.avg_score if t.n_reviews > 0 else None for t in trends]

    figure = _base(palette, height)
    # Monthly points sit behind as context; the rolling mean is the signal.
    figure.add_scatter(
        x=periods, y=monthly, mode="markers",
        marker=dict(size=5, color=palette.grid,
                    line=dict(color=palette.surface, width=1)),
        name="monthly mean",
        hovertemplate="%{x|%b %Y}<br>monthly mean %{y:.2f}★<extra></extra>",
    )
    figure.add_scatter(
        x=periods, y=rolling, mode="lines",
        line=dict(color=palette.series[0], width=2, shape="spline", smoothing=0.4),
        connectgaps=False,
        name="3-month rolling mean",
        hovertemplate="%{x|%b %Y}<br>rolling mean %{y:.2f}★<extra></extra>",
    )
    figure.update_layout(
        showlegend=True, hovermode="x unified", margin=dict(l=8, r=8, t=8, b=8)
    )
    figure.update_yaxes(title_text="rating", range=[1, 5], dtick=1)
    return figure


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def confidence_breakdown(
    components: dict[str, float], weights: dict[str, float], palette: Palette,
    *, height: int = 230,
) -> go.Figure:
    """One hue, magnitude comparison, direct-labelled.

    The weakest component is what a sceptical reader wants first, so it is
    marked with the critical status colour *and* named in its label - never
    colour alone.
    """
    names = list(components.keys())
    values = [components[n] for n in names]
    weakest = min(range(len(values)), key=lambda i: values[i]) if values else -1

    colors = [
        palette.critical if (i == weakest and values[i] < 0.35) else palette.series[0]
        for i in range(len(values))
    ]
    labels = [
        f"{n} · {int(weights.get(n, 0) * 100)}%" + ("  ⚠ weakest" if i == weakest and values[i] < 0.35 else "")
        for i, n in enumerate(names)
    ]

    figure = _base(palette, height)
    figure.add_bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=BAR_CORNER,
                    line=dict(color=palette.surface, width=MARK_GAP)),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(color=palette.ink_secondary, size=12),
        cliponaxis=False,
        hovertemplate="%{y}<br>score %{x:.2f}<extra></extra>",
    )
    figure.update_layout(bargap=0.3, margin=dict(l=8, r=48, t=8, b=8))
    figure.update_xaxes(visible=False, range=[0, 1.08])
    figure.update_yaxes(showgrid=False, autorange="reversed",
                        tickfont=dict(color=palette.ink_secondary, size=12))
    return figure


def confidence_histogram(
    needs: Sequence, palette: Palette, *, height: int = 190
) -> go.Figure:
    """How many needs land in each confidence band. Status colours + labels."""
    order = ["high", "medium", "low"]
    counts = {b: 0 for b in order}
    for need in needs:
        counts[need.confidence.band] = counts.get(need.confidence.band, 0) + 1

    figure = _base(palette, height)
    figure.add_bar(
        x=[b.capitalize() for b in order],
        y=[counts[b] for b in order],
        marker=dict(
            color=[band_color(palette, b) for b in order],
            cornerradius=BAR_CORNER,
            line=dict(color=palette.surface, width=MARK_GAP),
        ),
        text=[counts[b] for b in order],
        textposition="outside",
        textfont=dict(color=palette.ink_secondary, size=12),
        cliponaxis=False,
        hovertemplate="%{x} confidence<br>%{y} needs<extra></extra>",
    )
    figure.update_layout(bargap=0.42, margin=dict(l=8, r=8, t=8, b=8))
    figure.update_yaxes(visible=False, rangemode="tozero")
    figure.update_xaxes(tickfont=dict(color=palette.ink_secondary, size=12))
    return figure


# ---------------------------------------------------------------------------
# Cluster map - emphasis form
# ---------------------------------------------------------------------------


def cluster_scatter(
    projection: Sequence[dict], palette: Palette, *,
    highlight_cluster: int | None = None, cluster_labels: dict[int, str] | None = None,
    height: int = 420,
) -> go.Figure:
    """2D projection of review segments.

    With 12-20 clusters, colouring by cluster would need 20 hues - past every
    palette gate and unreadable under colour-vision deficiency. So the map uses
    emphasis: everything is context grey, and the selected theme lights up. That
    also answers the question a PM actually asks, which is "where does *this*
    theme sit?", not "what do all twenty look like at once?".
    """
    if not projection:
        return empty_figure(palette, "No projection for this run", height)

    cluster_labels = cluster_labels or {}
    figure = _base(palette, height)

    context = [p for p in projection if p.get("cluster") != highlight_cluster]
    if context:
        figure.add_scattergl(
            x=[p["x"] for p in context], y=[p["y"] for p in context],
            mode="markers",
            marker=dict(size=5, color=palette.grid, opacity=0.85),
            name="other segments",
            hoverinfo="skip",
        )

    if highlight_cluster is not None:
        lit = [p for p in projection if p.get("cluster") == highlight_cluster]
        if lit:
            figure.add_scattergl(
                x=[p["x"] for p in lit], y=[p["y"] for p in lit],
                mode="markers",
                marker=dict(
                    size=9, color=palette.series[0],
                    # 2px surface ring so overlapping marks stay countable.
                    line=dict(color=palette.surface, width=2),
                ),
                name=cluster_labels.get(highlight_cluster, f"cluster {highlight_cluster}"),
                customdata=[[p.get("score"), p.get("text", "")[:140]] for p in lit],
                hovertemplate=(
                    "%{customdata[0]}★<br>%{customdata[1]}<extra></extra>"
                ),
            )

    figure.update_layout(margin=dict(l=8, r=8, t=8, b=8), hovermode="closest")
    for axis in (figure.update_xaxes, figure.update_yaxes):
        axis(visible=False, showgrid=False, zeroline=False)
    return figure


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------


def value_map(
    needs: Sequence, palette: Palette, *, height: int = 380
) -> go.Figure:
    """Reach against impact, sized by confidence.

    One series, so no legend and no categorical colours. The top few are
    direct-labelled; labelling all twelve would be unreadable.
    """
    if not needs:
        return empty_figure(palette, "No needs extracted for this run", height)

    ranked = sorted(needs, key=lambda n: n.priority.value_score, reverse=True)
    top = {n.need_id for n in ranked[:4]}

    figure = _base(palette, height)
    figure.add_scatter(
        x=[n.priority.reach for n in ranked],
        y=[n.priority.impact for n in ranked],
        mode="markers+text",
        marker=dict(
            size=[12 + 26 * n.confidence.total for n in ranked],
            color=palette.series[0],
            opacity=0.82,
            line=dict(color=palette.surface, width=2),
        ),
        text=[
            (n.statement[:34] + "…") if n.need_id in top else ""
            for n in ranked
        ],
        textposition="top center",
        textfont=dict(color=palette.ink_secondary, size=11),
        customdata=[
            [n.statement[:110], n.confidence.total, n.confidence.band, n.priority.rank]
            for n in ranked
        ],
        hovertemplate=(
            "%{customdata[0]}<br>"
            "reach %{x:.1%} · impact %{y:.2f}★<br>"
            "confidence %{customdata[1]:.2f} (%{customdata[2]})"
            "<extra>rank %{customdata[3]}</extra>"
        ),
        cliponaxis=False,
    )
    figure.update_layout(margin=dict(l=8, r=8, t=24, b=8))
    figure.update_xaxes(title_text="reach — share of reviews touching the need",
                        tickformat=".0%", rangemode="tozero")
    figure.update_yaxes(title_text="impact — stars below the store rating",
                        rangemode="tozero")
    return figure


def helpful_votes_curve(
    reviews: Sequence, palette: Palette, *, height: int = 190
) -> go.Figure:
    """Cumulative share of helpful votes. Shows how concentrated endorsement is."""
    votes = sorted((max(0, r.helpful_count) for r in reviews), reverse=True)
    total = sum(votes)
    if not votes or total == 0:
        return empty_figure(palette, "No helpful votes recorded", height)

    cumulative, running = [], 0
    for vote in votes:
        running += vote
        cumulative.append(running / total)
    share_of_reviews = [(i + 1) / len(votes) for i in range(len(votes))]

    figure = _base(palette, height)
    figure.add_scatter(
        x=share_of_reviews, y=cumulative, mode="lines",
        line=dict(color=palette.series[0], width=2),
        fill="tozeroy",
        fillcolor=_blend(palette.series[0], 0.16),
        hovertemplate="top %{x:.0%} of reviews hold %{y:.0%} of votes<extra></extra>",
    )
    figure.update_layout(margin=dict(l=8, r=8, t=8, b=8), hovermode="x")
    figure.update_xaxes(title_text="share of reviews", tickformat=".0%")
    figure.update_yaxes(title_text="share of votes", tickformat=".0%", range=[0, 1])
    return figure
