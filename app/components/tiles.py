"""Stat tiles, pills and section headers.

A single headline number is a stat tile, not a one-bar bar chart. These render as
plain HTML against the theme tokens so they line up with the chart surfaces
exactly - `st.metric` sits on a different background and breaks the grid.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from app.theme import Palette, band_icon, compact_number


def stat(
    label: str,
    value: str,
    *,
    sub: str | None = None,
    delta: float | None = None,
    delta_suffix: str = "",
    palette: Palette | None = None,
    higher_is_better: bool = True,
) -> None:
    """One KPI. `delta` is rendered with an arrow *and* a sign, never colour alone."""
    parts = [
        '<div class="aipm-stat">',
        f'<div class="aipm-stat__label">{escape(label)}</div>',
        f'<div class="aipm-stat__value">{escape(value)}</div>',
    ]
    if delta is not None and palette is not None:
        improving = (delta >= 0) if higher_is_better else (delta <= 0)
        color = palette.good if improving else palette.critical
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        parts.append(
            f'<div class="aipm-stat__delta" style="color:{color}">'
            f"{arrow} {delta:+.2f}{escape(delta_suffix)}</div>"
        )
    if sub:
        parts.append(f'<div class="aipm-stat__sub">{escape(sub)}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def pill(text: str, variant: str = "") -> str:
    """Inline badge. Returns HTML so callers can compose a row of them."""
    css = f"aipm-pill {variant}".strip()
    return f'<span class="{css}">{escape(text)}</span>'


def pill_row(pills: list[str]) -> None:
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:0.35rem;margin:0.15rem 0 0.5rem">'
        + "".join(pills)
        + "</div>",
        unsafe_allow_html=True,
    )


def confidence_pill(band: str, total: float) -> str:
    variant = {"high": "aipm-pill--good", "medium": "aipm-pill--warn"}.get(
        band, "aipm-pill--crit"
    )
    # Icon + label, so the band never depends on hue alone.
    return pill(f"{band_icon(band)} {band} confidence · {total:.2f}", variant)


def section(title: str, description: str | None = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.markdown(
            f'<div class="aipm-muted" style="margin-top:-0.5rem;margin-bottom:0.6rem">'
            f"{escape(description)}</div>",
            unsafe_allow_html=True,
        )


def kicker(text: str) -> None:
    st.markdown(f'<div class="aipm-kicker">{escape(text)}</div>', unsafe_allow_html=True)


def rule() -> None:
    st.markdown('<hr class="aipm-rule"/>', unsafe_allow_html=True)


def hero(value: str, caption: str) -> None:
    st.markdown(
        f'<div class="aipm-hero">{escape(value)}</div>'
        f'<div class="aipm-muted" style="margin-top:0.2rem">{escape(caption)}</div>',
        unsafe_allow_html=True,
    )


def sampling_caveat(stats) -> None:
    """Disclose the quota-capped scrape wherever the sample mean is on screen.

    Without this the dashboard shows a 2.6★ mean beside a 4.4★ store rating and
    looks broken. It is not broken - the corpus is a capped scrape - but the UI
    has to say so.
    """
    if not stats.sample_is_quota_capped:
        return
    st.caption(
        f"ℹ️ The review sample is quota-capped by the scraper "
        f"({stats.n_star_levels} of 5 star levels, near-equal counts), so the "
        f"**sample mean is not this app's rating**. The store rating "
        f"({stats.store_score}) is the real one; impact is measured against it."
    )


def app_summary_line(row: dict) -> str:
    bits = [f"{compact_number(row['n_reviews'])} reviews"]
    if row.get("store_score"):
        bits.append(f"{row['store_score']}★ on store")
    if row.get("n_needs"):
        bits.append(f"{row['n_needs']} needs")
    return " · ".join(bits)
