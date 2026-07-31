"""Design tokens, page CSS and the shared Plotly template.

Colours are the validated palette, not taste. The categorical slots and the
surfaces they sit on were checked with the palette validator (lightness band,
chroma floor, CVD separation, normal-vision floor, contrast); changing a hex here
means re-running it.

Two rules this module exists to enforce:

* charts and Streamlit chrome share one surface, so a card never sits on a
  slightly different white than the plot inside it;
* every chart gets the same recessive grid, hairline axes and tabular tick
  figures without each page remembering to ask.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


@dataclass(frozen=True)
class Palette:
    """One mode's tokens. Light is the shipped default; dark is selected, not flipped."""

    surface: str
    plane: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    border: str

    # Categorical slots, in fixed order. Never cycle past the end.
    series: tuple[str, ...]

    # Diverging poles + neutral midpoint, for ordered scales like 1-5 stars.
    diverging_low: str
    diverging_mid: str
    diverging_high: str

    # Status. Fixed across modes, never themed, never reused as a series colour.
    #
    # This trio deliberately does NOT pass the categorical palette gate: good vs
    # critical measure ΔE 4.1 under deuteranopia, and warning sits below 3:1 on
    # the light surface. That gate scopes to palettes where hue carries identity,
    # and these are status colours - so the mitigation is the rule that they
    # always ship with an icon *and* a written label ("●●● high confidence ·
    # 0.73", axis ticks "High/Medium/Low"), never as a bare swatch. Do not
    # "fix" this by re-stepping the hues; fix it by never dropping the label.
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"

    @property
    def muted_mark(self) -> str:
        """De-emphasis fill for the emphasis form (one series lit, rest context)."""
        return self.grid


LIGHT = Palette(
    surface="#fcfcfb",
    plane="#f9f9f7",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
    diverging_low="#d03b3b",
    diverging_mid="#f0efec",
    diverging_high="#2a78d6",
)

DARK = Palette(
    surface="#1a1a19",
    plane="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    border="rgba(255,255,255,0.10)",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"),
    diverging_low="#d03b3b",
    diverging_mid="#383835",
    diverging_high="#3987e5",
)


def active_palette() -> Palette:
    """Resolve the palette to whatever Streamlit is *actually* rendering.

    `st.context.theme` reports the browser's colour-scheme preference, which is
    not the same thing: when `theme.base` is pinned in `config.toml`, Streamlit
    ignores the browser and the context value disagrees with the rendered chrome.
    Trusting it there paints dark cards onto a light page. The pinned option wins;
    the browser preference is consulted only when nothing is pinned.
    """
    try:
        pinned = st.get_option("theme.base")
    except Exception:
        pinned = None

    if pinned == "dark":
        return DARK
    if pinned == "light":
        return LIGHT

    try:
        if getattr(st.context, "theme", None) and st.context.theme.type == "dark":
            return DARK
    except Exception:
        pass
    return LIGHT


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

TEMPLATE_NAME = "aipm"


def register_plotly_template(palette: Palette) -> str:
    """Install the shared template. Idempotent - safe to call on every rerun."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_STACK, size=13, color=palette.ink_secondary),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(palette.series),
        margin=dict(l=8, r=8, t=32, b=8),
        hoverlabel=dict(
            bgcolor=palette.surface,
            bordercolor=palette.border,
            font=dict(family=FONT_STACK, size=12, color=palette.ink),
        ),
        # Recessive chrome: a hairline horizontal grid only, no vertical rules,
        # no plot border box.
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=palette.axis,
            linewidth=1,
            ticks="outside",
            tickcolor=palette.axis,
            ticklen=4,
            tickfont=dict(color=palette.ink_muted, size=12),
            title=dict(font=dict(color=palette.ink_muted, size=12)),
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=palette.grid,
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="",
            tickfont=dict(color=palette.ink_muted, size=12),
            title=dict(font=dict(color=palette.ink_muted, size=12)),
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=palette.ink_secondary, size=12),
            title=dict(text=""),
        ),
        title=dict(
            font=dict(color=palette.ink, size=15),
            x=0,
            xanchor="left",
            y=0.97,
            yanchor="top",
        ),
    )
    pio.templates[TEMPLATE_NAME] = template
    return TEMPLATE_NAME


def chart_config(filename: str = "chart") -> dict:
    """Plotly modebar config: keep export, drop the toys."""
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d", "select2d", "autoScale2d", "zoomIn2d", "zoomOut2d",
        ],
        "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2},
        "responsive": True,
    }


# ---------------------------------------------------------------------------
# Page CSS
# ---------------------------------------------------------------------------


def _css(palette: Palette) -> str:
    return f"""
<style>
  :root {{
    --surface: {palette.surface};
    --plane: {palette.plane};
    --ink: {palette.ink};
    --ink-2: {palette.ink_secondary};
    --ink-muted: {palette.ink_muted};
    --border: {palette.border};
    --grid: {palette.grid};
    --accent: {palette.series[0]};
    --good: {palette.good};
    --warning: {palette.warning};
    --critical: {palette.critical};
    --radius: 12px;
  }}

  /* Set the face on ancestors only and let inheritance carry it.
     Do NOT use an attribute selector like [class*="st-"]: Streamlit applies its
     icon font with a single-class rule (.st-emotion-cache-xxxx), which has the
     same 0-1-0 specificity as an attribute selector - and because this block is
     injected at runtime it lands *after* Streamlit's stylesheet and wins. That
     overrides "Material Symbols Rounded" on the icon spans, so every icon
     renders as its literal ligature name ("keyboard_double_arrow_left",
     "arrow_right", "upload"). Targeting an ancestor keeps our font everywhere
     text is inherited while leaving elements that set their own face alone. */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
    font-family: {FONT_STACK};
  }}

  .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1400px; }}

  /* Tighten Streamlit's default vertical rhythm - the stock spacing makes a
     dense analytics page feel unfinished rather than airy. */
  [data-testid="stVerticalBlock"] {{ gap: 0.85rem; }}

  h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.015em; font-weight: 650; }}
  h1 {{ font-size: 1.9rem; margin-bottom: 0.15rem; }}
  h2 {{ font-size: 1.25rem; margin-top: 0.4rem; }}
  h3 {{ font-size: 1.02rem; }}

  /* ---- surfaces ---- */
  .aipm-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.05rem 1.2rem;
    height: 100%;
  }}
  .aipm-card--flush {{ padding: 0.9rem 1rem; }}

  /* ---- stat tile ---- */
  .aipm-stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem 1rem 0.9rem;
    height: 100%;
  }}
  .aipm-stat__label {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--ink-muted); margin-bottom: 0.3rem;
  }}
  .aipm-stat__value {{
    font-size: 1.75rem; font-weight: 660; color: var(--ink);
    line-height: 1.12; letter-spacing: -0.02em;
  }}
  .aipm-stat__sub {{ font-size: 0.79rem; color: var(--ink-2); margin-top: 0.22rem; }}
  .aipm-stat__delta {{ font-size: 0.79rem; font-weight: 600; margin-top: 0.22rem; }}

  /* ---- hero figure ---- */
  .aipm-hero {{ font-size: 3rem; font-weight: 680; letter-spacing: -0.03em;
                color: var(--ink); line-height: 1; }}

  /* ---- pills ---- */
  .aipm-pill {{
    display: inline-flex; align-items: center; gap: 0.32rem;
    padding: 0.16rem 0.55rem; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600; line-height: 1.5;
    border: 1px solid var(--border); color: var(--ink-2);
    background: color-mix(in srgb, var(--surface) 88%, var(--ink) 4%);
    white-space: nowrap;
  }}
  .aipm-pill--good {{ color: var(--good); border-color: color-mix(in srgb, var(--good) 35%, transparent); }}
  .aipm-pill--warn {{ color: color-mix(in srgb, var(--warning) 75%, var(--ink) 25%);
                      border-color: color-mix(in srgb, var(--warning) 45%, transparent); }}
  .aipm-pill--crit {{ color: var(--critical); border-color: color-mix(in srgb, var(--critical) 35%, transparent); }}
  .aipm-pill--accent {{ color: var(--accent); border-color: color-mix(in srgb, var(--accent) 35%, transparent); }}

  /* ---- confidence meter ---- */
  .aipm-meter {{ display: flex; height: 9px; border-radius: 999px; overflow: hidden;
                 background: var(--grid); gap: 2px; }}
  .aipm-meter__seg {{ height: 100%; }}
  .aipm-meter__scale {{ display: flex; justify-content: space-between;
                        font-size: 0.7rem; color: var(--ink-muted); margin-top: 0.3rem; }}

  /* ---- quote ---- */
  .aipm-quote {{
    border-left: 3px solid var(--accent);
    padding: 0.5rem 0 0.5rem 0.85rem; margin: 0.45rem 0;
    color: var(--ink-2); font-size: 0.9rem; line-height: 1.5;
  }}
  .aipm-quote__meta {{ font-size: 0.74rem; color: var(--ink-muted); margin-top: 0.3rem;
                       font-variant-numeric: tabular-nums; }}

  /* ---- misc ---- */
  .aipm-muted {{ color: var(--ink-muted); font-size: 0.84rem; }}

  /* Clamp catalogue blurbs to a fixed box so every card in a row is the same
     height and the action buttons line up. min-height alone does not do it -
     a three-line description still pushes its button down. */
  .aipm-clamp {{
    color: var(--ink-muted); font-size: 0.84rem; line-height: 1.45;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; height: 2.9em;
  }}
  .aipm-rule {{ height: 1px; background: var(--border); border: 0; margin: 1.1rem 0; }}
  .aipm-kicker {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
                  text-transform: uppercase; color: var(--ink-muted); }}

  [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
  div[data-testid="stDataFrame"] {{ border-radius: var(--radius); }}

  /* Streamlit renders each chart in an iframe-less div; kill the default
     bottom margin so cards wrap plots tightly. */
  [data-testid="stPlotlyChart"] {{ margin-bottom: -0.4rem; }}

  section[data-testid="stSidebar"] {{ border-right: 1px solid var(--border); }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

  /* ---- custom sidebar nav ----
     Replaces Streamlit's flat page tree. Nav items are buttons styled to read as
     links; the active one gets a filled background and a left rule so the
     current view is obvious without relying on colour alone. */
  .aipm-nav-group {{
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--ink-muted);
    margin: 0.9rem 0 0.3rem;
  }}
  .aipm-nav-app {{
    font-weight: 650; font-size: 0.95rem; color: var(--ink);
    margin: 0.9rem 0 0.35rem; line-height: 1.3;
  }}
  /* Nested items sit under the app name to show they belong to it. */
  .aipm-nav-nested {{ padding-left: 0.55rem; border-left: 2px solid var(--grid); }}
</style>
"""


def configure_app(layout: str = "wide") -> Palette:
    """Everything that must happen exactly once per script run.

    `st.set_page_config` may only be called once, and under `st.navigation` the
    entrypoint and the selected view run within the *same* script run - so this
    belongs to the entrypoint alone. Views call `active_palette()` instead.

    Per-view browser titles come from `st.Page(title=..., icon=...)`.
    """
    st.set_page_config(
        page_title="AI PM Assistant",
        page_icon="📱",
        layout=layout,
        initial_sidebar_state="expanded",
    )
    palette = active_palette()
    register_plotly_template(palette)
    st.markdown(_css(palette), unsafe_allow_html=True)
    return palette


# ---------------------------------------------------------------------------
# Small shared formatters
# ---------------------------------------------------------------------------


def band_class(band: str) -> str:
    return {"high": "aipm-pill--good", "medium": "aipm-pill--warn"}.get(
        band, "aipm-pill--crit"
    )


def band_icon(band: str) -> str:
    """Status never travels as colour alone."""
    return {"high": "●●●", "medium": "●●○", "low": "●○○"}.get(band, "○○○")


def band_color(palette: Palette, band: str) -> str:
    return {"high": palette.good, "medium": palette.warning}.get(band, palette.critical)


def compact_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M".replace(".0M", "M")
    if value >= 1_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return f"{value:,.0f}"
