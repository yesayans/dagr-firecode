"""Home - the application catalogue.

Entry point. Everything here reads from precomputed runs; no analysis is
triggered by rendering this page, which is the whole reason the precompute
script exists.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.tiles import (  # noqa: E402
    kicker, pill, pill_row, rule, section, stat,
)
from app.state import (  # noqa: E402
    Selection, get_llm_client, get_selection, load_catalog, load_manifest,
    switch_to,
)
from app.theme import active_palette, compact_number  # noqa: E402

palette = active_palette()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(manifest) -> None:
    with st.sidebar:
        st.caption("Hidden user needs from app reviews — with evidence.")

        if manifest is None:
            st.warning("No precomputed data found.", icon="⚠️")
            st.code("python scripts/precompute_demo.py", language="bash")
            return

        st.markdown("**Precomputed run**")
        st.caption(f"{manifest.n_apps} apps · {manifest.total_needs} needs")
        st.caption(f"Generated {manifest.created_at:%d %b %Y, %H:%M} UTC")

        rule()
        st.markdown("**Pipeline**")
        st.caption(f"Embeddings — `{manifest.embed_model.split('/')[-1]}` ({manifest.embed_dim}d)")
        if manifest.llm_enabled:
            st.caption(f"Reasoning — `{manifest.llm_model}`")
        else:
            st.caption("Reasoning — heuristic (no LLM configured)")

        client = get_llm_client()
        if client.available:
            st.caption("✅ Chat is available")
        else:
            st.caption("✕ Chat needs an LLM endpoint")

        if manifest.warnings:
            with st.expander(f"Run warnings ({len(manifest.warnings)})"):
                for warning in manifest.warnings[:20]:
                    st.caption(f"• {warning}")


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def app_card(row: dict, *, is_selected: bool) -> None:
    with st.container(border=True):
        top = st.columns([0.72, 0.28])
        with top[0]:
            # Two-line clamp: a long name would otherwise push this card's
            # button below its neighbours'.
            st.markdown(
                f'<div style="font-weight:650;font-size:1.02rem;line-height:1.3;'
                f'height:2.6em;overflow:hidden;display:-webkit-box;'
                f'-webkit-line-clamp:2;-webkit-box-orient:vertical">'
                f"{escape(row['name'])}</div>"
                f'<div class="aipm-kicker">{escape(row["category"])}</div>',
                unsafe_allow_html=True,
            )
        with top[1]:
            if row.get("store_score"):
                st.markdown(
                    f'<div style="text-align:right"><span class="aipm-hero" '
                    f'style="font-size:1.5rem">{row["store_score"]}</span>'
                    f'<span class="aipm-muted">★</span></div>',
                    unsafe_allow_html=True,
                )

        pill_row([
            pill(f"{compact_number(row['n_reviews'])} reviews"),
            pill(f"{row['n_clusters']} themes"),
            pill(f"{row['n_needs']} needs", "aipm-pill--accent"),
        ])

        # Truncated in Python, not by CSS: a `-webkit-line-clamp` here depends on
        # our stylesheet surviving Streamlit's markdown sanitiser, and when it
        # does not the card grows to a full store listing.
        blurb = " ".join((row.get("description") or "").split())
        if len(blurb) > 118:
            blurb = blurb[:118].rsplit(" ", 1)[0] + "…"
        st.markdown(
            f'<div class="aipm-clamp">{escape(blurb) or "&nbsp;"}</div>',
            unsafe_allow_html=True,
        )

        if row["status"] != "complete":
            st.error(row.get("error") or "This run failed.", icon="🚫")
            return

        # Same tab, but the destination URL still names the app because
        # `switch_to` passes it as query params. That keeps refresh, bookmarks
        # and sharing working without opening a tab per click.
        if st.button(
            "Selected \u2713 \u00b7 reopen" if is_selected else "Open analysis",
            key=f"open_{row['app_id']}",
            width="stretch",
            type="secondary" if is_selected else "primary",
        ):
            switch_to("views/dashboard.py", Selection(row["app_id"], row["run_id"]))

        # Uploaded apps were never "selected" by the demo strategy, so the
        # demo-selection wording would be a lie for them.
        is_demo = bool(row["selection_score"])
        with st.expander("Why this app is in the demo" if is_demo else "About this dataset"):
            for reason in row["selection_reasons"]:
                st.caption(f"• {reason}")


def main() -> None:
    manifest = load_manifest()
    render_sidebar(manifest)

    st.title("Application catalogue")
    st.markdown(
        '<div class="aipm-muted" style="margin-top:-0.35rem;margin-bottom:0.9rem">'
        "Every app below has a completed analysis on disk. Opening one is a single "
        "database read — no embedding, clustering or model call happens while you "
        "browse.</div>",
        unsafe_allow_html=True,
    )

    # Emptiness is decided by the catalogue, not the manifest: an app analysed
    # through the Upload page is real even when the precompute script has never
    # been run, and telling that user "no analysis found" would be false.
    catalog = load_catalog()
    if not catalog:
        st.info(
            "No analysis found yet. Run the precompute script, or upload your own "
            "CSVs on the **Upload dataset** page.",
            icon="🗄️",
        )
        st.code("python scripts/precompute_demo.py", language="bash")
        return

    # --- KPIs, summed over what is actually listed -------------------------
    # Taking these from the manifest would leave them disagreeing with the grid
    # the moment anything is uploaded.
    n_uploaded = sum(1 for row in catalog if not row["selection_score"])
    columns = st.columns(4)
    with columns[0]:
        stat(
            "Applications", str(len(catalog)),
            sub=f"{n_uploaded} uploaded" if n_uploaded else "analysed end to end",
        )
    with columns[1]:
        stat(
            "Reviews", compact_number(sum(row["n_reviews"] for row in catalog)),
            sub="segmented and clustered",
        )
    with columns[2]:
        stat(
            "Hidden needs", str(sum(row["n_needs"] for row in catalog)),
            sub="each one cited and scored",
        )
    with columns[3]:
        if manifest:
            stat(
                "Precompute cost", f"${manifest.total_cost_usd:.2f}",
                sub=f"{manifest.duration_s / 60:.0f} min of compute",
            )
        else:
            stat("Precompute cost", "—", sub="no precomputed run")

    rule()

    # --- filters -----------------------------------------------------------
    categories = sorted({row["category"] for row in catalog})
    # Filters sit in one row above the grid, with visible labels - a collapsed
    # label leaves an unexplained empty box.
    filters = st.columns([0.42, 0.34, 0.24])
    with filters[0]:
        query = st.text_input("Search", placeholder="Filter by name…")
    with filters[1]:
        chosen = st.multiselect("Category", categories, placeholder="All categories")
    with filters[2]:
        order = st.selectbox("Sort by", ["Most needs", "Most reviews", "Name", "Rating"])

    rows = [
        row for row in catalog
        if (not query or query.lower() in row["name"].lower())
        and (not chosen or row["category"] in chosen)
    ]
    sort_keys = {
        "Most needs": lambda r: -r["n_needs"],
        "Most reviews": lambda r: -r["n_reviews"],
        "Name": lambda r: r["name"].lower(),
        "Rating": lambda r: -(r["store_score"] or 0),
    }
    rows.sort(key=sort_keys[order])

    if not rows:
        st.info("No applications match those filters.", icon="🔍")
        return

    section(f"{len(rows)} application{'s' if len(rows) != 1 else ''}")

    selection = get_selection()
    for start in range(0, len(rows), 3):
        for column, row in zip(st.columns(3), rows[start : start + 3], strict=False):
            with column:
                app_card(row, is_selected=row["app_id"] == selection.app_id)

    rule()
    kicker("How this works")
    st.markdown(
        "Reviews are split into segments, embedded, and clustered by density. "
        "Python computes every number — volume, cohesion, time spread, citation "
        "survival — and the language model only reads a dozen representative "
        "quotes per theme to name the latent need. Confidence is the weighted "
        "combination of six measured components, never a model's guess."
    )


main()
