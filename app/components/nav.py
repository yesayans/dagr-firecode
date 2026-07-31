"""The sidebar navigation.

Replaces Streamlit's built-in page tree, which is a flat list of five siblings.
Three of those pages only mean anything *inside* a selected application, so a
flat list makes moving between levels look like moving between peers. This
renders the actual shape instead:

    ← All applications
    ─────────────
    Google Wallet          ← only once an app is chosen
       Dashboard
       Needs & Evidence
       Chat
    ─────────────
    Upload dataset

Items are buttons rather than links because navigation goes through
`state.switch_to`, which carries the selection in the query string. The current
view renders disabled - you cannot navigate to where you already are, and the
disabled state marks it without relying on colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from app.state import Selection, get_selection, selected_result, switch_to


@dataclass(frozen=True)
class NavPages:
    """The registered views, so the nav and the entrypoint agree on identity."""

    catalogue: StreamlitPage
    dashboard: StreamlitPage
    details: StreamlitPage
    chat: StreamlitPage
    upload: StreamlitPage

    @property
    def app_scoped(self) -> tuple[tuple[StreamlitPage, str], ...]:
        """The views that require a selected application, in reading order."""
        return (
            (self.dashboard, "Dashboard"),
            (self.details, "Needs & Evidence"),
            (self.chat, "Chat"),
        )


def _is_current(page: StreamlitPage, current: StreamlitPage) -> bool:
    return page.url_path == current.url_path


def _nav_item(
    page: StreamlitPage,
    label: str,
    current: StreamlitPage,
    *,
    selection: Selection | None = None,
    key_prefix: str = "nav",
) -> None:
    active = _is_current(page, current)
    if st.button(
        label,
        key=f"{key_prefix}_{page.url_path or 'catalogue'}",
        width="stretch",
        type="secondary" if active else "tertiary",
        disabled=active,
    ):
        switch_to(page, selection)


def render_sidebar_nav(pages: NavPages, current: StreamlitPage) -> None:
    """Draw the nav. Call once, from the entrypoint, before running the view."""
    selection = get_selection()

    with st.sidebar:
        st.markdown("### AI PM Assistant")

        showed_anything = False

        # An "up" action, not a sibling in a set - so it is *omitted* on the
        # catalogue rather than shown disabled. A disabled item earns its place
        # only when it answers "which of these am I on?", which is why the view
        # items below keep that treatment; here there is nowhere up to go and
        # nothing to mark, so a greyed-out control would be pure noise.
        #
        # Clears the selection: the catalogue is a level up, so carrying `?app=`
        # there would describe a state it is not in.
        if not _is_current(pages.catalogue, current):
            _nav_item(
                pages.catalogue, "← All applications", current,
                selection=Selection(None, None),
            )
            showed_anything = True

        if selection.is_set:
            result = selected_result()
            name = result.app.name if result else selection.app_id
            st.markdown(
                f'<div class="aipm-nav-app">{escape(name)}</div>',
                unsafe_allow_html=True,
            )
            # Indented with a spacer column, not a wrapper div: Streamlit renders
            # each widget into its own container, so raw <div> tags emitted
            # around st.button calls never actually enclose them.
            for page, label in pages.app_scoped:
                spacer, item = st.columns([0.08, 0.92])
                with item:
                    _nav_item(page, label, current, selection=selection)
                del spacer
            showed_anything = True

        # Only a separator if there is something above it to separate from - on
        # the catalogue this block is the whole nav.
        if showed_anything:
            st.markdown('<hr class="aipm-rule"/>', unsafe_allow_html=True)
        _nav_item(
            pages.upload, "Upload dataset", current, selection=Selection(None, None)
        )
        # Separates the nav from whatever contextual content the view adds below.
        st.markdown('<hr class="aipm-rule"/>', unsafe_allow_html=True)
