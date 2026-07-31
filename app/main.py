"""Entrypoint.

    streamlit run app/main.py

Owns the two things that must happen exactly once per script run - page config
plus CSS, and page registration - then hands off to the selected view.

Navigation is registered with `position="hidden"`, so Streamlit's built-in page
tree is suppressed and `app/components/nav.py` draws a hierarchy-aware one in its
place. Calling `st.navigation` also makes Streamlit ignore any `pages/`
directory, so the old auto-nav is gone through the supported API rather than a
CSS override - which matters here, because overriding Streamlit's own chrome with
CSS is exactly what broke the Material Symbols icon font earlier.

Each view keeps a real `url_path`, so deep links (`/dashboard?app=15&run=...`),
refresh and the browser back button all keep working - the things hand-rolled
`st.session_state` routing would have cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.nav import NavPages, render_sidebar_nav  # noqa: E402
from app.theme import configure_app  # noqa: E402

# Page config and the stylesheet, before anything renders.
configure_app()


def _build_pages() -> NavPages:
    """Register the views. `url_path` is the public URL for each.

    Registered by **file path**, not by callable. `st.switch_page` only accepts a
    page that `st.navigation` knows about, and views need to name their targets
    (`switch_to("views/dashboard.py")`) - with callables they would have to
    import the page objects from here, which imports the views, which is a cycle.
    Paths resolve relative to this entrypoint.
    """
    return NavPages(
        catalogue=st.Page(
            "views/catalogue.py", title="Catalogue", icon="📱", url_path="", default=True
        ),
        dashboard=st.Page(
            "views/dashboard.py", title="Dashboard", icon="📊", url_path="dashboard"
        ),
        details=st.Page(
            "views/details.py", title="Needs & Evidence", icon="🔎", url_path="details"
        ),
        chat=st.Page("views/chat.py", title="Chat", icon="💬", url_path="chat"),
        upload=st.Page(
            "views/upload.py", title="Upload dataset", icon="📥", url_path="upload"
        ),
    )


pages = _build_pages()

current = st.navigation(
    [pages.catalogue, pages.dashboard, pages.details, pages.chat, pages.upload],
    position="hidden",
)

render_sidebar_nav(pages, current)
current.run()
