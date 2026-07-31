"""Typed session state and cached resource wiring.

Every cross-page value goes through an accessor here rather than a raw
`st.session_state["..."]` string scattered across pages - a typo in one page
otherwise silently creates a second, empty piece of state.

Caching follows the two Streamlit rules that matter:

* `@st.cache_resource` for things that hold connections or models (the
  repository, the LLM client, the embedder) - one per process, never copied;
* `@st.cache_data` for values derived from them, keyed by ids, so switching apps
  is instant and switching back is free.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aipm.chat.retriever import RetrievalIndex, Retriever, build_index  # noqa: E402
from aipm.config import Settings  # noqa: E402
from aipm.embeddings.cache import EmbeddingCache  # noqa: E402
from aipm.embeddings.provider import build_embedding_provider  # noqa: E402
from aipm.embeddings.store import EmbeddingService  # noqa: E402
from aipm.llm.client import LlmClient, build_llm_client  # noqa: E402
from aipm.schemas import AnalysisResult, ChatMessage, DemoManifest, Review  # noqa: E402
from aipm.storage.sqlite_repo import SqliteRepository  # noqa: E402

if TYPE_CHECKING:  # avoids importing Streamlit internals at runtime
    from streamlit.navigation.page import StreamlitPage

# --- session keys -----------------------------------------------------------

_APP_ID = "aipm.app_id"
_RUN_ID = "aipm.run_id"
_CHAT = "aipm.chat_history"
_UPLOAD = "aipm.upload"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings()


@st.cache_resource(show_spinner=False)
def get_repository() -> SqliteRepository:
    """One connection for the process. Streamlit reruns must not reopen it."""
    settings = get_settings()
    settings.ensure_dirs()
    repository = SqliteRepository(settings.sqlite_path)
    repository.init_schema()
    return repository


@st.cache_resource(show_spinner=False)
def get_llm_client() -> LlmClient:
    return build_llm_client(get_settings())


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    provider = build_embedding_provider(settings)
    return EmbeddingService(
        provider, EmbeddingCache(settings.embedding_cache_dir / "vectors.sqlite")
    )


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=300)
def load_manifest() -> DemoManifest | None:
    return get_repository().get_demo_manifest()


@st.cache_data(show_spinner=False)
def load_result(app_id: str, run_id: str | None = None) -> AnalysisResult | None:
    """Keyed by `(app_id, run_id)` so a run is deserialised once per session."""
    repository = get_repository()
    if run_id:
        result = repository.get_result(run_id)
        if result is not None:
            return result
    return repository.get_latest_result(app_id)


@st.cache_data(show_spinner=False)
def load_reviews(app_id: str, run_id: str | None = None) -> list[Review]:
    """`run_id` participates in the cache key only to invalidate on re-analysis."""
    return get_repository().get_reviews(app_id)


@st.cache_data(show_spinner=False)
def load_catalog() -> list[dict[str, Any]]:
    """Rows for the catalogue grid: every app that has a completed run.

    Sourced from storage, not from the demo manifest. The manifest records what
    `precompute_demo.py` produced, so reading it directly meant an app analysed
    through the Upload page was stored, openable by URL, and yet absent from the
    catalogue. The manifest still supplies the selection rationale for the demo
    apps, which is metadata the repository has no business knowing.
    """
    repository = get_repository()
    manifest = load_manifest()
    reasons = {
        entry.app_id: (entry.selection_score, entry.selection_reasons)
        for entry in (manifest.entries if manifest else [])
    }

    rows: list[dict[str, Any]] = []
    for entry in repository.list_catalog():
        entry.selection_score, entry.selection_reasons = reasons.get(
            entry.app_id, (0.0, ["Analysed from an uploaded dataset."])
        )
        app = repository.get_app(entry.app_id)
        rows.append(
            {
                "app_id": entry.app_id,
                "name": entry.app_name,
                "category": entry.category,
                "run_id": entry.run_id,
                "n_reviews": entry.n_reviews,
                "n_clusters": entry.n_clusters,
                "n_needs": entry.n_needs,
                "status": entry.status.value,
                "error": entry.error,
                # Zero for anything the demo strategy did not pick, which is how
                # the catalogue tells an uploaded app from a demo one.
                "selection_score": entry.selection_score,
                "selection_reasons": entry.selection_reasons,
                "store_score": app.score if app else None,
                "downloads": app.downloads_numeric if app else None,
                "description": app.description if app else "",
            }
        )
    return rows


@st.cache_resource(show_spinner="Building the retrieval index…")
def get_retrieval_index(app_id: str, run_id: str) -> RetrievalIndex:
    """Cached as a *resource*: it holds a numpy matrix that must not be copied."""
    result = load_result(app_id, run_id)
    if result is None:
        return RetrievalIndex()
    reviews = load_reviews(app_id, run_id)
    embed = None
    try:
        embed = get_embedding_service().embed_texts
    except Exception:
        pass  # BM25-only retrieval is a valid degraded mode
    return build_index(result, reviews, embed_texts=embed)


def get_retriever(app_id: str, run_id: str) -> Retriever:
    embed = None
    try:
        embed = get_embedding_service().embed_texts
    except Exception:
        pass
    return Retriever(get_retrieval_index(app_id, run_id), embed_texts=embed)


def clear_caches() -> None:
    """After a new run is persisted, stale cached reads must not survive."""
    load_manifest.clear()
    load_result.clear()
    load_reviews.clear()
    load_catalog.clear()
    get_retrieval_index.clear()


# ---------------------------------------------------------------------------
# Selection state
# ---------------------------------------------------------------------------


#: Query-string keys. The URL is the source of truth for which app is open.
QUERY_APP = "app"
QUERY_RUN = "run"


@dataclass(frozen=True)
class Selection:
    app_id: str | None
    run_id: str | None

    @property
    def is_set(self) -> bool:
        return bool(self.app_id)

    def to_query(self) -> dict[str, str]:
        params = {QUERY_APP: self.app_id} if self.app_id else {}
        if self.app_id and self.run_id:
            params[QUERY_RUN] = self.run_id
        return params


def switch_to(page: "str | StreamlitPage", selection: Selection | None = None) -> None:
    """Navigate to `page` in the same tab, carrying the selection in the URL.

    Always use this instead of a bare `st.switch_page`. `st.switch_page`
    **clears every query parameter** when `query_params` is omitted ("all
    non-embed query parameters are cleared during navigation"), so a bare call
    silently drops `?app=...` and the destination URL stops identifying
    anything - breaking refresh, bookmarks and sharing.

    Pass `Selection(None, None)` to deliberately clear, which is what navigating
    to the catalogue or the upload page should do. Clearing drops the session
    state too, not just the query string: leaving `aipm.app_id` behind would let
    `get_selection` resurrect the app from session state, and the sidebar would
    still show an app section on a catalogue URL that says there is none.
    """
    if selection is None:
        selection = get_selection()
    elif not selection.is_set:
        st.session_state.pop(_APP_ID, None)
        st.session_state.pop(_RUN_ID, None)
        reset_chat()

    st.switch_page(page, query_params=selection.to_query())


def get_selection() -> Selection:
    """Read the selection, preferring the URL.

    A new browser tab is a new Streamlit session with empty `session_state`, so
    session state alone cannot survive a refresh or a pasted link. The query
    string can, so it wins; session state is the fallback for the brief window
    during navigation.

    Deliberately does **not** write back to the query string. Navigation carries
    the selection via `switch_to`, and an auto-rewrite here would re-stamp
    `?app=...` onto the catalogue after you left an app - a URL that claims a
    selection the page does not have.
    """
    params = st.query_params
    app_id = params.get(QUERY_APP) or st.session_state.get(_APP_ID)
    run_id = params.get(QUERY_RUN) or st.session_state.get(_RUN_ID)

    if app_id and st.session_state.get(_APP_ID) != app_id:
        # Landed here by URL rather than by clicking. Adopt it, and drop any chat
        # history belonging to the app we were previously looking at.
        st.session_state[_APP_ID] = app_id
        st.session_state[_RUN_ID] = run_id
        reset_chat()

    return Selection(app_id, run_id)


def select_app(app_id: str, run_id: str | None = None) -> None:
    previous = st.session_state.get(_APP_ID)
    st.session_state[_APP_ID] = app_id
    st.session_state[_RUN_ID] = run_id
    # Keep the address bar in step, so a refresh or a copied URL still resolves.
    st.query_params.update(Selection(app_id, run_id).to_query())
    if previous != app_id:
        # Chat is scoped to one app; carrying history across apps would let the
        # model answer about the wrong product.
        reset_chat()


def selected_result() -> AnalysisResult | None:
    selection = get_selection()
    if not selection.is_set:
        return None
    return load_result(selection.app_id, selection.run_id)


def require_selection(page_name: str, *, route: str = "Dashboard") -> AnalysisResult | None:
    """Guard for pages that need an app. Renders its own empty state.

    Reached when someone opens a deep page with no `?app=` - a bare bookmark, or
    a link that lost its query string. `route` is this page's URL segment, so the
    recovery links keep you on the page you asked for instead of bouncing you to
    the dashboard.

    Also the right place to repair the URL. Streamlit's own sidebar nav links are
    bare hrefs with no query string, so arriving that way leaves the address bar
    without `?app=` even though session state still knows the app. Repairing it
    *here* rather than in `get_selection` is deliberate: only app-scoped pages
    call this, so the catalogue never gets stamped with a selection it does not
    have.
    """
    selection = get_selection()
    result = selected_result()
    if result is not None:
        if selection.app_id and st.query_params.get(QUERY_APP) != selection.app_id:
            # Only when it differs - an unconditional write every rerun loops.
            st.query_params.update(selection.to_query())
        return result

    st.info(
        f"**No application selected.** {page_name} needs one. "
        "Pick an app on the Home page, or open a link that names one.",
        icon="📱",
    )
    if st.button("Go to Home", type="primary"):
        switch_to("views/catalogue.py", Selection(None, None))

    catalog = load_catalog()
    if catalog:
        st.caption("Or jump straight in:")
        # Buttons, not links: this is recovery on the page you already asked for,
        # so it should fill in the missing app here rather than open a new tab.
        # `select_app` writes the query string, so the URL becomes shareable.
        columns = st.columns(min(4, len(catalog)))
        for column, row in zip(columns, catalog[:4], strict=False):
            with column:
                if st.button(row["name"][:24], key=f"quick_{route}_{row['app_id']}",
                             width="stretch"):
                    select_app(row["app_id"], row["run_id"])
                    st.rerun()
    return None


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------


def chat_history() -> list[ChatMessage]:
    return st.session_state.setdefault(_CHAT, [])


def append_chat(message: ChatMessage) -> None:
    chat_history().append(message)


def reset_chat() -> None:
    st.session_state[_CHAT] = []


# ---------------------------------------------------------------------------
# Upload state
# ---------------------------------------------------------------------------


def upload_state() -> dict[str, Any]:
    return st.session_state.setdefault(_UPLOAD, {})


def set_upload(key: str, value: Any) -> None:
    upload_state()[key] = value


def clear_upload() -> None:
    st.session_state[_UPLOAD] = {}
