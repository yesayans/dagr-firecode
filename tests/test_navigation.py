"""Deep-link routing.

The selected app travels in the query string, so a dashboard URL identifies what
it is showing. Two Streamlit behaviours make this fragile enough to pin down:

* `st.switch_page` **clears all query parameters** when `query_params` is
  omitted, so any bare call silently drops the selection from the URL;
* a new browser tab (or a refresh) is a new session with empty `session_state`,
  so the URL is the only thing that can carry a selection into it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.state import QUERY_APP, QUERY_RUN, Selection

APP_DIR = Path(__file__).resolve().parents[1] / "app"


class TestSelectionQuery:
    def test_carries_app_and_run(self):
        assert Selection("15", "run_abc").to_query() == {
            QUERY_APP: "15", QUERY_RUN: "run_abc"
        }

    def test_run_is_optional(self):
        assert Selection("15", None).to_query() == {QUERY_APP: "15"}

    def test_empty_selection_clears_the_query(self):
        """`switch_to(..., Selection(None, None))` is how we navigate to Home."""
        assert Selection(None, None).to_query() == {}

    def test_run_without_app_is_meaningless_and_dropped(self):
        assert Selection(None, "run_abc").to_query() == {}

    def test_is_set_tracks_the_app_only(self):
        assert Selection("15", None).is_set
        assert not Selection(None, "run_abc").is_set

    @pytest.mark.parametrize("app_id", ["15", "com.example.app", "a b"])
    def test_values_pass_through_unmangled(self, app_id: str):
        """Streamlit encodes these itself; we must not double-encode."""
        assert Selection(app_id, None).to_query()[QUERY_APP] == app_id


def _switch_page_calls() -> list[tuple[str, int, bool]]:
    """Every `st.switch_page(...)` under app/, with whether it passes query_params."""
    found: list[tuple[str, int, bool]] = []
    for path in APP_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "switch_page":
                has_qp = any(kw.arg == "query_params" for kw in node.keywords)
                found.append((str(path.relative_to(APP_DIR)), node.lineno, has_qp))
    return found


class TestNavigationGoesThroughOneDoor:
    def test_switch_page_is_called_exactly_once(self):
        """Centralised in `switch_to`, so the query params cannot be forgotten."""
        calls = _switch_page_calls()
        assert len(calls) == 1, f"expected one call site, found {calls}"
        assert calls[0][0] == "state.py"

    def test_that_call_passes_query_params(self):
        _, _, has_query_params = _switch_page_calls()[0]
        assert has_query_params, (
            "st.switch_page without query_params clears the query string, "
            "dropping ?app= and breaking refresh, bookmarks and sharing"
        )

    def test_pages_navigate_via_switch_to(self):
        """No page should reach for `st.switch_page` directly."""
        offenders = [
            path.name
            for path in APP_DIR.rglob("*.py")
            if path.name != "state.py" and "st.switch_page(" in path.read_text()
        ]
        assert not offenders, f"these bypass switch_to: {offenders}"


def _switch_to_targets() -> list[tuple[str, str]]:
    """Every literal string passed to `switch_to(...)`, as (caller, target)."""
    found: list[tuple[str, str]] = []
    for path in APP_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name != "switch_to" or not node.args:
                continue
            target = node.args[0]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                found.append((path.name, target.value))
    return found


class TestNavigationTargets:
    """Targets are strings now, so a typo would only surface on click."""

    def test_there_are_targets_to_check(self):
        assert _switch_to_targets(), "expected literal switch_to targets to verify"

    def test_every_target_resolves_to_a_real_view(self):
        missing = [
            (caller, target)
            for caller, target in _switch_to_targets()
            if not (APP_DIR / target).exists()
        ]
        assert not missing, f"switch_to targets that do not exist: {missing}"

    def test_no_target_points_at_the_entrypoint(self):
        """`main.py` is the entrypoint, not a registered page."""
        offenders = [
            (caller, target)
            for caller, target in _switch_to_targets()
            if Path(target).name == "main.py"
        ]
        assert not offenders, f"cannot switch_page to the entrypoint: {offenders}"

    def test_every_target_is_registered_in_main(self):
        """`st.switch_page` rejects a file st.navigation does not know about."""
        registered = (APP_DIR / "main.py").read_text()
        unregistered = [
            (caller, target)
            for caller, target in _switch_to_targets()
            if f'"{target}"' not in registered
        ]
        assert not unregistered, f"not registered via st.Page: {unregistered}"

    def test_the_old_pages_directory_is_gone(self):
        """Streamlit ignores pages/ under st.navigation; leaving it is confusing."""
        assert not (APP_DIR / "pages").exists()
