"""Theme: palette resolution and the injected stylesheet.

The CSS here is global and injected at runtime, which makes it the one place in
the UI that can silently break Streamlit's own chrome.
"""

from __future__ import annotations

import re

import pytest

from app.theme import (
    DARK,
    LIGHT,
    Palette,
    band_color,
    band_icon,
    compact_number,
    _css,
)


def _strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


class TestInjectedCss:
    def test_does_not_use_a_broad_class_attribute_selector(self):
        """`[class*="st-"]` clobbers Streamlit's Material Symbols icon font.

        Streamlit applies its icon face with a single-class rule
        (`.st-emotion-cache-xxxx`, specificity 0-1-0). An attribute selector has
        the same specificity, and this stylesheet is injected *after* theirs, so
        it wins - and every icon renders as its literal ligature name
        ("keyboard_double_arrow_left", "arrow_right", "upload").
        """
        # Strip comments first: the rule above is *documented* in the stylesheet,
        # so a naive substring check matches its own explanation.
        css = _strip_css_comments(_css(LIGHT))
        assert '[class*="st-"]' not in css
        assert "[class*='st-']" not in css

    def test_font_is_applied_to_ancestors_so_it_inherits(self):
        css = _css(LIGHT)
        assert '[data-testid="stAppViewContainer"]' in css
        assert "html, body" in css

    def test_never_sets_font_family_on_a_wildcard(self):
        """A `* { font-family }` rule would break icons the same way."""
        css = _strip_css_comments(_css(LIGHT))
        assert "* {" not in css

    @pytest.mark.parametrize("palette", [LIGHT, DARK])
    def test_every_token_is_emitted(self, palette: Palette):
        css = _css(palette)
        for token in (palette.surface, palette.ink, palette.series[0], palette.good):
            assert token in css


class TestPalette:
    def test_light_and_dark_are_distinct_surfaces(self):
        assert LIGHT.surface != DARK.surface
        assert LIGHT.ink != DARK.ink

    def test_status_colours_are_mode_invariant(self):
        """Status is fixed and never themed."""
        for role in ("good", "warning", "serious", "critical"):
            assert getattr(LIGHT, role) == getattr(DARK, role)

    def test_categorical_slots_have_a_fixed_order(self):
        assert LIGHT.series[0] == "#2a78d6"
        assert len(LIGHT.series) == len(DARK.series)

    def test_band_colour_and_icon_agree_on_the_bands(self):
        for band in ("high", "medium", "low"):
            assert band_color(LIGHT, band)
            assert band_icon(band)

    def test_band_icon_is_distinct_per_band(self):
        """Status must never depend on hue alone."""
        icons = {band_icon(b) for b in ("high", "medium", "low")}
        assert len(icons) == 3


class TestCompactNumber:
    @pytest.mark.parametrize(
        "value,expected",
        [(999, "999"), (1_000, "1k"), (31_951, "32k"), (1_000_000, "1M")],
    )
    def test_formats(self, value, expected):
        assert compact_number(value) == expected


class TestRuleSpecificity:
    """Streamlit styles `hr` with a descendant selector.

    `.st-emotion-cache-xxxx hr { margin: 2em 0 }` is specificity 0-1-1, which
    outranks a bare `.aipm-rule` (0-1-0) wherever it sits in the cascade. The
    divider silently rendered at 2em for that reason; the element selector is
    what makes our declaration competitive.
    """

    def test_rule_selector_includes_the_element(self):
        css = _strip_css_comments(_css(LIGHT))
        assert "hr.aipm-rule" in css, (
            "a bare .aipm-rule loses to Streamlit's `<cache-class> hr` rule"
        )
        # The bare form must not be the one carrying the declaration.
        assert "\n  .aipm-rule {" not in css

    def test_rule_declares_no_margin(self):
        css = _strip_css_comments(_css(LIGHT))
        block = css.split("hr.aipm-rule")[1].split("}")[0]
        assert "margin: 0" in block
