"""Normalisation of raw CSV values into the shapes `schemas.py` promises.

Pure functions only - no IO, no pandas. That keeps every rule here trivially
unit-testable and reusable from both the batch loader and the upload page.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime
from typing import Any

#: Play-store category strings arrive polluted with chart placements, e.g.
#: ``"#9 top free news & magazines, News & Magazines"``. The rank fragment is
#: presentation noise and must not become a category.
_CHART_RANK_RE = re.compile(
    r"^\s*#\d+\s+top\s+(?:free|paid|grossing)\b.*$", re.IGNORECASE
)

_DOWNLOADS_RE = re.compile(r"[\d.]+")
_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S")


def normalize_text(value: Any) -> str:
    """Collapse whitespace and normalise unicode. Never returns ``None``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    # Strip control characters that survive NFKC and break SQLite/JSON round-trips.
    # Whitespace controls (tab, newline, CR) are separators, not noise - dropping
    # them outright would join words that were on either side of them.
    text = "".join(
        ch for ch in text if ch.isspace() or not unicodedata.category(ch).startswith("C")
    )
    return " ".join(text.split())


def normalize_app_id(value: Any) -> str:
    """App ids are joined across two CSVs, so the type must not drift.

    Pandas will happily read ``app_id`` as int in one file and float in another;
    ``1`` and ``1.0`` would then fail to join silently.
    """
    text = normalize_text(value)
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_categories(value: Any) -> list[str]:
    """Split a category cell, dropping chart-placement fragments.

    >>> normalize_categories("#9 top free news & magazines, News & Magazines")
    ['News & Magazines']
    >>> normalize_categories("Food & Drink")
    ['Food & Drink']
    """
    raw = normalize_text(value)
    if not raw:
        return []
    out: list[str] = []
    for part in raw.split(","):
        cleaned = part.strip()
        if not cleaned or _CHART_RANK_RE.match(cleaned):
            continue
        if cleaned.lower().startswith("#"):
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out


def primary_category(value: Any) -> str:
    """The single category used for diversity-aware demo selection."""
    cats = normalize_categories(value)
    return cats[0] if cats else "Uncategorised"


def parse_downloads(value: Any) -> int | None:
    """Handle both plain integers and Play-store strings like ``"10,000,000+"``/``"1.5M"``."""
    text = normalize_text(value).lower().replace(",", "").replace("+", "")
    if not text:
        return None
    multiplier = 1
    if text and text[-1] in _MULTIPLIERS:
        multiplier = _MULTIPLIERS[text[-1]]
        text = text[:-1]
    match = _DOWNLOADS_RE.search(text)
    if not match:
        return None
    try:
        return int(float(match.group()) * multiplier)
    except (ValueError, OverflowError):
        return None


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = normalize_text(value)
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    try:  # last resort: let pandas' parser have a go
        import pandas as pd

        parsed = pd.to_datetime(text, errors="coerce")
        return None if parsed is None or pd.isna(parsed) else parsed.date()
    except Exception:  # pragma: no cover - pandas always present in practice
        return None


def parse_score(value: Any, *, lo: float = 1.0, hi: float = 5.0) -> float | None:
    """Star rating, clamped. Out-of-range values are data errors, not opinions."""
    text = normalize_text(value)
    if not text:
        return None
    try:
        score = float(text)
    except ValueError:
        return None
    if math.isnan(score):
        return None
    return min(max(score, lo), hi)


def parse_int(value: Any, default: int = 0) -> int:
    text = normalize_text(value).replace(",", "")
    if not text:
        return default
    try:
        return int(float(text))
    except (ValueError, OverflowError):
        return default
