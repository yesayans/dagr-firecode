"""Review text cleaning. Pure functions, no IO."""

from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
#: Three or more repeats of the same character ("sooooo", "!!!!!") carry no extra
#: meaning but do fragment the embedding space.
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_emoji(text: str) -> str:
    """Drop pictographic characters. They survive NFKC and add embedding noise."""
    return "".join(ch for ch in text if unicodedata.category(ch) != "So")


def clean_review_text(text: str, *, drop_emoji: bool = True) -> str:
    """Normalise a raw review into something worth embedding.

    Deliberately conservative: this keeps punctuation and casing intact because
    both carry signal for the LLM when it reads representative quotes later.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _URL_RE.sub(" ", out)
    out = _EMAIL_RE.sub(" ", out)
    if drop_emoji:
        out = strip_emoji(out)
    out = _ELONGATION_RE.sub(r"\1\1", out)
    out = _WHITESPACE_RE.sub(" ", out)
    return out.strip()


def token_count(text: str) -> int:
    """Whitespace token count. Good enough for a length threshold, and free."""
    return len(text.split())


def normalise_for_dedup(text: str) -> str:
    """Aggressive normalisation used only for duplicate detection.

    Case, punctuation and whitespace are all discarded so that "Great app!" and
    "great app" collapse to the same key.
    """
    lowered = unicodedata.normalize("NFKD", text.lower())
    kept = [ch for ch in lowered if ch.isalnum() or ch.isspace()]
    return _WHITESPACE_RE.sub(" ", "".join(kept)).strip()
