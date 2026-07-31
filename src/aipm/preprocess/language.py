"""Cheap, dependency-free English detection.

A full language-id model is overkill here and adds a heavyweight dependency for
one boolean. Two orthogonal signals - script and function words - are enough to
keep non-English reviews from fragmenting the clusters, which is the only thing
this filter exists to do.

The function is deliberately biased toward *keeping* text: dropping a real
English review costs us evidence, whereas letting one Spanish review through
costs almost nothing.
"""

from __future__ import annotations

import re

#: High-frequency English function words. Content words are avoided on purpose -
#: they overlap heavily with other Latin-script languages.
_STOPWORDS = frozenset(
    """
    the be to of and a in that have i it for not on with he as you do at this but
    his by from they we say her she or an will my one all would there their what
    so up out if about who get which go me when make can like time no just him know
    take people into year your good some could them see other than then now look
    only come its over also back after use two how our work first well way even
    new want because any these give day most us is are was were app phone
    """.split()
)

_WORD_RE = re.compile(r"[a-z']+")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_ASCII_LETTER_RE = re.compile(r"[a-zA-Z]")

#: Below this share of Latin letters the text is a different script entirely.
MIN_LATIN_SHARE = 0.65
#: Share of tokens that must be English function words, for texts long enough to
#: judge. Short texts fall back to the script check alone.
MIN_STOPWORD_SHARE = 0.10
MIN_TOKENS_FOR_STOPWORD_CHECK = 6


def latin_share(text: str) -> float:
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    ascii_letters = _ASCII_LETTER_RE.findall(text)
    return len(ascii_letters) / len(letters)


def stopword_share(text: str) -> float:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in _STOPWORDS) / len(words)


def is_english(text: str) -> bool:
    """True if `text` is plausibly English.

    >>> is_english("the app keeps crashing when I open my inbox")
    True
    >>> is_english("la aplicacion se cierra sola cuando abro el correo")
    False
    """
    if not text or not text.strip():
        return False
    if latin_share(text) < MIN_LATIN_SHARE:
        return False
    words = _WORD_RE.findall(text.lower())
    if len(words) < MIN_TOKENS_FOR_STOPWORD_CHECK:
        # Too short to judge by function words; the script check already passed.
        return True
    return stopword_share(text) >= MIN_STOPWORD_SHARE


def detect_language(text: str) -> str:
    """Coarse label for `Review.lang`. Only English is positively identified."""
    return "en" if is_english(text) else "other"
