"""Per-review quality weighting and praise detection.

Two distinct jobs, both cheap and deterministic:

* `quality_weight` downweights low-information text so that 400 "good app"
  reviews cannot outvote 40 detailed bug reports.
* `is_pure_praise` marks segments that carry no actionable signal. They stay in
  the statistics (they are real ratings) but are kept out of clustering, where
  they otherwise form one enormous useless cluster.
"""

from __future__ import annotations

import re

from aipm.preprocess.clean import token_count

_PRAISE_TERMS = frozenset(
    """
    good great awesome excellent perfect love loved lovely nice best amazing
    fantastic wonderful cool super fine ok okay useful helpful easy convenient
    recommend recommended thanks thank thx brilliant fab decent solid
    """.split()
)

_PROBLEM_TERMS = frozenset(
    """
    crash crashes crashed crashing bug bugs broken freeze freezes frozen error
    errors fail fails failed failing slow lag laggy stuck hang hangs glitch
    glitches wrong cannot cant can't won't wont doesn't doesnt unable refuse
    refused missing lost delete deleted charge charged refund scam fix fixed
    problem problems issue issues terrible awful worst hate useless annoying
    """.split()
)

#: Reviewers describing a workaround are describing an unmet need. This is the
#: strongest hidden-need signal in the corpus, so it earns a quality bonus.
_WORKAROUND_TERMS = frozenset(
    """
    workaround instead manually myself website browser desktop reinstall
    uninstall restart resort alternative switched switch competitor
    """.split()
)

_WORD_RE = re.compile(r"[a-z']+")

#: Below this many tokens a review cannot say anything specific.
_MIN_INFORMATIVE_TOKENS = 5
_RICH_TOKENS = 25


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def lexical_diversity(text: str) -> float:
    """Unique-token ratio. Catches "good good good good" style spam."""
    words = _words(text)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def quality_weight(text: str) -> float:
    """A 0..1 weight expressing how much signal this review carries.

    Length is the dominant term (short reviews cannot be specific), modulated by
    lexical diversity and a bonus for concrete problem or workaround vocabulary.
    """
    if not text or not text.strip():
        return 0.0
    n_tokens = token_count(text)
    if n_tokens < _MIN_INFORMATIVE_TOKENS:
        return 0.15

    # Saturating length score: past ~25 tokens, more words stop adding evidence.
    length_score = min(1.0, n_tokens / _RICH_TOKENS)
    diversity = lexical_diversity(text)

    words = set(_words(text))
    specificity = 0.0
    if words & _PROBLEM_TERMS:
        specificity += 0.15
    if words & _WORKAROUND_TERMS:
        specificity += 0.15

    weight = 0.55 * length_score + 0.30 * diversity + specificity
    return round(min(1.0, max(0.0, weight)), 4)


def is_pure_praise(text: str) -> bool:
    """True for short, wholly positive text with no problem vocabulary.

    >>> is_pure_praise("great app love it")
    True
    >>> is_pure_praise("great app but it crashes on login")
    False
    """
    words = _words(text)
    if not words:
        return True
    if set(words) & _PROBLEM_TERMS:
        return False
    if len(words) > 12:
        # Long text without praise-only structure is worth clustering even if
        # positive; it may describe a workaround or an unmet expectation.
        return False
    return bool(set(words) & _PRAISE_TERMS)
