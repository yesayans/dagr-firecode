"""Stable hashing for cache keys. Same input must always give the same key."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any) -> str:
    """Deterministic hash of any JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def text_hash(text: str) -> str:
    """Cache key for an embedding. Normalised so trivial whitespace differences hit."""
    normalised = " ".join(text.strip().split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def new_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{stable_hash(list(parts))}"
