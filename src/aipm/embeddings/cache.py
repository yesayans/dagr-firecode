"""Disk cache for embedding vectors, keyed by ``sha256(text)``.

Applies to every backend, not just the paid one: re-running the demo precompute
must be near-free regardless of which provider produced the vectors.

SQLite rather than one file per hash - a 40k-unit app would otherwise create 40k
tiny files, which is slow to write and miserable to clean up.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aipm.utils.hashing import text_hash
from aipm.utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""


class EmbeddingCache:
    """Content-addressed vector store. Safe for concurrent readers."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0)
            self._local.conn = conn
        return conn

    def get_many(self, model: str, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Return ``{text_hash: vector}`` for whichever texts are already cached."""
        if not texts:
            return {}
        hashes = {text_hash(t) for t in texts}
        out: dict[str, np.ndarray] = {}
        conn = self._connect()
        hash_list = list(hashes)
        # SQLite caps host parameters (999 on older builds); chunk to stay under it.
        for start in range(0, len(hash_list), 900):
            chunk = hash_list[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT text_hash, dim, vector FROM embeddings "  # noqa: S608 - fixed placeholders
                f"WHERE model = ? AND text_hash IN ({placeholders})",
                (model, *chunk),
            ).fetchall()
            for th, dim, blob in rows:
                out[th] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return out

    def put_many(self, model: str, texts: Sequence[str], vectors: np.ndarray) -> None:
        if len(texts) == 0:
            return
        if len(texts) != len(vectors):
            raise ValueError(f"texts/vectors length mismatch: {len(texts)} vs {len(vectors)}")
        dim = int(vectors.shape[1])
        payload = [
            (model, text_hash(t), dim, np.asarray(v, dtype=np.float32).tobytes())
            for t, v in zip(texts, vectors, strict=True)
        ]
        conn = self._connect()
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                payload,
            )

    def stats(self) -> dict[str, int]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT model, COUNT(*) FROM embeddings GROUP BY model"
        ).fetchall()
        return {model: count for model, count in rows}

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


class NullEmbeddingCache:
    """No-op cache. Injected in tests that must observe every provider call."""

    def get_many(self, model: str, texts: Sequence[str]) -> dict[str, np.ndarray]:
        return {}

    def put_many(self, model: str, texts: Sequence[str], vectors: np.ndarray) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        return None
