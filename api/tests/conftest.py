from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[1]
REPO = API_DIR.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(REPO))


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from src.config import Settings, get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    # Point data dir via Settings — we override after construct
    s = Settings()
    # Use temp store path by patching data_dir property usage in LocalJsonStore
    get_settings.cache_clear()
    return s


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    from src.config import get_settings
    from src.store import LocalJsonStore, reset_store_singleton

    get_settings.cache_clear()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    reset_store_singleton()
    store = LocalJsonStore(path=tmp_path / "store.json")
    yield store
    reset_store_singleton()
