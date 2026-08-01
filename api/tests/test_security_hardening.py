"""Regression tests for security hardening (hashes, upload limits, LLM caps)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import get_settings
from src.data_ingestion import _stable_review_id
from src.llm_extractor import LlmGapResponse
from src.matching_space import _assert_under_cache, _sha256_bytes
from src.null_model import roadmap_content_hash


def test_stable_review_id_is_deterministic():
    a = _stable_review_id("pkg", "Hello", 3, "2020-01-01")
    b = _stable_review_id("pkg", "Hello", 3, "2020-01-01")
    assert a == b
    assert len(a) == 16


def test_roadmap_content_hash_stable():
    assert roadmap_content_hash(["a", "b"]) == roadmap_content_hash(["a", "b"])
    assert roadmap_content_hash(["a"]) != roadmap_content_hash(["b"])


def test_llm_response_rejects_oversized_fields():
    with pytest.raises(ValidationError):
        LlmGapResponse(
            latent_need="x" * 501,
            verdict="UNVERIFIED",
            confidence=50,
            confidence_justification="ok",
            one_sentence_summary="ok",
            cited_review_ids=["r1"],
            surface_complaint="ok",
            workaround="ok",
        )
    with pytest.raises(ValidationError):
        LlmGapResponse(
            latent_need="ok",
            verdict="UNVERIFIED",
            confidence=50,
            confidence_justification="ok",
            one_sentence_summary="ok",
            cited_review_ids=["r1"],
            surface_complaint="y" * 501,
            workaround="",
        )


def test_cache_path_rejects_traversal(tmp_path: Path):
    cache = tmp_path / "cache" / "null_thresholds"
    cache.mkdir(parents=True)
    ok = cache / "abc.pkl"
    ok.write_bytes(b"x")
    settings = SimpleNamespace(data_dir=tmp_path)
    assert _assert_under_cache(ok, settings).exists()  # type: ignore[arg-type]

    outside = tmp_path / "evil.pkl"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError):
        _assert_under_cache(outside, settings)  # type: ignore[arg-type]


def test_sha256_bytes_matches_stdlib():
    data = b"hello"
    assert _sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_custom_upload_rejects_oversized_csv(monkeypatch: pytest.MonkeyPatch):
    from src import main as main_mod

    monkeypatch.setenv("MAX_CSV_UPLOAD_BYTES", "64")
    get_settings.cache_clear()
    try:
        client = TestClient(main_mod.app)
        big = b"review_text,rating\n" + (b"x" * 200) + b",1\n"
        res = client.post(
            "/apps/custom",
            data={"app_name": "Tiny"},
            files={"reviews": ("reviews.csv", big, "text/csv")},
        )
        assert res.status_code == 413
    finally:
        get_settings.cache_clear()


def test_custom_upload_rejects_non_csv_extension(monkeypatch: pytest.MonkeyPatch):
    from src import main as main_mod

    get_settings.cache_clear()
    try:
        client = TestClient(main_mod.app)
        body = (
            b"review_text,rating\n"
            b'"This is a long enough complaint about missing offline sync forever.",2\n'
        )
        res = client.post(
            "/apps/custom",
            data={"app_name": "Tiny"},
            files={"reviews": ("reviews.exe", body, "application/octet-stream")},
        )
        assert res.status_code == 400
        assert "csv" in res.json()["detail"].lower()
    finally:
        get_settings.cache_clear()
