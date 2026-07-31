"""Evidence-grounded job chat."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.chat import ChatTurn, _parse_llm_json, answer_job_chat, build_evidence_pack
from src.config import get_settings
from src.main import app as fastapi_app
from src.store import LocalJsonStore, reset_store_singleton


def _sample_job(*, status: str = "completed") -> dict:
    return {
        "id": "job-1",
        "status": status,
        "roadmap_source": "web",
        "summary": "Top need is offline sync.",
        "app": {
            "id": "app-1",
            "display_name": "Acme Notes",
            "package_name": "custom.acme.notes",
            "roadmap_source": "web",
        },
        "stats": {
            "total_reviews": 40,
            "reviews_total": 40,
            "reviews_need_bearing": 30,
            "clusters": 5,
            "roadmap_items": 3,
            "llm_used": True,
        },
        "gaps": [
            {
                "id": "g1",
                "rank": 1,
                "need": "Reliable offline sync",
                "verdict": "UNVERIFIED",
                "confidence": 80,
                "one_sentence_summary": "Users lose work offline.",
                "metrics": {
                    "keywords": ["offline", "sync"],
                    "matched_item_title": "Offline sync for teams",
                    "best_similarity": 0.4,
                    "matched_item_state": "open",
                },
                "evidence": [
                    {
                        "evidence_id": "rev-abc",
                        "source_type": "review",
                        "title": "review",
                        "snippet": "The offline sync never finishes and I lose notes.",
                        "url": None,
                        "payload": {"review_id": "rev-abc"},
                    }
                ],
            }
        ],
    }


def test_build_evidence_pack_includes_gaps_and_evidence():
    pack = build_evidence_pack(_sample_job())
    assert "Acme Notes" in pack
    assert "Gap #1" in pack
    assert "Reliable offline sync" in pack
    assert "rev-abc" in pack
    assert "offline sync never finishes" in pack.lower()


def test_build_evidence_pack_truncates():
    job = _sample_job()
    job["gaps"][0]["need"] = "x" * 20_000
    pack = build_evidence_pack(job, max_chars=500)
    assert len(pack) <= 500
    assert "truncated" in pack


def test_parse_llm_json_fenced():
    raw = (
        '```json\n{"answer": "Prioritize sync", "citations": '
        '[{"gap_rank": 1, "evidence_id": "rev-abc", "quote": "lose notes"}]}\n```'
    )
    parsed = _parse_llm_json(raw)
    assert parsed.answer.startswith("Prioritize")
    assert parsed.citations[0].gap_rank == 1


def test_answer_job_chat_mocked_llm():
    get_settings.cache_clear()
    settings = get_settings()
    object.__setattr__(settings, "openrouter_api_key", "test-key")
    object.__setattr__(settings, "openrouter_model", "test-model")
    object.__setattr__(
        settings, "llm_base_url", "https://example.test/v1/chat/completions"
    )

    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "Prioritize offline sync.",
                            "citations": [
                                {
                                    "gap_rank": 1,
                                    "evidence_id": "rev-abc",
                                    "quote": "lose notes",
                                },
                                {
                                    "gap_rank": 99,
                                    "evidence_id": "fake",
                                    "quote": "bogus",
                                },
                            ],
                        }
                    )
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = payload
    mock_resp.text = "ok"

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("src.chat.httpx.AsyncClient", return_value=mock_client):
        reply = asyncio.run(
            answer_job_chat(
                _sample_job(),
                "What should we prioritize?",
                history=[ChatTurn(role="user", content="hi")],
                settings=settings,
            )
        )

    assert "offline sync" in reply.answer.lower()
    assert any(c.gap_rank == 1 for c in reply.citations)
    assert all(c.gap_rank != 99 for c in reply.citations)
    assert all(c.evidence_id != "fake" for c in reply.citations)


def test_chat_route_409_when_not_completed(tmp_path, monkeypatch):
    get_settings.cache_clear()
    reset_store_singleton()
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    get_settings.cache_clear()
    store = LocalJsonStore(path=tmp_path / "store.json")
    app_row = store.upsert_app(
        {
            "package_name": "custom.x",
            "display_name": "X",
            "review_count": 1,
            "roadmap_source": "none",
        }
    )
    job = store.create_job(
        {
            "app_id": app_row["id"],
            "status": "running",
            "stage": "embedding",
            "progress": 40,
        }
    )
    monkeypatch.setattr("src.main.get_store", lambda settings=None: store)

    client = TestClient(fastapi_app)
    r = client.post(
        f"/jobs/{job['id']}/chat", json={"message": "hello", "history": []}
    )
    assert r.status_code == 409


def test_chat_route_503_without_llm(tmp_path, monkeypatch):
    get_settings.cache_clear()
    reset_store_singleton()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    object.__setattr__(settings, "openrouter_api_key", None)

    store = LocalJsonStore(path=tmp_path / "store.json")
    app_row = store.upsert_app(
        {
            "package_name": "custom.y",
            "display_name": "Y",
            "review_count": 1,
            "roadmap_source": "none",
        }
    )
    job = store.create_job(
        {
            "app_id": app_row["id"],
            "status": "queued",
            "stage": "queued",
            "progress": 0,
        }
    )
    store.update_job(
        job["id"], {"status": "completed", "stage": "done", "progress": 100}
    )

    monkeypatch.setattr("src.main.get_store", lambda settings=None: store)
    monkeypatch.setattr("src.main.get_settings", lambda: settings)

    client = TestClient(fastapi_app)
    r = client.post(f"/jobs/{job['id']}/chat", json={"message": "hello"})
    assert r.status_code == 503
