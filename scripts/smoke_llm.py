"""Smoke-test the configured LLM endpoint (Autorouter / OpenRouter)."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.config import Settings  # noqa: E402


def main() -> None:
    s = Settings()
    print(
        {
            "llm_enabled": s.llm_enabled,
            "model": s.openrouter_model,
            "url": s.llm_base_url,
            "key_set": bool(s.openrouter_api_key),
        }
    )
    if not s.llm_enabled:
        print("FAIL: no API key")
        sys.exit(1)
    r = httpx.post(
        s.llm_base_url,
        headers={
            "Authorization": f"Bearer {s.openrouter_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": s.openrouter_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": 'Reply with exactly this JSON object: {"ok": true}',
                }
            ],
        },
        timeout=60.0,
    )
    print("status", r.status_code)
    print(r.text[:800])
    r.raise_for_status()


if __name__ == "__main__":
    main()
