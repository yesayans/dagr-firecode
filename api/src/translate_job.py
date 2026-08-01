"""Translate completed job analysis text into a UI locale via the configured LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_LLM_URL = "https://openrouter.ai/api/v1/chat/completions"

LocaleCode = Literal["en", "ru", "hy"]

LOCALE_NAMES: dict[str, str] = {
    "en": "English",
    "ru": "Russian",
    "hy": "Armenian",
}


class TranslatedGap(BaseModel):
    gap_id: str
    need: str = Field(max_length=800)
    one_sentence_summary: str = Field(default="", max_length=800)
    latent_reasoning: str = Field(default="", max_length=2500)
    confidence_rationale: str = Field(default="", max_length=2500)
    surface_complaints: list[str] = Field(default_factory=list)
    workarounds: list[str] = Field(default_factory=list)
    evidence: list[dict[str, str]] = Field(default_factory=list)


class TranslationPayload(BaseModel):
    summary: str = Field(default="", max_length=2000)
    gaps: list[TranslatedGap] = Field(default_factory=list)


def _clip(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _build_source_blob(job: dict[str, Any]) -> dict[str, Any]:
    gaps_out: list[dict[str, Any]] = []
    for g in job.get("gaps") or []:
        metrics = g.get("metrics") or {}
        evidence = []
        for ev in (g.get("evidence") or [])[:6]:
            if (ev.get("source_type") or "") != "review":
                continue
            evidence.append(
                {
                    "evidence_id": str(ev.get("evidence_id") or ""),
                    "title": _clip(str(ev.get("title") or ""), 120),
                    "snippet": _clip(str(ev.get("snippet") or ""), 280),
                }
            )
        gaps_out.append(
            {
                "gap_id": str(g.get("id") or ""),
                "need": _clip(str(g.get("need") or ""), 500),
                "one_sentence_summary": _clip(
                    str(g.get("one_sentence_summary") or ""), 500
                ),
                "latent_reasoning": _clip(str(g.get("latent_reasoning") or ""), 1200),
                "confidence_rationale": _clip(
                    str(g.get("confidence_rationale") or ""), 800
                ),
                "surface_complaints": [
                    _clip(str(x), 400)
                    for x in (metrics.get("surface_complaints") or [])[:3]
                    if str(x).strip()
                ],
                "workarounds": [
                    _clip(str(x), 400)
                    for x in (metrics.get("workarounds") or [])[:3]
                    if str(x).strip()
                ],
                "evidence": evidence,
            }
        )
    return {
        "summary": _clip(str(job.get("summary") or ""), 800),
        "gaps": gaps_out,
    }


def _parse_payload(raw: str) -> TranslationPayload:
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    return TranslationPayload.model_validate(data)


async def translate_job_analysis(
    job: dict[str, Any],
    locale: LocaleCode,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    if (job.get("status") or "") != "completed":
        raise ValueError("job is not completed")
    if locale not in LOCALE_NAMES:
        raise ValueError("unsupported locale")

    source = _build_source_blob(job)
    if locale == "en":
        # Identity mapping — useful for "show original" without another LLM call.
        return {
            "locale": "en",
            "summary": source["summary"],
            "gaps": source["gaps"],
            "model": None,
        }

    if not settings.llm_enabled:
        raise RuntimeError(
            "LLM is not configured (set OPENROUTER_API_KEY / Autorouter key)"
        )

    lang = LOCALE_NAMES[locale]
    system = (
        f"You translate product-analysis text into {lang}. "
        "Keep meaning, technical terms, package names, URLs, and review IDs unchanged. "
        "Do not invent new needs. Return JSON only with shape: "
        '{"summary": string, "gaps": [{"gap_id": string, "need": string, '
        '"one_sentence_summary": string, "latent_reasoning": string, '
        '"confidence_rationale": string, "surface_complaints": string[], '
        '"workarounds": string[], '
        '"evidence": [{"evidence_id": string, "title": string, "snippet": string}]}]}'
    )
    user = (
        f"Target language: {lang} ({locale}).\n"
        "Translate every string field in this JSON. Preserve gap_id and evidence_id.\n\n"
        f"{json.dumps(source, ensure_ascii=False)}"
    )

    url = (settings.llm_base_url or DEFAULT_LLM_URL).strip()
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "dagr-translate",
    }
    body: dict[str, Any] = {
        "model": settings.openrouter_model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if "openrouter.ai" in url:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM translate HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        raw = (
            ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ""
        )

    try:
        parsed = _parse_payload(raw)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.warning("translate parse failed: %s", e)
        raise RuntimeError(f"LLM returned invalid translation JSON: {e}") from e

    # Keep only known gap ids; drop hallucinated rows.
    known = {str(g.get("id") or "") for g in (job.get("gaps") or [])}
    gaps = [g.model_dump() for g in parsed.gaps if g.gap_id in known]
    return {
        "locale": locale,
        "summary": parsed.summary.strip(),
        "gaps": gaps,
        "model": settings.openrouter_model,
    }
