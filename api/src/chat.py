"""Evidence-grounded chat over a completed analysis job."""

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
MAX_PACK_CHARS = 12_000
MAX_HISTORY_TURNS = 8
MAX_SNIPPET = 280

ChatRole = Literal["user", "assistant"]


class ChatTurn(BaseModel):
    role: ChatRole
    content: str


class ChatCitation(BaseModel):
    gap_rank: int | None = None
    evidence_id: str | None = None
    quote: str = ""


class ChatReply(BaseModel):
    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    model: str = ""


class LlmChatJson(BaseModel):
    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)


def build_evidence_pack(job: dict[str, Any], *, max_chars: int = MAX_PACK_CHARS) -> str:
    """Compact, citeable context from a completed job only."""
    app = job.get("app") or {}
    stats = job.get("stats") or {}
    gaps = sorted(job.get("gaps") or [], key=lambda g: int(g.get("rank") or 0))

    lines: list[str] = [
        f"App: {app.get('display_name') or ''} ({app.get('package_name') or ''})",
        f"Roadmap source: {job.get('roadmap_source') or app.get('roadmap_source') or 'none'}",
        f"Summary: {job.get('summary') or ''}",
        (
            f"Stats: reviews={stats.get('reviews_total') or stats.get('total_reviews')}, "
            f"need_bearing={stats.get('reviews_need_bearing')}, "
            f"clusters={stats.get('clusters')}, "
            f"roadmap_items={stats.get('roadmap_items')}, "
            f"llm_used={stats.get('llm_used')}"
        ),
        "",
        "GAPS AND EVIDENCE:",
    ]

    for g in gaps:
        rank = g.get("rank")
        metrics = g.get("metrics") or {}
        matched = metrics.get("matched_item_title")
        block = [
            f"### Gap #{rank}",
            f"Need: {g.get('need') or ''}",
            f"Verdict: {g.get('verdict') or ''}",
            f"Confidence: {g.get('confidence')}",
            f"Hiddenness: {metrics.get('hiddenness')}",
            f"Insight score: {metrics.get('insight_score')}",
            f"Summary: {g.get('one_sentence_summary') or ''}",
            f"Keywords: {', '.join(metrics.get('keywords') or g.get('keywords') or [])}",
        ]
        surfaces = metrics.get("surface_complaints") or []
        workarounds = metrics.get("workarounds") or []
        if surfaces:
            block.append(f"Surface complaints: {'; '.join(str(s) for s in surfaces)}")
        if workarounds:
            block.append(f"Workarounds: {'; '.join(str(w) for w in workarounds)}")
        if matched:
            block.append(
                f"Matched roadmap: {matched} "
                f"(sim={metrics.get('best_similarity')}, state={metrics.get('matched_item_state')})"
            )
        block.append("Evidence:")
        for ev in g.get("evidence") or []:
            eid = ev.get("evidence_id") or ""
            st = ev.get("source_type") or ""
            title = (ev.get("title") or "")[:120]
            snippet = (ev.get("snippet") or "")[:MAX_SNIPPET]
            url = ev.get("url") or ""
            payload = ev.get("payload") or {}
            rid = payload.get("review_id") or ""
            extra = f" review_id={rid}" if rid else ""
            block.append(
                f"  - [{st}] id={eid}{extra} title={title!r} snippet={snippet!r} url={url}"
            )
        lines.extend(block)
        lines.append("")

    pack = "\n".join(lines).strip()
    if len(pack) <= max_chars:
        return pack
    # Prefer keeping earlier (higher-ranked) gaps
    return pack[: max_chars - 40].rstrip() + "\n\n[truncated for length]"


def _build_messages(
    pack: str,
    message: str,
    history: list[ChatTurn],
) -> list[dict[str, str]]:
    system = (
        "You are dagr's evidence analyst. Answer ONLY using the EVIDENCE PACK. "
        "Do not invent needs, reviews, or roadmap items. If the pack is insufficient, "
        "say so clearly. Prefer what matters most: high-confidence gaps with dense "
        "review evidence. Reply with JSON only matching: "
        '{"answer": string, "citations": [{"gap_rank": number|null, '
        '"evidence_id": string|null, "quote": string}]}'
    )
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    msgs.append(
        {
            "role": "user",
            "content": f"EVIDENCE PACK:\n{pack}\n\n(End of pack. Wait for the question.)",
        }
    )
    msgs.append(
        {
            "role": "assistant",
            "content": '{"answer": "Ready. Ask about this job\'s gaps and evidence.", "citations": []}',
        }
    )
    for turn in history[-MAX_HISTORY_TURNS:]:
        msgs.append({"role": turn.role, "content": turn.content[:4000]})
    msgs.append({"role": "user", "content": message.strip()[:4000]})
    return msgs


def _parse_llm_json(raw: str) -> LlmChatJson:
    text = (raw or "").strip()
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # salvage outermost object
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    return LlmChatJson.model_validate(data)


async def answer_job_chat(
    job: dict[str, Any],
    message: str,
    history: list[ChatTurn] | None = None,
    settings: Settings | None = None,
) -> ChatReply:
    settings = settings or get_settings()
    if not settings.llm_enabled:
        raise RuntimeError(
            "LLM is not configured (set OPENROUTER_API_KEY / Autorouter key)"
        )
    if (job.get("status") or "") != "completed":
        raise ValueError("job is not completed")
    msg = (message or "").strip()
    if not msg:
        raise ValueError("message is empty")

    pack = build_evidence_pack(job)
    turns = list(history or [])
    messages = _build_messages(pack, msg, turns)

    url = (settings.llm_base_url or DEFAULT_LLM_URL).strip()
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "dagr-chat",
    }
    body: dict[str, Any] = {
        "model": settings.openrouter_model,
        "temperature": 0.2,
        "messages": messages,
    }
    if "openrouter.ai" in url:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
        content = r.json()["choices"][0]["message"]["content"]

    try:
        parsed = _parse_llm_json(content)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.warning("chat JSON parse failed: %s", e)
        # Retry once with parse error feedback
        retry_msgs = messages + [
            {
                "role": "assistant",
                "content": content[:2000],
            },
            {
                "role": "user",
                "content": (
                    f"Your previous JSON was invalid ({e}). "
                    "Return ONLY valid JSON with keys answer and citations."
                ),
            },
        ]
        async with httpx.AsyncClient(timeout=60.0) as client:
            r2 = await client.post(
                url, headers=headers, json={**body, "messages": retry_msgs}
            )
            if r2.status_code != 200:
                raise RuntimeError(f"LLM retry HTTP {r2.status_code}: {r2.text[:300]}")
            content2 = r2.json()["choices"][0]["message"]["content"]
        parsed = _parse_llm_json(content2)

    # Drop citations that don't reference known evidence ids / ranks
    known_ranks = {int(g.get("rank")) for g in (job.get("gaps") or []) if g.get("rank") is not None}
    known_eids = {
        str(ev.get("evidence_id"))
        for g in (job.get("gaps") or [])
        for ev in (g.get("evidence") or [])
        if ev.get("evidence_id")
    }
    citations: list[ChatCitation] = []
    for c in parsed.citations:
        if c.gap_rank is not None and c.gap_rank not in known_ranks:
            c = ChatCitation(
                gap_rank=None, evidence_id=c.evidence_id, quote=c.quote
            )
        if c.evidence_id and c.evidence_id not in known_eids:
            c = ChatCitation(
                gap_rank=c.gap_rank, evidence_id=None, quote=c.quote
            )
        if c.quote or c.gap_rank is not None or c.evidence_id:
            citations.append(c)

    return ChatReply(
        answer=parsed.answer.strip(),
        citations=citations,
        model=settings.openrouter_model,
    )
