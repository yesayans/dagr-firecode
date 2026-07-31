"""Latent-need extraction via OpenRouter, with honest no-LLM fallback."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from src.config import Settings, get_settings
from src.gap_analyzer import CandidateGap
from src.need_filter import is_need_bearing

logger = logging.getLogger(__name__)

DEFAULT_LLM_URL = "https://openrouter.ai/api/v1/chat/completions"

VERDICTS_ROADMAP = {"IGNORED", "UNDER-PRIORITIZED", "MISUNDERSTOOD"}
VERDICTS_NONE = {"UNVERIFIED"}

NeedSource = Literal["llm", "representative_review"]


class LlmGapResponse(BaseModel):
    latent_need: str
    verdict: str
    confidence: float = Field(ge=0, le=100)
    confidence_justification: str
    one_sentence_summary: str
    cited_review_ids: list[str]


class ExtractedGap(BaseModel):
    latent_need: str
    verdict: str
    confidence: float
    confidence_rationale: str
    one_sentence_summary: str
    latent_reasoning: str
    cited_review_ids: list[str]
    llm_used: bool
    need_source: NeedSource
    llm_confidence: float | None
    metrics: dict[str, Any]
    review_ids: list[str]
    matched_item: dict[str, Any] | None
    keywords: list[str]
    representative_review_id: str | None = None


class LatentNeedExtractor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.sem = asyncio.Semaphore(4)
        self.degraded: list[str] = []

    @property
    def llm_enabled(self) -> bool:
        return self.settings.llm_enabled

    async def extract_all(
        self,
        candidates: list[CandidateGap],
        reviews_by_id: dict[str, dict[str, Any]],
        roadmap_source: str,
        top_n: int = 5,
    ) -> list[ExtractedGap]:
        selected = candidates[: max(top_n, 5)]
        if not self.llm_enabled:
            self.degraded.append(
                "no OPENROUTER_API_KEY; quoting representative reviews"
            )
            return [
                self._representative_extract(c, reviews_by_id, roadmap_source)
                for c in selected
            ]

        tasks = [
            self._extract_one(c, reviews_by_id, roadmap_source) for c in selected
        ]
        return list(await asyncio.gather(*tasks))

    def extract_all_sync(
        self,
        candidates: list[CandidateGap],
        reviews_by_id: dict[str, dict[str, Any]],
        roadmap_source: str,
        top_n: int = 5,
    ) -> list[ExtractedGap]:
        return asyncio.run(
            self.extract_all(candidates, reviews_by_id, roadmap_source, top_n=top_n)
        )

    async def _extract_one(
        self,
        candidate: CandidateGap,
        reviews_by_id: dict[str, dict[str, Any]],
        roadmap_source: str,
    ) -> ExtractedGap:
        provided_ids = list(candidate.review_ids)[:12]
        sample = [
            {
                "review_id": rid,
                "rating": (reviews_by_id.get(rid) or {}).get("rating"),
                "text": (reviews_by_id.get(rid) or {}).get("review_text", "")[:400],
            }
            for rid in provided_ids
            if rid in reviews_by_id
        ]
        # When lexical/semantic roadmap matching is disabled, never ask the LLM
        # for IGNORED / UNDER-PRIORITIZED / MISUNDERSTOOD — those require a
        # defensible match we do not currently have.
        matching_on = bool(self.settings.roadmap_matching_enabled)
        effective_mode = (
            "none"
            if (not matching_on or roadmap_source == "none")
            else roadmap_source
        )
        allowed = VERDICTS_NONE if effective_mode == "none" else VERDICTS_ROADMAP
        matched = None
        if matching_on and candidate.matched_item:
            matched = {
                "title": (candidate.matched_item.get("text") or "")[:200],
                "state": candidate.matched_item.get("state"),
                "url": candidate.matched_item.get("url"),
            }

        prompt = _build_prompt(
            sample=sample,
            matched=matched,
            preliminary_verdict=(
                "UNVERIFIED" if effective_mode == "none" else candidate.verdict
            ),
            roadmap_mode=effective_mode,
            keywords=candidate.keywords,
            allowed=sorted(allowed),
        )

        async with self.sem:
            raw, err = await self._call_openrouter(prompt)
            if err or raw is None:
                logger.warning("LLM call failed: %s", err)
                self.degraded.append(f"LLM call failed: {err}")
                return self._representative_extract(
                    candidate, reviews_by_id, roadmap_source
                )

            parsed, parse_err = _parse_response(raw)
            if parse_err:
                retry_prompt = (
                    prompt
                    + f"\n\nYour previous JSON was invalid ({parse_err}). "
                    "Return ONLY valid JSON matching the schema."
                )
                raw2, err2 = await self._call_openrouter(retry_prompt)
                if err2 or raw2 is None:
                    return self._representative_extract(
                        candidate, reviews_by_id, roadmap_source
                    )
                parsed, parse_err2 = _parse_response(raw2)
                if parse_err2 or parsed is None:
                    return self._representative_extract(
                        candidate, reviews_by_id, roadmap_source
                    )

        assert parsed is not None
        cited = [c for c in parsed.cited_review_ids if c in set(provided_ids)]
        if not cited:
            cited = _fallback_citations(candidate, reviews_by_id, provided_ids)

        verdict = parsed.verdict.strip().upper()
        if effective_mode == "none":
            verdict = "UNVERIFIED"
        elif verdict not in allowed:
            verdict = candidate.verdict

        det = float(candidate.metrics["deterministic_confidence"])
        llm_conf = float(parsed.confidence)
        blended = round(0.6 * det + 0.4 * llm_conf, 2)
        rationale = (
            f"0.6 * deterministic ({det}) + 0.4 * llm ({llm_conf}). "
            f"{parsed.confidence_justification}"
        )
        if effective_mode == "none":
            reason = (
                "roadmap matching disabled (null model); "
                if not matching_on
                else ""
            )
            rationale = (
                f"0.6 * deterministic ({det}) + 0.4 * llm ({llm_conf}); "
                f"UNVERIFIED — {reason}justification uses user-side evidence only. "
                f"{parsed.confidence_justification}"
            )

        metrics = dict(candidate.metrics)
        metrics["llm_confidence"] = llm_conf
        metrics["deterministic_confidence"] = det

        return ExtractedGap(
            latent_need=parsed.latent_need.strip(),
            verdict=verdict,
            confidence=blended,
            confidence_rationale=rationale,
            one_sentence_summary=parsed.one_sentence_summary.strip(),
            latent_reasoning=parsed.confidence_justification.strip(),
            cited_review_ids=cited,
            llm_used=True,
            need_source="llm",
            llm_confidence=llm_conf,
            metrics=metrics,
            review_ids=list(candidate.review_ids),
            matched_item=candidate.matched_item,
            keywords=candidate.keywords,
            representative_review_id=None,
        )

    def _representative_extract(
        self,
        candidate: CandidateGap,
        reviews_by_id: dict[str, dict[str, Any]],
        roadmap_source: str,
    ) -> ExtractedGap:
        """
        No LLM: do not synthesise a need statement. Quote the most representative
        need-bearing review (centroid-nearest) verbatim.
        """
        rid, quote = _pick_representative_review(candidate, reviews_by_id)
        keywords = candidate.keywords or []
        theme = ", ".join(keywords[:6]) if keywords else "(no keywords)"

        det = float(candidate.metrics["deterministic_confidence"])
        matching_on = bool(self.settings.roadmap_matching_enabled)
        verdict = (
            "UNVERIFIED"
            if (not matching_on or roadmap_source == "none")
            else candidate.verdict
        )
        if roadmap_source == "none":
            rationale = (
                f"deterministic only ({det}); no LLM — quoting representative "
                f"need-bearing review {rid}; cluster size={candidate.cluster_size}, "
                f"cohesion={candidate.cohesion:.2f}."
            )
        else:
            rationale = (
                f"deterministic only ({det}); no LLM — quoting representative "
                f"need-bearing review {rid}."
            )

        summary = (
            f"Supporting themes: {theme}. "
            f"Quoted review {rid} (closest need-bearing member to cluster centroid)."
        )
        cited = [rid] if rid else _fallback_citations(
            candidate, reviews_by_id, list(candidate.review_ids)[:8]
        )
        metrics = dict(candidate.metrics)
        metrics["llm_confidence"] = None

        return ExtractedGap(
            latent_need=quote,
            verdict=verdict,
            confidence=det,
            confidence_rationale=rationale,
            one_sentence_summary=summary[:320],
            latent_reasoning=(
                f"need_source=representative_review; review_id={rid}; "
                f"keywords=[{theme}]. Not an inferred latent goal — a real user sentence."
            ),
            cited_review_ids=cited,
            llm_used=False,
            need_source="representative_review",
            llm_confidence=None,
            metrics=metrics,
            review_ids=list(candidate.review_ids),
            matched_item=candidate.matched_item,
            keywords=candidate.keywords,
            representative_review_id=rid,
        )

    async def _call_openrouter(self, prompt: str) -> tuple[str | None, str | None]:
        url = (self.settings.llm_base_url or DEFAULT_LLM_URL).strip()
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "dagr",
        }
        body: dict[str, Any] = {
            "model": self.settings.openrouter_model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract LATENT unmet user needs from app reviews. "
                        "A latent need is the unspoken underlying goal, NOT a summary "
                        "of surface complaints. Reply with JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        # OpenRouter supports response_format; some Autorouter/Gemini proxies do not.
        if "openrouter.ai" in url:
            body["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(url, headers=headers, json=body)
                if r.status_code != 200:
                    return None, f"HTTP {r.status_code}: {r.text[:300]}"
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                return content, None
        except Exception as e:
            return None, str(e)


def _pick_representative_review(
    candidate: CandidateGap,
    reviews_by_id: dict[str, dict[str, Any]],
) -> tuple[str | None, str]:
    """Prefer centroid-nearest text when it is need-bearing; else first need-bearing."""
    rep_text = (candidate.representative_text or "").strip()
    # Match representative_text back to a review id
    if rep_text:
        for rid in candidate.review_ids:
            rev = reviews_by_id.get(rid) or {}
            text = str(rev.get("review_text") or "").strip()
            if text == rep_text or text.startswith(rep_text[:80]):
                if is_need_bearing(text, rev.get("rating")):
                    return rid, text
                break

    for rid in candidate.review_ids:
        rev = reviews_by_id.get(rid) or {}
        text = str(rev.get("review_text") or "").strip()
        if text and is_need_bearing(text, rev.get("rating")):
            return rid, text

    # Last resort: any member text (should be rare — clusters are need-bearing-only)
    for rid in candidate.review_ids:
        rev = reviews_by_id.get(rid) or {}
        text = str(rev.get("review_text") or "").strip()
        if text:
            return rid, text
    return None, rep_text or "(no review text)"


def _build_prompt(
    *,
    sample: list[dict[str, Any]],
    matched: dict[str, Any] | None,
    preliminary_verdict: str,
    roadmap_mode: str,
    keywords: list[str],
    allowed: list[str],
) -> str:
    return (
        "Extract the LATENT NEED (unspoken underlying goal) from these reviews.\n"
        f"Roadmap mode: {roadmap_mode}\n"
        f"Preliminary verdict: {preliminary_verdict}\n"
        f"Allowed verdicts: {allowed}\n"
        f"Cluster keywords: {keywords}\n"
        f"Closest roadmap item (or null): {json.dumps(matched)}\n"
        f"Reviews (with ids):\n{json.dumps(sample, ensure_ascii=False)}\n\n"
        "Return JSON with keys: latent_need, verdict, confidence (0-100), "
        "confidence_justification, one_sentence_summary, cited_review_ids.\n"
        "cited_review_ids MUST be a subset of the provided review ids.\n"
        f"In mode 'none' you MUST use verdict UNVERIFIED and justify only from "
        "user-side evidence density."
    )


def _parse_response(raw: str) -> tuple[LlmGapResponse | None, str | None]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError as e2:
                return None, str(e2)
        else:
            return None, str(e)
    try:
        return LlmGapResponse.model_validate(data), None
    except ValidationError as e:
        return None, str(e)


def validate_citations(
    cited: list[str], provided_ids: list[str]
) -> list[str]:
    """Drop hallucinated review ids. Public helper for tests."""
    allowed = set(provided_ids)
    return [c for c in cited if c in allowed]


def _fallback_citations(
    candidate: CandidateGap,
    reviews_by_id: dict[str, dict[str, Any]],
    provided_ids: list[str],
) -> list[str]:
    out = [rid for rid in provided_ids if rid in reviews_by_id][:3]
    if out:
        return out
    return list(candidate.review_ids)[:3]


def main() -> None:
    settings = get_settings()
    ext = LatentNeedExtractor(settings)
    cand = CandidateGap(
        cluster_id=0,
        review_ids=["r1", "r2", "r3", "r4", "r5"],
        verdict="UNVERIFIED",
        best_similarity=None,
        matched_item=None,
        metrics={
            "deterministic_confidence": 72.5,
            "components": {},
            "weights": {},
            "cluster_size": 5,
            "total_reviews": 40,
            "cluster_share": 0.125,
            "best_similarity": None,
            "matched_item_title": None,
            "matched_item_url": None,
            "matched_item_state": None,
            "matched_item_age_days": None,
            "mean_rating": 2.0,
            "rating_spread": 0.4,
            "cohesion": 0.7,
            "llm_confidence": None,
            "keywords": ["stream", "cache", "playback"],
        },
        keywords=["stream", "cache", "playback"],
        representative_text=(
            "Lack of a persistent stream cache leads to a potentially long pause "
            "when resuming playback on slow networks."
        ),
        cohesion=0.7,
        mean_rating=4.0,
        cluster_size=5,
    )
    quote = cand.representative_text
    reviews = {
        "r1": {"review_id": "r1", "review_text": quote, "rating": 4},
        "r2": {"review_id": "r2", "review_text": "wish for faster cache", "rating": 4},
        "r3": {"review_id": "r3", "review_text": "network pause on resume", "rating": 3},
        "r4": {"review_id": "r4", "review_text": "stream buffer missing", "rating": 4},
        "r5": {"review_id": "r5", "review_text": "slow networks hurt playback", "rating": 3},
    }
    assert validate_citations(["r1", "hallucinated"], ["r1", "r2"]) == ["r1"]
    out = ext._representative_extract(cand, reviews, "none")
    assert out.need_source == "representative_review"
    assert out.llm_used is False
    assert out.latent_need == quote
    print(
        {
            "llm_enabled": ext.llm_enabled,
            "need_source": out.need_source,
            "need": out.latent_need[:80],
            "cited": out.cited_review_ids,
        }
    )


if __name__ == "__main__":
    main()
