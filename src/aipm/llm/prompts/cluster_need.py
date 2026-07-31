"""Cluster -> hidden need extraction.

The prompt's job is to push past the complaint to the job-to-be-done. Two devices
do most of the work:

* Requiring a **workaround** field. Users describing a hack are describing an
  unmet need, and forcing the model to look for one reliably surfaces needs that
  a summariser would have restated as "users report crashes".
* Requiring the surface complaint and the latent need as *separate* fields, so
  the model cannot pass one off as the other.

`evidence_strength` is advisory only. The confidence number the product displays
is computed in Python from cluster geometry, volume, time spread and citation
survival - never taken from the model. It is collected here purely so the
pipeline can flag disagreement between the model's read and the computed score.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from aipm.schemas import NeedCategory

SYSTEM_PROMPT = """\
You are a senior product manager analysing user reviews of a mobile app.

Your job is to find the LATENT NEED behind a group of related complaints - the
job the user is trying to get done that the product is blocking. You are not
writing a summary and you are not counting anything.

Rules:
- Distinguish the surface complaint ("the app crashes") from the underlying need
  ("users need confidence that a submitted order actually went through").
- Look hard for workarounds. If users describe going to the website, restarting,
  re-entering data, or using a competitor, that workaround IS the evidence of an
  unmet need. Quote it.
- Never invent a review. Only cite review ids that appear in the input.
- Never state a statistic, count, percentage or score in any text field. Those
  are computed elsewhere and your numbers would contradict them.
- If the reviews genuinely share no coherent theme, say so in the summary and
  give a low evidence_strength rather than inventing a need.

Reply with a single JSON object. No markdown fences, no commentary.\
"""

USER_TEMPLATE = """\
App: {app_name}
Category: {app_category}

This is one cluster of {n_members} related review segments.

Distinguishing keywords (computed with c-TF-IDF, not by you):
{keywords}

Representative review segments (id | stars | text):
{samples}

Return JSON with exactly this shape:
{schema}
"""


class ClusterInsight(BaseModel):
    """What the LLM is allowed to produce for one cluster. All text, one advisory float."""

    title: str = Field(
        description="Short human-readable cluster name, max 8 words, no numbers"
    )
    summary: str = Field(
        description="1-2 sentences describing what these users are experiencing"
    )
    surface_complaint: str = Field(
        description="What users literally complain about, in their own framing"
    )
    workaround: str = Field(
        default="",
        description="The workaround users describe, verbatim if possible. Empty if none.",
    )
    hidden_need: str = Field(
        description="The latent need, phrased as 'Users need ...'. Not a restatement "
        "of the complaint."
    )
    underlying_goal: str = Field(
        description="The job-to-be-done this need is blocking"
    )
    category: NeedCategory = Field(
        default=NeedCategory.OTHER, description="One of the allowed need categories"
    )
    evidence_strength: float = Field(
        default=0.5,
        description="0.0-1.0. Your own read of how coherent and well-supported this "
        "theme is. ADVISORY ONLY - the displayed confidence is computed separately.",
    )
    confidence_rationale: str = Field(
        default="",
        description="One sentence on why the evidence is strong or weak. Qualitative "
        "only - no counts, no percentages.",
    )
    cited_review_ids: list[str] = Field(
        default_factory=list,
        description="Review ids from the input that best evidence this need, 2-5 of them",
    )

    @field_validator("evidence_strength", mode="before")
    @classmethod
    def _coerce_strength(cls, value: object) -> float:
        """Models return "0.8", "high", or 80. Normalise, never reject."""
        if isinstance(value, str):
            text = value.strip().lower()
            named = {"very high": 0.9, "high": 0.8, "medium": 0.55, "moderate": 0.55,
                     "low": 0.3, "very low": 0.15}
            if text in named:
                return named[text]
            try:
                value = float(text.rstrip("%"))
            except ValueError:
                return 0.5
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.5
        if number > 1.0:  # model answered on a 0-100 scale
            number /= 100.0
        return min(max(number, 0.0), 1.0)

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> object:
        """Map near-miss category names onto the enum instead of failing validation."""
        if not isinstance(value, str):
            return value
        text = value.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "bug": "reliability", "bugs": "reliability", "crash": "reliability",
            "stability": "reliability", "speed": "performance", "slow": "performance",
            "ux": "usability", "ui": "usability", "design": "usability",
            "feature": "feature_gap", "missing_feature": "feature_gap",
            "privacy": "trust_privacy", "trust": "trust_privacy",
            "security": "trust_privacy", "price": "pricing", "cost": "pricing",
            "billing": "pricing", "customer_support": "support", "service": "support",
        }
        return aliases.get(text, text)

    @field_validator("title", "summary", "surface_complaint", "hidden_need", "underlying_goal")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty")
        return cleaned


def build_messages(
    *,
    app_name: str,
    app_category: str,
    keywords: list[str],
    samples: list[tuple[str, int | None, str]],
    schema: str,
    max_sample_chars: int = 400,
) -> list[dict[str, str]]:
    """Render the chat messages for one cluster.

    `samples` are `(review_id, stars, text)` triples - already reduced to
    representatives by `clustering.representatives`, never a whole cluster.
    """
    rendered = "\n".join(
        f"- {review_id} | {stars if stars is not None else '?'}* | {text[:max_sample_chars]}"
        for review_id, stars, text in samples
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                app_name=app_name,
                app_category=app_category,
                n_members=len(samples),
                keywords=", ".join(keywords) if keywords else "(none extracted)",
                samples=rendered,
                schema=schema,
            ),
        },
    ]
