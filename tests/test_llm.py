"""LLM layer: JSON extraction, schema validation, repair retry, citation guards."""

from __future__ import annotations

import numpy as np
import pytest

from aipm.llm.client import NullLlmClient
from aipm.llm.guards import backfill_evidence, validate_citations
from aipm.llm.prompts.cluster_need import ClusterInsight, build_messages
from aipm.llm.structured import (
    StructuredOutputError,
    extract_json_object,
    generate_structured,
    parse_json_response,
    schema_hint,
    strip_code_fences,
)
from aipm.schemas import Review
from tests.conftest import StubLlmClient, insight_json


class TestJsonExtraction:
    def test_plain_json(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self):
        assert parse_json_response('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_around_the_object(self):
        text = 'Here is the analysis:\n{"a": 1}\nHope that helps!'
        assert parse_json_response(text) == {"a": 1}

    def test_braces_inside_quoted_strings(self):
        """Review quotes contain braces; a regex-based extractor breaks here."""
        assert parse_json_response('{"quote": "he said {weird} things"}')["quote"] == (
            "he said {weird} things"
        )

    def test_escaped_quotes(self):
        assert parse_json_response('{"q": "she said \\"no\\""}')["q"] == 'she said "no"'

    def test_nested_objects(self):
        assert extract_json_object('{"a": {"b": {"c": 1}}}') == '{"a": {"b": {"c": 1}}}'

    def test_empty_response_raises(self):
        with pytest.raises(StructuredOutputError):
            parse_json_response("")

    def test_malformed_json_raises(self):
        with pytest.raises(StructuredOutputError):
            parse_json_response("{not json at all")

    def test_array_rejected(self):
        with pytest.raises(StructuredOutputError, match="object"):
            parse_json_response("[1, 2, 3]")

    def test_strip_fences_without_fences(self):
        assert strip_code_fences("  bare  ") == "bare"


class TestClusterInsightValidation:
    def test_evidence_strength_from_percentage_scale(self):
        assert ClusterInsight.model_validate(
            {**_minimal(), "evidence_strength": 80}
        ).evidence_strength == 0.8

    def test_evidence_strength_from_word(self):
        assert ClusterInsight.model_validate(
            {**_minimal(), "evidence_strength": "high"}
        ).evidence_strength == 0.8

    def test_evidence_strength_garbage_defaults(self):
        assert ClusterInsight.model_validate(
            {**_minimal(), "evidence_strength": "no idea"}
        ).evidence_strength == 0.5

    def test_evidence_strength_clamped(self):
        assert ClusterInsight.model_validate(
            {**_minimal(), "evidence_strength": -3}
        ).evidence_strength == 0.0

    def test_category_alias_mapped(self):
        assert ClusterInsight.model_validate(
            {**_minimal(), "category": "bugs"}
        ).category.value == "reliability"

    def test_unknown_category_rejected_not_silently_wrong(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClusterInsight.model_validate({**_minimal(), "category": "quantum"})

    def test_empty_required_field_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ClusterInsight.model_validate({**_minimal(), "hidden_need": "   "})


class TestGenerateStructured:
    def test_valid_first_response(self):
        client = StubLlmClient([insight_json()])
        insight, usage = generate_structured(client, ClusterInsight, _messages())
        assert insight.category.value == "reliability"
        assert usage.n_calls == 1

    def test_repairs_after_one_invalid_response(self):
        """The repair retry is what turns most malformed output into usable data."""
        client = StubLlmClient(["not json at all", insight_json()])
        insight, usage = generate_structured(client, ClusterInsight, _messages())
        assert insight.hidden_need
        assert usage.n_calls == 2

    def test_repair_prompt_includes_the_validation_error(self):
        client = StubLlmClient(["{}", insight_json()])
        generate_structured(client, ClusterInsight, _messages())
        repair = client.calls[1][-1]["content"]
        assert "could not be parsed" in repair and "Required shape" in repair

    def test_raises_when_repair_also_fails(self):
        client = StubLlmClient(["garbage", "still garbage"])
        with pytest.raises(StructuredOutputError):
            generate_structured(client, ClusterInsight, _messages())

    def test_schema_hint_lists_every_field(self):
        hint = schema_hint(ClusterInsight)
        assert all(name in hint for name in ClusterInsight.model_fields)


class TestNullClient:
    def test_reports_unavailable_with_a_reason(self):
        client = NullLlmClient("no key")
        assert not client.available
        assert client.healthcheck() == (False, "no key")


class TestCitationGuard:
    def _reviews(self) -> dict[str, Review]:
        return {
            f"r{i}": Review(review_id=f"r{i}", app_id="a1", text=f"review text {i}",
                            score=2, helpful_count=i)
            for i in range(4)
        }

    def test_hallucinated_id_dropped(self):
        audit = validate_citations(
            ["r0", "does-not-exist"], need_statement="n",
            cluster_review_ids={"r0", "r1"}, reviews_by_id=self._reviews(),
        )
        assert audit.n_unknown_id == 1 and audit.n_validated == 1

    def test_out_of_cluster_id_dropped(self):
        """A real id pulled from elsewhere in the prompt is still fabricated."""
        audit = validate_citations(
            ["r0", "r3"], need_statement="n",
            cluster_review_ids={"r0", "r1"}, reviews_by_id=self._reviews(),
        )
        assert audit.n_out_of_cluster == 1

    def test_irrelevant_citation_dropped_under_threshold(self):
        need_vector = np.array([1.0, 0.0], dtype=np.float32)
        vectors = {
            "r0": np.array([1.0, 0.0], dtype=np.float32),   # identical
            "r1": np.array([0.0, 1.0], dtype=np.float32),   # orthogonal
        }
        audit = validate_citations(
            ["r0", "r1"], need_statement="n", cluster_review_ids={"r0", "r1"},
            reviews_by_id=self._reviews(), need_vector=need_vector,
            review_vectors=vectors, threshold=0.3,
        )
        assert audit.n_validated == 1 and audit.n_below_threshold == 1

    def test_relevance_skipped_without_vectors(self):
        audit = validate_citations(
            ["r0"], need_statement="n", cluster_review_ids={"r0"},
            reviews_by_id=self._reviews(),
        )
        assert audit.n_validated == 1

    def test_duplicate_citations_counted_once(self):
        audit = validate_citations(
            ["r0", "r0"], need_statement="n", cluster_review_ids={"r0"},
            reviews_by_id=self._reviews(),
        )
        assert audit.n_validated == 1

    def test_backfill_marks_uncited_evidence_unvalidated(self):
        """Backfilled context must not inflate the grounding score."""
        audit = validate_citations(
            ["r0"], need_statement="n", cluster_review_ids={"r0", "r1", "r2"},
            reviews_by_id=self._reviews(),
        )
        audit = backfill_evidence(
            audit, cluster_review_ids=["r0", "r1", "r2"],
            reviews_by_id=self._reviews(), target=3,
        )
        assert len(audit.evidence) == 3
        assert audit.n_validated == 1
        assert sum(1 for e in audit.evidence if not e.validated) == 2


class TestPromptConstruction:
    def test_includes_keywords_and_samples(self):
        messages = build_messages(
            app_name="Test App", app_category="Food & Drink",
            keywords=["crash", "login"],
            samples=[("r1", 1, "it crashes on login")],
            schema=schema_hint(ClusterInsight),
        )
        user = messages[1]["content"]
        assert "Test App" in user and "crash, login" in user and "r1" in user

    def test_system_prompt_forbids_inventing_numbers(self):
        system = build_messages(
            app_name="A", app_category="B", keywords=[], samples=[], schema="{}"
        )[0]["content"]
        assert "Never state a statistic" in system

    def test_long_samples_truncated(self):
        messages = build_messages(
            app_name="A", app_category="B", keywords=[],
            samples=[("r1", 1, "x" * 5000)], schema="{}", max_sample_chars=100,
        )
        assert "x" * 101 not in messages[1]["content"]


def _minimal() -> dict:
    return {
        "title": "t", "summary": "s", "surface_complaint": "c",
        "hidden_need": "Users need something", "underlying_goal": "g",
    }


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "analyse this cluster"}]


class TestTransportHeaders:
    def test_user_agent_is_overridden(self):
        """Some routers' WAFs 403 the SDK's own `OpenAI/Python...` agent."""
        from aipm.llm.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1", api_key="k",
            model="gemini-2.5-flash", user_agent="aipm/0.1.0",
        )
        assert client._client.default_headers["User-Agent"] == "aipm/0.1.0"

    def test_sdk_default_kept_when_unset(self):
        from aipm.llm.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(
            base_url="https://example.invalid/v1", api_key="k",
            model="gemini-2.5-flash", user_agent="",
        )
        assert "OpenAI" in client._client.default_headers["User-Agent"]

    def test_pricing_matches_bare_and_namespaced_ids(self):
        from aipm.llm.client import OpenAICompatibleClient

        def price(model: str) -> float:
            return OpenAICompatibleClient(
                base_url="https://example.invalid/v1", api_key="k", model=model
            )._price(1_000_000, 0)

        assert price("gemini-2.5-flash") == pytest.approx(0.30)
        assert price("google/gemini-2.5-flash") == pytest.approx(0.30)
        assert price("some-unknown-model") == 0.0
