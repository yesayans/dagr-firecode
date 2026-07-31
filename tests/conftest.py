"""Shared fixtures.

Everything here is offline and deterministic: the `fixture` embedding backend and
a stub LLM, so the whole pipeline is exercised in milliseconds with no network.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from aipm.embeddings.cache import NullEmbeddingCache
from aipm.embeddings.provider import FixtureEmbeddingProvider
from aipm.embeddings.store import EmbeddingService
from aipm.llm.client import LlmClient, LlmResponse, Usage
from aipm.schemas import App, Review


class StubLlmClient(LlmClient):
    """Returns canned responses in order. Records the prompts it was given."""

    def __init__(self, responses: list[str], *, model: str = "stub-model") -> None:
        self._responses = list(responses)
        self._model = model
        self.calls: list[list[dict[str, str]]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LlmResponse:
        self.calls.append(messages)
        text = self._responses.pop(0) if self._responses else "{}"
        return LlmResponse(
            text=text,
            usage=Usage(prompt_tokens=10, completion_tokens=20, n_calls=1),
            model=self._model,
        )


def insight_json(**overrides) -> str:
    payload = {
        "title": "Orders silently fail",
        "summary": "Users submit an order and get no confirmation.",
        "surface_complaint": "The order never went through",
        "workaround": "I re-order from the website instead",
        "hidden_need": "Users need certainty that a submitted order was actually received",
        "underlying_goal": "place an order once and trust it landed",
        "category": "reliability",
        "evidence_strength": 0.8,
        "confidence_rationale": "Several users describe the same failure and the same workaround.",
        "cited_review_ids": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def app() -> App:
    return App(
        app_id="app1",
        name="Test App",
        score=4.2,
        categories=["Food & Drink"],
        ratings_count=1000,
    )


@pytest.fixture
def reviews() -> list[Review]:
    """A corpus with three separable themes plus praise and a duplicate."""
    base = date(2024, 1, 1)
    templates = [
        ("The app crashes every time I open my order history and I lose the cart", 1),
        ("It keeps force closing when I try to view past orders, so frustrating", 1),
        ("Constant crashes on the orders screen, had to reinstall twice", 2),
        ("Payment failed three times but my card was still charged each attempt", 1),
        ("I was charged twice for one order and support never refunded me", 1),
        ("Double charged again, I have to call the bank to reverse it myself", 2),
        ("Delivery driver could not find my address because the map is wrong", 2),
        ("The map shows the wrong location so drivers always call me confused", 3),
        ("GPS puts my house on the next street, I have to meet the driver outside", 2),
    ]
    out: list[Review] = []
    for i, (text, score) in enumerate(templates):
        for repeat in range(4):
            out.append(
                Review(
                    review_id=f"r{i}_{repeat}",
                    app_id="app1",
                    text=text,
                    score=score,
                    review_date=base + timedelta(days=30 * repeat + i),
                    helpful_count=(i * 3 + repeat) % 11,
                )
            )
    out.append(
        Review(review_id="praise1", app_id="app1", text="Great app love it", score=5,
               review_date=base, helpful_count=0)
    )
    out.append(
        Review(review_id="dupe1", app_id="app1", helpful_count=1, score=1,
               text="The app crashes every time I open my order history and I lose the cart",
               review_date=base)
    )
    return out


@pytest.fixture
def embedding_service() -> EmbeddingService:
    return EmbeddingService(FixtureEmbeddingProvider(dim=32), NullEmbeddingCache())
