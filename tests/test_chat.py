"""Chat: BM25, hybrid retrieval and the grounded agent."""

from __future__ import annotations

from datetime import date

import pytest

from aipm.chat.agent import ChatAgent, format_context, format_run_facts
from aipm.chat.retriever import BM25, Retriever, build_index, tokenize
from aipm.schemas import (
    AnalysisResult,
    AnalysisRun,
    App,
    Cluster,
    ConfidenceBreakdown,
    Need,
    OverviewStats,
    PriorityScore,
    Review,
    RunStatus,
)
from tests.conftest import StubLlmClient


@pytest.fixture
def chat_reviews() -> list[Review]:
    texts = [
        ("the app crashes whenever I try to add a new card", 1, 12),
        ("payment failed at the register and I had to use my physical card", 1, 30),
        ("face id never works so I type the pin every single time", 2, 4),
        ("love the design, tapping to pay is instant and reliable", 5, 1),
        ("refund took three weeks and support never replied to me", 1, 8),
    ]
    return [
        Review(review_id=f"r{i}", app_id="app1", text=t, score=s,
               review_date=date(2025, 1, i + 1), helpful_count=h)
        for i, (t, s, h) in enumerate(texts)
    ]


@pytest.fixture
def chat_result(app: App) -> AnalysisResult:
    return AnalysisResult(
        run=AnalysisRun(run_id="run1", app_id="app1", params_hash="h",
                        status=RunStatus.COMPLETE, n_reviews=5, n_clusters=2),
        app=app,
        stats=OverviewStats(n_reviews=5, avg_score=2.0, store_score=4.2,
                            score_distribution={1: 3, 2: 1, 3: 0, 4: 0, 5: 1}),
        clusters=[
            Cluster(cluster_id="c1", run_id="run1", size=3, label="Payment failures",
                    summary="Cards decline at the register", keywords=["payment", "card"]),
        ],
        needs=[
            Need(need_id="n1", run_id="run1",
                 statement="Users need payment to succeed at the point of sale",
                 underlying_goal="pay without carrying a physical wallet",
                 workarounds=["I had to use my physical card"],
                 confidence=ConfidenceBreakdown(total=0.8),
                 priority=PriorityScore(reach=0.4, impact=1.5)),
        ],
    )


class TestBM25:
    def test_ranks_the_matching_document_first(self):
        docs = ["the app crashes on login", "delivery driver could not find my address",
                "refund never arrived"]
        scores = BM25(docs).scores("refund")
        assert int(scores.argmax()) == 2

    def test_unknown_term_scores_nothing(self):
        assert BM25(["alpha beta"]).scores("zeta").max() == 0.0

    def test_empty_corpus(self):
        assert len(BM25([]).scores("anything")) == 0

    def test_tokenizer_lowercases_and_strips_punctuation(self):
        assert tokenize("Crashes, badly!") == ["crashes", "badly"]

    def test_longer_documents_are_length_normalised(self):
        """Without normalisation a padded document wins by sheer length."""
        short = "refund"
        padded = "refund " + " ".join(f"filler{i}" for i in range(200))
        scores = BM25([short, padded]).scores("refund")
        assert scores[0] > scores[1]


class TestBuildIndex:
    def test_indexes_reviews_clusters_and_needs(self, chat_result, chat_reviews):
        index = build_index(chat_result, chat_reviews)
        kinds = {e.kind for e in index.entries}
        assert kinds == {"review", "cluster", "need"}
        assert len(index) == len(chat_reviews) + 1 + 1

    def test_respects_the_review_cap(self, chat_result, chat_reviews):
        index = build_index(chat_result, chat_reviews, max_reviews=2)
        assert sum(1 for e in index.entries if e.kind == "review") == 2

    def test_cap_keeps_the_most_helpful_reviews(self, chat_result, chat_reviews):
        index = build_index(chat_result, chat_reviews, max_reviews=1)
        kept = next(e for e in index.entries if e.kind == "review")
        assert kept.review_id == "r1"  # 30 helpful votes, the highest

    def test_works_without_embeddings(self, chat_result, chat_reviews):
        index = build_index(chat_result, chat_reviews, embed_texts=None)
        assert index.vectors is None and index.bm25 is not None

    def test_embedding_failure_degrades_to_lexical(self, chat_result, chat_reviews):
        def explode(_texts):
            raise RuntimeError("model unavailable")

        index = build_index(chat_result, chat_reviews, embed_texts=explode)
        assert index.vectors is None
        assert Retriever(index).retrieve("payment")


class TestRetriever:
    def test_finds_the_relevant_review(self, chat_result, chat_reviews):
        chunks = Retriever(build_index(chat_result, chat_reviews)).retrieve("refund", k=3)
        assert any("refund" in c.text for c in chunks)

    def test_respects_k(self, chat_result, chat_reviews):
        assert len(Retriever(build_index(chat_result, chat_reviews)).retrieve("app", k=2)) <= 2

    def test_kind_filter(self, chat_result, chat_reviews):
        chunks = Retriever(build_index(chat_result, chat_reviews)).retrieve(
            "payment", k=5, kinds=["need"]
        )
        assert all(c.kind == "need" for c in chunks)

    def test_empty_query_returns_nothing(self, chat_result, chat_reviews):
        assert Retriever(build_index(chat_result, chat_reviews)).retrieve("   ") == []

    def test_empty_index_returns_nothing(self):
        from aipm.chat.retriever import RetrievalIndex

        assert Retriever(RetrievalIndex()).retrieve("anything") == []


class TestRunFacts:
    def test_states_precomputed_figures(self, chat_result):
        facts = format_run_facts(chat_result)
        assert "reviews analysed: 5" in facts
        assert "store rating: 4.2" in facts

    def test_discloses_the_quota_caveat(self, chat_result):
        chat_result.stats.sample_is_quota_capped = True
        assert "quota-capped" in format_run_facts(chat_result)


class TestChatAgent:
    def _agent(self, chat_result, chat_reviews, responses) -> ChatAgent:
        index = build_index(chat_result, chat_reviews)
        return ChatAgent(
            StubLlmClient(responses), Retriever(index), result=chat_result,
            reviews_by_id={r.review_id: r for r in chat_reviews},
        )

    def test_answers_from_retrieved_context(self, chat_result, chat_reviews):
        agent = self._agent(chat_result, chat_reviews,
                            ["Users fall back to physical cards [r1]."])
        # Shares vocabulary with the corpus: this index is lexical-only, so a
        # purely conceptual phrasing ("what workarounds…") would correctly find
        # nothing. Semantic retrieval is what makes that phrasing work in the app.
        answer = agent.answer("physical card payment failed")
        assert "physical cards" in answer.content
        assert not answer.refused

    def test_prompt_carries_the_context_and_the_facts(self, chat_result, chat_reviews):
        agent = self._agent(chat_result, chat_reviews, ["ok"])
        agent.answer("refund")
        prompt = agent.client.calls[0][-1]["content"]
        assert "RUN FACTS" in prompt and "CONTEXT" in prompt

    def test_system_prompt_forbids_calculating(self, chat_result, chat_reviews):
        agent = self._agent(chat_result, chat_reviews, ["ok"])
        agent.answer("refund")
        system = agent.client.calls[0][0]["content"]
        assert "Do NOT compute statistics" in system

    def test_cited_reviews_are_marked_and_sorted_first(self, chat_result, chat_reviews):
        agent = self._agent(chat_result, chat_reviews, ["See [r4] for the refund case."])
        answer = agent.answer("refund")
        assert answer.citations[0].review_id == "r4"
        assert answer.citations[0].validated

    def test_refuses_when_nothing_retrieves(self, chat_result, chat_reviews):
        agent = self._agent(chat_result, chat_reviews, ["should not be called"])
        answer = agent.answer("zzzz_nonexistent_term_qqq")
        assert answer.refused
        assert not agent.client.calls  # the model is never consulted

    def test_no_llm_still_returns_the_evidence(self, chat_result, chat_reviews):
        from aipm.llm.client import NullLlmClient

        agent = ChatAgent(
            NullLlmClient("no key"), Retriever(build_index(chat_result, chat_reviews)),
            result=chat_result,
            reviews_by_id={r.review_id: r for r in chat_reviews},
        )
        answer = agent.answer("refund")
        assert answer.refused and answer.citations

    def test_transport_failure_is_reported_not_raised(self, chat_result, chat_reviews):
        from aipm.llm.client import LlmError, LlmResponse

        class Failing(StubLlmClient):
            def complete(self, messages, **kwargs) -> LlmResponse:
                raise LlmError("endpoint down")

        agent = ChatAgent(
            Failing([]), Retriever(build_index(chat_result, chat_reviews)),
            result=chat_result,
            reviews_by_id={r.review_id: r for r in chat_reviews},
        )
        answer = agent.answer("refund")
        assert answer.error and answer.citations

    def test_blank_question(self, chat_result, chat_reviews):
        assert self._agent(chat_result, chat_reviews, []).answer("  ").content

    def test_context_labels_each_chunk_kind(self, chat_result, chat_reviews):
        chunks = Retriever(build_index(chat_result, chat_reviews)).retrieve("payment", k=5)
        rendered = format_context(chunks)
        assert any(tag in rendered for tag in ("REVIEW", "THEME", "NEED"))
