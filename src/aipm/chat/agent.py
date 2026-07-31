"""The grounded chat agent.

Scoped to one app's analysis run. The model is given retrieved context and told
to answer from it or say it cannot - a chat box that confidently invents a
statistic would undo everything the confidence model is for.

Numbers in answers come from the retrieved run, not from the model's arithmetic:
the system prompt forbids computing, and the app-level facts the model is allowed
to state are injected pre-computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from aipm.chat.retriever import Retriever
from aipm.llm.client import LlmClient, LlmError, Usage
from aipm.schemas import AnalysisResult, ChatMessage, Evidence, RetrievedChunk, Review
from aipm.utils.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a product analyst answering questions about ONE mobile app, using only
the review evidence supplied below.

Rules:
- Answer only from the CONTEXT. If it does not contain the answer, say so plainly
  and suggest what the user could look at instead. Do not use outside knowledge
  about the app or company.
- Cite the review ids you relied on, inline, in square brackets: [r_18_ab12cd34].
  Only cite ids that appear in the CONTEXT.
- Do NOT compute statistics, counts, percentages or averages. If a figure is not
  already stated in the CONTEXT or the RUN FACTS, say it is not available.
- Be concise and concrete. A product manager is reading this between meetings.
- Quote users' own words when it makes the point better than paraphrase.\
"""

USER_TEMPLATE = """\
APP: {app_name}

RUN FACTS (already computed - quote these rather than deriving your own):
{facts}

CONTEXT:
{context}

QUESTION: {question}
"""

#: How much of each retrieved chunk reaches the prompt.
_MAX_CHUNK_CHARS = 600


@dataclass
class ChatAnswer:
    content: str
    citations: list[Evidence] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    refused: bool = False
    error: str | None = None


def format_run_facts(result: AnalysisResult) -> str:
    """Pre-computed figures the model may quote. Keeps arithmetic out of the LLM."""
    stats = result.stats
    lines = [
        f"- reviews analysed: {stats.n_reviews:,}",
        f"- store rating: {stats.store_score if stats.store_score is not None else 'unknown'}",
        f"- mean rating of the sampled reviews: {stats.avg_score:.2f}",
        f"- rating distribution (stars: count): {stats.score_distribution}",
        f"- share of 1-2 star reviews: {stats.pct_negative:.0%}",
        f"- themes found: {stats.n_clusters}",
        f"- needs extracted: {len(result.needs)}",
    ]
    if stats.date_range:
        lines.append(f"- review dates: {stats.date_range[0]} to {stats.date_range[1]}")
    if stats.sample_is_quota_capped:
        lines.append(
            "- CAVEAT: the review sample is quota-capped by the scraper, so the mean "
            "rating of the sample is not the app's real rating. Use the store rating."
        )
    for need in result.needs[:5]:
        lines.append(
            f"- need '{need.statement[:70]}': confidence {need.confidence.total:.2f} "
            f"({need.confidence.band}), reach {need.priority.reach:.1%}"
        )
    return "\n".join(lines)


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        label = {"review": "REVIEW", "cluster": "THEME", "need": "NEED"}.get(
            chunk.kind, chunk.kind.upper()
        )
        blocks.append(f"[{label} {chunk.ref_id}]\n{chunk.text[:_MAX_CHUNK_CHARS]}")
    return "\n\n".join(blocks) if blocks else "(nothing retrieved)"


class ChatAgent:
    """Answers questions about one run, with citations resolved back to reviews."""

    def __init__(
        self,
        client: LlmClient,
        retriever: Retriever,
        *,
        result: AnalysisResult,
        reviews_by_id: dict[str, Review] | None = None,
        top_k: int = 8,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.result = result
        self.reviews_by_id = reviews_by_id or {}
        self.top_k = top_k

    def answer(
        self, question: str, history: Sequence[ChatMessage] = ()
    ) -> ChatAnswer:
        if not question.strip():
            return ChatAnswer(content="Ask me something about this app's reviews.")

        chunks = self.retriever.retrieve(question, k=self.top_k)
        if not chunks:
            return ChatAnswer(
                content=(
                    "I could not find anything in this app's analysed reviews that "
                    "relates to that. Try naming a feature, a symptom, or a theme "
                    "from the Application Details page."
                ),
                refused=True,
            )

        if not self.client.available:
            return ChatAnswer(
                content=(
                    "No language model is configured, so I can only show you the "
                    "most relevant reviews rather than summarise them."
                ),
                chunks=chunks,
                citations=self._citations(chunks),
                refused=True,
            )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Prior turns give the model pronoun resolution ("why is that?") without
        # letting it treat its own past answers as evidence - context is rebuilt
        # from retrieval every turn.
        for message in list(history)[-6:]:
            messages.append({"role": message.role, "content": message.content})
        messages.append(
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    app_name=self.result.app.name,
                    facts=format_run_facts(self.result),
                    context=format_context(chunks),
                    question=question,
                ),
            }
        )

        try:
            response = self.client.complete(messages, max_tokens=900)
        except LlmError as exc:
            log.warning("chat completion failed: %s", exc)
            return ChatAnswer(
                content=(
                    "The language model could not be reached, so I cannot summarise. "
                    "The most relevant reviews are shown below."
                ),
                chunks=chunks,
                citations=self._citations(chunks),
                error=str(exc),
            )

        return ChatAnswer(
            content=response.text.strip(),
            citations=self._citations(chunks, mentioned_in=response.text),
            chunks=chunks,
            usage=response.usage,
        )

    def _citations(
        self, chunks: Sequence[RetrievedChunk], *, mentioned_in: str | None = None
    ) -> list[Evidence]:
        """Resolve retrieved reviews to Evidence for the citation expander.

        When the answer text is supplied, cited reviews are surfaced first - the
        rest stay available as supporting context.
        """
        out: list[Evidence] = []
        for chunk in chunks:
            if chunk.kind != "review" or not chunk.review_id:
                continue
            review = self.reviews_by_id.get(chunk.review_id)
            if review is None:
                continue
            was_cited = bool(mentioned_in and chunk.review_id in mentioned_in)
            out.append(
                Evidence(
                    review_id=review.review_id,
                    quote=review.text,
                    relevance=round(chunk.score, 4),
                    review_score=review.score,
                    review_date=review.review_date,
                    helpful_count=review.helpful_count,
                    validated=was_cited,
                )
            )
        out.sort(key=lambda e: (e.validated, e.relevance), reverse=True)
        return out


SUGGESTED_QUESTIONS = (
    "What are users most frustrated about?",
    "What workarounds do people describe?",
    "Which problems mention losing money?",
    "What do the 5-star reviews praise?",
    "Has anything changed recently?",
)
