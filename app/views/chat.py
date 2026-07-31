"""AI Chat - grounded question answering over one app's reviews.

Scoped to the selected application and to what its run actually contains. The
agent retrieves before it answers and is instructed to refuse rather than reach
beyond the retrieved context, so "I can't tell from these reviews" is a correct
answer and not a bug.

Every answer carries the reviews it drew on, expandable inline.
"""

from __future__ import annotations

import sys
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aipm.chat.agent import SUGGESTED_QUESTIONS, ChatAgent  # noqa: E402
from aipm.schemas import ChatMessage  # noqa: E402
from app.components.tiles import kicker, pill, pill_row, rule  # noqa: E402
from app.state import (  # noqa: E402
    append_chat, chat_history, get_llm_client, get_retriever, load_reviews,
    require_selection, reset_chat,
)
from app.theme import active_palette  # noqa: E402

palette = active_palette()


def render_sidebar(result, n_indexed: int) -> None:
    with st.sidebar:
        st.caption("Chat is scoped to this application only.")
        st.markdown("**What I can see**")
        st.caption(f"{result.run.n_reviews:,} analysed reviews")
        st.caption(f"{result.run.n_clusters} themes · {len(result.needs)} needs")
        st.caption(f"{n_indexed:,} retrievable passages")
        rule()
        st.markdown("**How answers are built**")
        st.caption(
            "Your question retrieves passages by keyword *and* meaning; only "
            "those reach the model. Figures come from the precomputed run — the "
            "model is instructed never to calculate."
        )
        rule()
        if st.button("Clear conversation", width="stretch"):
            reset_chat()
            st.rerun()


def render_citations(citations) -> None:
    if not citations:
        return
    verified = [c for c in citations if c.validated]
    label = (
        f"Sources — {len(verified)} cited, {len(citations) - len(verified)} retrieved"
        if verified else f"Retrieved context — {len(citations)} reviews"
    )
    with st.expander(label):
        for item in citations[:10]:
            mark = "✓ cited in the answer" if item.validated else "◦ retrieved"
            color = palette.good if item.validated else palette.ink_muted
            stars = f"{item.review_score}★" if item.review_score else "—"
            date = item.review_date.isoformat() if item.review_date else "undated"
            st.markdown(
                f'<div class="aipm-quote" style="border-left-color:{color}">'
                f"{escape(item.quote[:400])}"
                f'<div class="aipm-quote__meta">'
                f"<span style='color:{color}'>{mark}</span> · {stars} · {date} · "
                f"{item.helpful_count} helpful · <code>{escape(item.review_id)}</code>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


def ask(agent: ChatAgent, question: str) -> None:
    append_chat(ChatMessage(role="user", content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning…"):
            answer = agent.answer(question, history=chat_history()[:-1])
        st.markdown(answer.content)
        if answer.error:
            st.caption(f"🚫 {answer.error}")
        render_citations(answer.citations)
        if answer.usage.n_calls:
            st.caption(
                f"{answer.usage.total_tokens:,} tokens · ${answer.usage.cost_usd:.4f}"
            )

    append_chat(
        ChatMessage(
            role="assistant", content=answer.content,
            citations=answer.citations, created_at=datetime.utcnow(),
        )
    )


def main() -> None:
    result = require_selection("The chat", route="AI_Chat")
    if result is None:
        return

    retriever = get_retriever(result.app.app_id, result.run.run_id)
    render_sidebar(result, len(retriever.index))

    st.title("Ask about this app")
    st.markdown(
        f'<div class="aipm-muted" style="margin-top:-0.35rem">'
        f"Answers are grounded in {result.run.n_reviews:,} analysed reviews of "
        f"<strong>{escape(result.app.name)}</strong>. If the reviews do not "
        f"contain the answer, I will say so.</div>",
        unsafe_allow_html=True,
    )

    client = get_llm_client()
    if not client.available:
        st.warning(
            "No language model is configured, so I can retrieve the most relevant "
            "reviews but not summarise them. Set `LLM_BASE_URL` and `LLM_API_KEY` "
            "to enable full answers.",
            icon="⚠️",
        )

    if len(retriever.index) == 0:
        st.error("Nothing is indexed for this run.", icon="🚫")
        return

    reviews = {r.review_id: r for r in load_reviews(result.app.app_id, result.run.run_id)}
    agent = ChatAgent(client, retriever, result=result, reviews_by_id=reviews)

    history = chat_history()

    if not history:
        rule()
        kicker("Try one of these")
        columns = st.columns(len(SUGGESTED_QUESTIONS))
        for column, question in zip(columns, SUGGESTED_QUESTIONS, strict=False):
            with column:
                if st.button(question, width="stretch", key=f"sug_{question}"):
                    ask(agent, question)
                    st.rerun()

        rule()
        kicker("Themes I can talk about")
        pill_row([
            pill(c.label[:44], "aipm-pill--accent")
            for c in sorted(result.clusters, key=lambda c: c.size, reverse=True)[:8]
            if c.label
        ])
    else:
        for message in history:
            with st.chat_message(message.role):
                st.markdown(message.content)
                if message.role == "assistant":
                    render_citations(message.citations)

    if question := st.chat_input(f"Ask about {result.app.name}…"):
        ask(agent, question)
        st.rerun()


main()
