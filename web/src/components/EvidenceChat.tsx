"use client";

import { useRef, useState } from "react";
import { postJobChat } from "@/lib/api";
import type { ChatCitation, ChatMessage } from "@/lib/types";
import { useI18n } from "@/lib/i18n";

interface Props {
  jobId: string;
}

function formatChatError(err: unknown): string {
  const raw = err instanceof Error ? err.message : "Chat failed";
  try {
    const parsed = JSON.parse(raw) as { detail?: unknown };
    if (typeof parsed.detail === "string") {
      if (parsed.detail === "Not Found") {
        return "Chat API not found — restart the API server so /jobs/{id}/chat is loaded.";
      }
      return parsed.detail;
    }
  } catch {
    /* plain text */
  }
  return raw;
}

export function EvidenceChat({ jobId }: Props) {
  const { t } = useI18n();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const suggestions = [
    t("suggestPrioritize"),
    t("suggestStrongest"),
    t("suggestOffline"),
  ];

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setError(null);
    setBusy(true);
    setInput("");
    const nextUser: ChatMessage = { role: "user", content: message };
    const history = [...messages, nextUser];
    setMessages(history);
    try {
      const res = await postJobChat(jobId, {
        message,
        history: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      });
      setMessages([
        ...history,
        {
          role: "assistant",
          content: res.answer,
          citations: res.citations,
        },
      ]);
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      });
    } catch (err) {
      setError(formatChatError(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-xl border border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-5 py-4 sm:px-6">
        <h2 className="text-xl font-semibold tracking-tight text-[var(--foreground)]">
          {t("chatTitle")}
        </h2>
        <p className="mt-1 text-sm text-[var(--muted)]">{t("chatSubtitle")}</p>
      </div>

      <div className="flex flex-wrap gap-2 px-5 py-3 sm:px-6">
        {suggestions.map((s) => (
          <button
            key={s}
            type="button"
            disabled={busy}
            onClick={() => send(s)}
            className="rounded-full border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-1.5 text-xs text-[var(--foreground)] transition hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>

      <div className="max-h-80 space-y-3 overflow-y-auto px-5 py-2 sm:px-6">
        {messages.length === 0 && (
          <p className="py-6 text-center text-sm text-[var(--muted)]">
            {t("chatEmpty")}
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={`${m.role}-${i}`}
            className={
              m.role === "user"
                ? "ml-8 rounded-lg border border-[color-mix(in_srgb,var(--accent)_35%,transparent)] bg-[color-mix(in_srgb,var(--accent)_12%,transparent)] px-3 py-2 text-sm text-[var(--foreground)]"
                : "mr-4 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-2 text-sm text-[var(--foreground)]"
            }
          >
            <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
            {m.role === "assistant" && m.citations && m.citations.length > 0 && (
              <ul className="mt-2 space-y-1 border-t border-[var(--border)] pt-2">
                {m.citations.map((c, j) => (
                  <CitationRow key={j} citation={c} />
                ))}
              </ul>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error && (
        <p
          role="alert"
          className="mx-5 mb-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-200 sm:mx-6"
        >
          {error}
        </p>
      )}

      <form
        className="flex gap-2 border-t border-[var(--border)] px-5 py-4 sm:px-6"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
          placeholder={t("chatPlaceholder")}
          className="min-w-0 flex-1 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-2.5 text-sm text-[var(--foreground)] outline-none focus:border-[var(--accent)] disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-lg bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50 dark:text-zinc-950"
        >
          {busy ? "…" : t("chatAsk")}
        </button>
      </form>
    </section>
  );
}

function CitationRow({ citation }: { citation: ChatCitation }) {
  const parts: string[] = [];
  if (citation.gap_rank != null) parts.push(`Gap #${citation.gap_rank}`);
  if (citation.evidence_id) parts.push(citation.evidence_id);
  return (
    <li className="font-mono text-[11px] leading-snug text-[var(--muted)]">
      <span>{parts.join(" · ") || "citation"}</span>
      {citation.quote ? (
        <span className="mt-0.5 block italic">
          “{citation.quote.slice(0, 180)}
          {citation.quote.length > 180 ? "…" : ""}”
        </span>
      ) : null}
    </li>
  );
}
