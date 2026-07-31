"use client";

import { useState } from "react";
import type { Gap, RoadmapSource, Verdict } from "@/lib/types";
import { ConfidenceBreakdown } from "./ConfidenceBreakdown";
import { EvidenceTrace } from "./EvidenceTrace";
import {
  hasRetrospectiveFields,
  RetrospectiveValidation,
} from "./RetrospectiveValidation";

const VERDICT_STYLES: Record<Verdict, string> = {
  IGNORED: "bg-red-500/15 text-red-700 ring-red-500/40 dark:text-red-300",
  "UNDER-PRIORITIZED":
    "bg-amber-500/15 text-amber-800 ring-amber-500/40 dark:text-amber-300",
  MISUNDERSTOOD:
    "bg-violet-500/15 text-violet-800 ring-violet-500/40 dark:text-violet-300",
  UNVERIFIED:
    "bg-slate-500/15 text-slate-700 ring-slate-500/40 dark:text-slate-300",
};

export function GapCard({
  gap,
  roadmapSource,
  llmUsed = true,
  defaultExpanded = false,
}: {
  gap: Gap;
  roadmapSource: RoadmapSource;
  llmUsed?: boolean;
  defaultExpanded?: boolean;
}) {
  const [open, setOpen] = useState(defaultExpanded);
  const isNone = roadmapSource === "none";
  const showMatch =
    !isNone &&
    (gap.metrics.matched_item_title != null ||
      gap.metrics.best_similarity != null);
  const showRetrospective = hasRetrospectiveFields(gap.metrics);
  const keywords = Array.isArray(gap.metrics.keywords)
    ? gap.metrics.keywords
    : [];
  const templateGenerated = !llmUsed;

  return (
    <article className="overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-sm">
      <div className="grid gap-6 p-6 lg:grid-cols-[auto_1fr_auto] lg:items-start">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-muted)] font-mono text-xl font-semibold text-[var(--foreground)]">
          {gap.rank}
        </div>

        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex rounded-md px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ring-1 ring-inset ${VERDICT_STYLES[gap.verdict]}`}
            >
              {gap.verdict}
            </span>
            {showRetrospective && gap.metrics.validated_by_later_roadmap && (
              <span className="inline-flex rounded-md bg-emerald-500/15 px-2.5 py-1 text-xs font-semibold uppercase tracking-wide text-emerald-300 ring-1 ring-inset ring-emerald-500/40">
                Validated in hindsight
              </span>
            )}
            {keywords.slice(0, 4).map((kw) => (
              <span
                key={kw}
                className="rounded-md bg-[var(--surface-muted)] px-2 py-1 text-xs text-[var(--muted)]"
              >
                {kw}
              </span>
            ))}
          </div>

          <div>
            <h3
              className={`text-xl font-semibold tracking-tight sm:text-2xl ${
                templateGenerated
                  ? "text-[var(--muted)]"
                  : "text-[var(--foreground)]"
              }`}
            >
              {gap.need}
            </h3>
            {templateGenerated && (
              <p className="mt-1.5 inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-2 py-1 font-mono text-[11px] font-medium uppercase tracking-wide text-[var(--muted)]">
                <span
                  className="h-1.5 w-1.5 rounded-sm bg-[var(--muted)]"
                  aria-hidden
                />
                template-generated (no LLM configured)
              </p>
            )}
          </div>
          <p className="max-w-3xl text-base leading-relaxed text-[var(--muted)]">
            {gap.one_sentence_summary}
          </p>

          {showMatch && (
            <div className="rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-4 py-3">
              <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                {gap.metrics.matched_item_title
                  ? "Matched roadmap item"
                  : "Roadmap comparison"}
              </p>
              <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                {gap.metrics.matched_item_title && gap.metrics.matched_item_url ? (
                  <a
                    href={gap.metrics.matched_item_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-[var(--accent)] hover:opacity-80"
                  >
                    {gap.metrics.matched_item_title} ↗
                  </a>
                ) : gap.metrics.matched_item_title ? (
                  <span className="text-sm text-[var(--foreground)]">
                    {gap.metrics.matched_item_title}
                  </span>
                ) : (
                  <span className="text-sm text-[var(--muted)]">
                    No contemporaneous roadmap match
                  </span>
                )}
                {gap.metrics.best_similarity != null && (
                  <span className="font-mono text-xs text-[var(--muted)]">
                    nearest similarity{" "}
                    {Number(gap.metrics.best_similarity).toFixed(2)}
                    {gap.metrics.matched_item_state
                      ? ` · ${gap.metrics.matched_item_state}`
                      : ""}
                  </span>
                )}
              </div>
            </div>
          )}

          {showRetrospective && (
            <RetrospectiveValidation metrics={gap.metrics} />
          )}
        </div>

        <div className="text-left lg:text-right">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
            Confidence
          </p>
          <p className="mt-1 font-mono text-5xl font-semibold leading-none tabular-nums text-[var(--foreground)] sm:text-6xl">
            {Math.round(gap.confidence)}
            <span className="text-2xl text-[var(--muted)]">%</span>
          </p>
          {gap.metrics.llm_confidence == null && (
            <p className="mt-2 font-mono text-[11px] uppercase tracking-wide text-[var(--muted)]">
              deterministic
            </p>
          )}
        </div>
      </div>

      <div className="border-t border-[var(--border)] px-6 py-3">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-3 text-left text-sm font-medium text-[var(--muted)] transition hover:text-[var(--foreground)]"
          aria-expanded={open}
        >
          <span>
            {open ? "Hide" : "Show"} evidence & confidence breakdown
          </span>
          <span className="font-mono text-[var(--muted)]">{open ? "−" : "+"}</span>
        </button>
      </div>

      {open && (
        <div className="space-y-8 border-t border-[var(--border)] bg-[var(--surface-muted)]/50 px-6 py-6">
          <ConfidenceBreakdown
            metrics={gap.metrics}
            confidence={gap.confidence}
          />
          <div>
            <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
              Evidence trace
            </h4>
            <EvidenceTrace evidence={gap.evidence} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                Confidence rationale
              </h4>
              <p className="text-sm leading-relaxed text-[var(--muted)]">
                {gap.confidence_rationale}
              </p>
            </div>
            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                Latent reasoning
              </h4>
              <p className="text-sm leading-relaxed text-[var(--muted)]">
                {gap.latent_reasoning}
              </p>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}
