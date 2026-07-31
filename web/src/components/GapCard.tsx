"use client";

import { useState } from "react";
import type { Gap, RoadmapSource, Verdict } from "@/lib/types";
import { ConfidenceBreakdown } from "./ConfidenceBreakdown";
import { EvidenceTrace } from "./EvidenceTrace";

const VERDICT_STYLES: Record<Verdict, string> = {
  IGNORED: "bg-red-500/15 text-red-300 ring-red-500/40",
  "UNDER-PRIORITIZED": "bg-amber-500/15 text-amber-300 ring-amber-500/40",
  MISUNDERSTOOD: "bg-purple-500/15 text-purple-300 ring-purple-500/40",
  UNVERIFIED: "bg-slate-500/20 text-slate-300 ring-slate-500/40",
};

export function GapCard({
  gap,
  roadmapSource,
  defaultExpanded = false,
}: {
  gap: Gap;
  roadmapSource: RoadmapSource;
  defaultExpanded?: boolean;
}) {
  const [open, setOpen] = useState(defaultExpanded);
  const isNone = roadmapSource === "none";
  const showMatch =
    !isNone &&
    (gap.metrics.matched_item_title !== null ||
      gap.metrics.best_similarity !== null);

  return (
    <article className="overflow-hidden rounded-xl border border-white/10 bg-zinc-900/70 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
      <div className="grid gap-6 p-6 lg:grid-cols-[auto_1fr_auto] lg:items-start">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-white/5 font-mono text-xl font-semibold text-zinc-300">
          {gap.rank}
        </div>

        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex rounded-md px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ring-1 ring-inset ${VERDICT_STYLES[gap.verdict]}`}
            >
              {gap.verdict}
            </span>
            {gap.metrics.keywords.slice(0, 4).map((kw) => (
              <span
                key={kw}
                className="rounded-md bg-white/5 px-2 py-1 text-xs text-zinc-400"
              >
                {kw}
              </span>
            ))}
          </div>

          <h3 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">
            {gap.need}
          </h3>
          <p className="max-w-3xl text-base leading-relaxed text-zinc-400">
            {gap.one_sentence_summary}
          </p>

          {showMatch && (
            <div className="rounded-md border border-white/8 bg-black/25 px-4 py-3">
              <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-500">
                Matched roadmap item
              </p>
              <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                {gap.metrics.matched_item_url ? (
                  <a
                    href={gap.metrics.matched_item_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-teal-300 hover:text-teal-200"
                  >
                    {gap.metrics.matched_item_title ?? "Open item"} ↗
                  </a>
                ) : (
                  <span className="text-sm text-zinc-200">
                    {gap.metrics.matched_item_title ?? "Unmatched"}
                  </span>
                )}
                {gap.metrics.best_similarity !== null && (
                  <span className="font-mono text-xs text-zinc-500">
                    similarity {gap.metrics.best_similarity.toFixed(2)}
                    {gap.metrics.matched_item_state
                      ? ` · ${gap.metrics.matched_item_state}`
                      : ""}
                  </span>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="text-left lg:text-right">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-500">
            Confidence
          </p>
          <p className="mt-1 font-mono text-5xl font-semibold leading-none tabular-nums text-white sm:text-6xl">
            {Math.round(gap.confidence)}
            <span className="text-2xl text-zinc-500">%</span>
          </p>
        </div>
      </div>

      <div className="border-t border-white/8 px-6 py-3">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-3 text-left text-sm font-medium text-zinc-300 transition hover:text-white"
          aria-expanded={open}
        >
          <span>
            {open ? "Hide" : "Show"} evidence & confidence breakdown
          </span>
          <span className="font-mono text-zinc-500">{open ? "−" : "+"}</span>
        </button>
      </div>

      {open && (
        <div className="space-y-8 border-t border-white/8 bg-black/20 px-6 py-6">
          <ConfidenceBreakdown
            metrics={gap.metrics}
            confidence={gap.confidence}
          />
          <div>
            <h4 className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-zinc-500">
              Evidence trace
            </h4>
            <EvidenceTrace evidence={gap.evidence} />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-[0.14em] text-zinc-500">
                Confidence rationale
              </h4>
              <p className="text-sm leading-relaxed text-zinc-400">
                {gap.confidence_rationale}
              </p>
            </div>
            <div>
              <h4 className="mb-1 text-xs font-semibold uppercase tracking-[0.14em] text-zinc-500">
                Latent reasoning
              </h4>
              <p className="text-sm leading-relaxed text-zinc-400">
                {gap.latent_reasoning}
              </p>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}
