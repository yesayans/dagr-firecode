"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { getJob, startAnalyze } from "@/lib/api";
import type { Job } from "@/lib/types";
import { GapCard } from "@/components/GapCard";
import { RoadmapSourceBadge } from "@/components/RoadmapSourceBadge";
import { StagePipeline } from "@/components/StagePipeline";
import { UnverifiedNotice } from "@/components/UnverifiedNotice";
import { formatReviewWindow } from "@/lib/dates";

const POLL_MS = 1500;

export default function JobPage() {
  const params = useParams<{ id: string }>();
  const jobId = params.id;

  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);

  const load = useCallback(async () => {
    try {
      const next = await getJob(jobId);
      setJob(next);
      setError(null);
      return next;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load job");
      return null;
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function tick() {
      const next = await load();
      if (cancelled || !next) return;
      if (next.status !== "completed" && next.status !== "failed") {
        timer = setTimeout(tick, POLL_MS);
      }
    }

    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [load]);

  async function handleRetry() {
    if (!job) return;
    setRetrying(true);
    setError(null);
    try {
      const { job_id } = await startAnalyze({
        app_id: job.app.id,
        max_reviews: 2000,
        force: true,
      });
      window.location.href = `/jobs/${job_id}`;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retry failed");
      setRetrying(false);
    }
  }

  if (loading && !job) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-16 sm:px-10">
        <p className="font-mono text-sm text-zinc-500 animate-pulse-soft">
          Loading job…
        </p>
      </div>
    );
  }

  if (error && !job) {
    return (
      <div className="mx-auto max-w-5xl space-y-4 px-6 py-16 sm:px-10">
        <p className="text-red-300">{error}</p>
        <Link href="/" className="text-teal-300 hover:text-teal-200">
          ← Back to search
        </Link>
      </div>
    );
  }

  if (!job) return null;

  const isNone = job.roadmap_source === "none";
  const running =
    job.status === "queued" || job.status === "running";
  const failed = job.status === "failed";
  const completed = job.status === "completed";
  const reviewWindow = formatReviewWindow(
    job.stats.review_window_start,
    job.stats.review_window_end,
  );
  const degraded = Array.isArray(job.stats.degraded) ? job.stats.degraded : [];

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-10 sm:py-14">
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <Link
          href="/"
          className="text-sm text-zinc-500 transition hover:text-zinc-300"
        >
          ← New analysis
        </Link>
        <p className="font-mono text-xs text-zinc-600">{job.id}</p>
      </div>

      <header className="animate-fade-up space-y-4">
        <p className="font-mono text-xs uppercase tracking-[0.22em] text-teal-400/90">
          dagr · job
        </p>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">
              {job.app.display_name}
            </h1>
            <p className="mt-1 font-mono text-sm text-zinc-500">
              {job.app.package_name}
            </p>
            <p className="mt-3 font-mono text-sm text-zinc-400">
              {reviewWindow
                ? `reviews: ${reviewWindow}`
                : "reviews: window unknown"}
              {" · "}
              roadmap: live
              {job.stats.review_provenance
                ? ` · source ${job.stats.review_provenance}`
                : ""}
            </p>
          </div>
          <RoadmapSourceBadge source={job.roadmap_source} />
        </div>
      </header>

      {running && (
        <section className="animate-fade-up mt-10 rounded-xl border border-white/10 bg-zinc-900/60 p-6 sm:p-8">
          <StagePipeline stage={job.stage} progress={job.progress} />
        </section>
      )}

      {failed && (
        <section
          role="alert"
          className="mt-10 space-y-4 rounded-xl border border-red-500/40 bg-red-500/10 p-6 sm:p-8"
        >
          <h2 className="text-xl font-semibold text-red-200">
            Analysis failed
          </h2>
          <p className="text-sm leading-relaxed text-red-100/90">
            {job.error ?? error ?? "Unknown error"}
          </p>
          <button
            type="button"
            onClick={handleRetry}
            disabled={retrying}
            className="rounded-lg bg-white px-5 py-2.5 text-sm font-semibold text-zinc-900 disabled:opacity-60"
          >
            {retrying ? "Retrying…" : "Retry analysis"}
          </button>
        </section>
      )}

      {completed && (
        <div className="mt-10 space-y-8 animate-fade-up">
          {job.summary && (
            <p className="max-w-3xl text-lg leading-relaxed text-zinc-300">
              {job.summary}
            </p>
          )}

          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              ["Reviews", job.stats.total_reviews],
              ["Clusters", job.stats.clusters],
              ["Roadmap items", job.stats.roadmap_items],
              ["Elapsed", `${job.stats.elapsed_s}s`],
            ].map(([label, value]) => (
              <div
                key={String(label)}
                className="rounded-lg border border-white/8 bg-zinc-900/50 px-4 py-3"
              >
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-500">
                  {label}
                </dt>
                <dd className="mt-1 font-mono text-xl tabular-nums text-white">
                  {typeof value === "number" ? value.toLocaleString() : value}
                </dd>
              </div>
            ))}
          </dl>

          {degraded.length > 0 && (
            <div
              role="status"
              className="rounded-lg border-2 border-amber-400/50 bg-amber-500/15 px-5 py-4"
            >
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-amber-200">
                Degraded run — not full fidelity
              </p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-amber-50/90">
                {degraded.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>
          )}

          {isNone && <UnverifiedNotice />}

          <div className="space-y-2">
            <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
              {isNone
                ? "Surfaced Needs (no public roadmap found to verify against)"
                : "Gap ranking — unmet & under-served needs"}
            </h2>
            <p className="text-sm text-zinc-500">
              {job.gaps.length} gaps · embedding {job.stats.embedding_backend}
              {job.stats.llm_used
                ? " · LLM extract on"
                : " · deterministic extract (no LLM)"}
            </p>
          </div>

          {job.gaps.length === 0 ? (
            <div className="rounded-xl border border-dashed border-white/12 px-5 py-10 text-center text-zinc-500">
              No gaps emitted for this run.
            </div>
          ) : (
            <div className="space-y-5">
              {job.gaps.map((gap, i) => (
                <GapCard
                  key={gap.id}
                  gap={gap}
                  roadmapSource={job.roadmap_source}
                  defaultExpanded={i === 0}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {error && job && (
        <p className="mt-6 text-sm text-amber-300">Poll warning: {error}</p>
      )}
    </div>
  );
}
