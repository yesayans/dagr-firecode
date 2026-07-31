"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { getJob, startAnalyze } from "@/lib/api";
import type { Job } from "@/lib/types";
import { EvidenceChat } from "@/components/EvidenceChat";
import { GapCard } from "@/components/GapCard";
import { ReviewCharts } from "@/components/ReviewCharts";
import { RoadmapSourceBadge } from "@/components/RoadmapSourceBadge";
import { StagePipeline } from "@/components/StagePipeline";
import { UnverifiedNotice } from "@/components/UnverifiedNotice";
import { formatReviewWindow } from "@/lib/dates";
import { useI18n } from "@/lib/i18n";

const POLL_MS = 1500;

export default function JobPage() {
  const params = useParams<{ id: string }>();
  const jobId = params.id;
  const { t } = useI18n();

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
        <p className="font-mono text-sm text-[var(--muted)] animate-pulse-soft">
          {t("loadingJob")}
        </p>
      </div>
    );
  }

  if (error && !job) {
    return (
      <div className="mx-auto max-w-5xl space-y-4 px-6 py-16 sm:px-10">
        <p className="text-red-600 dark:text-red-300">{error}</p>
        <Link href="/" className="text-[var(--accent)] hover:opacity-80">
          {t("backSearch")}
        </Link>
      </div>
    );
  }

  if (!job) return null;

  const isNone = job.roadmap_source === "none";
  const running = job.status === "queued" || job.status === "running";
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
          className="text-sm text-[var(--muted)] transition hover:text-[var(--foreground)]"
        >
          {t("newAnalysis")}
        </Link>
        <p className="font-mono text-xs text-[var(--muted)]">{job.id}</p>
      </div>

      <header className="animate-fade-up space-y-4">
        <p className="font-mono text-xs uppercase tracking-[0.22em] text-[var(--accent)]">
          {t("jobLabel")}
        </p>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight text-[var(--foreground)] sm:text-5xl">
              {job.app.display_name}
            </h1>
            <p className="mt-1 font-mono text-sm text-[var(--muted)]">
              {job.app.package_name}
            </p>
            <p className="mt-3 font-mono text-sm text-[var(--muted)]">
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
        <section className="animate-fade-up mt-10 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 sm:p-8">
          <StagePipeline stage={job.stage} progress={job.progress} />
        </section>
      )}

      {failed && (
        <section
          role="alert"
          className="mt-10 space-y-4 rounded-xl border border-red-500/40 bg-red-500/10 p-6 sm:p-8"
        >
          <h2 className="text-xl font-semibold text-red-700 dark:text-red-200">
            {t("analysisFailed")}
          </h2>
          <p className="text-sm leading-relaxed text-red-800/90 dark:text-red-100/90">
            {job.error ?? error ?? "Unknown error"}
          </p>
          <button
            type="button"
            onClick={handleRetry}
            disabled={retrying}
            className="rounded-lg bg-[var(--foreground)] px-5 py-2.5 text-sm font-semibold text-[var(--background)] disabled:opacity-60"
          >
            {retrying ? t("retrying") : t("retry")}
          </button>
        </section>
      )}

      {completed && (
        <div className="mt-10 space-y-8 animate-fade-up">
          {job.summary && (
            <p className="max-w-3xl text-lg leading-relaxed text-[var(--muted)]">
              {job.summary}
            </p>
          )}

          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              [t("reviews"), job.stats.total_reviews],
              [t("clusters"), job.stats.clusters],
              [t("roadmapItems"), job.stats.roadmap_items],
              [t("elapsed"), `${job.stats.elapsed_s}s`],
            ].map(([label, value]) => (
              <div
                key={String(label)}
                className="rounded-lg border border-[var(--border)] bg-[var(--surface)] px-4 py-3"
              >
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                  {label}
                </dt>
                <dd className="mt-1 font-mono text-xl tabular-nums text-[var(--foreground)]">
                  {typeof value === "number" ? value.toLocaleString() : value}
                </dd>
              </div>
            ))}
          </dl>

          <ReviewCharts
            charts={job.stats.charts}
            gaps={job.gaps}
            reviewsNeedBearing={job.stats.reviews_need_bearing}
            totalReviews={
              job.stats.reviews_total ?? job.stats.total_reviews
            }
          />

          {degraded.length > 0 && (
            <div
              role="status"
              className="rounded-lg border-2 border-amber-400/50 bg-amber-500/15 px-5 py-4"
            >
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-amber-800 dark:text-amber-200">
                {t("degraded")}
              </p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-amber-950/90 dark:text-amber-50/90">
                {degraded.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>
          )}

          {isNone && <UnverifiedNotice />}

          <div className="space-y-2">
            <h2 className="text-2xl font-semibold tracking-tight text-[var(--foreground)] sm:text-3xl">
              {isNone ? t("gapTitleNone") : t("gapTitle")}
            </h2>
            <p className="text-sm text-[var(--muted)]">
              {job.gaps.length} gaps · embedding {job.stats.embedding_backend}
              {job.stats.llm_used
                ? " · LLM extract on"
                : " · deterministic extract (no LLM)"}
            </p>
          </div>

          {job.gaps.length === 0 ? (
            <div className="rounded-xl border border-dashed border-[var(--border)] px-5 py-10 text-center text-[var(--muted)]">
              {t("noGaps")}
            </div>
          ) : (
            <div className="space-y-5">
              {job.gaps.map((gap, i) => (
                <GapCard
                  key={gap.id}
                  gap={gap}
                  roadmapSource={job.roadmap_source}
                  llmUsed={job.stats.llm_used}
                  defaultExpanded={i === 0}
                />
              ))}
            </div>
          )}

          <EvidenceChat jobId={job.id} />
        </div>
      )}

      {error && job && (
        <p className="mt-6 text-sm text-amber-700 dark:text-amber-300">
          Poll warning: {error}
        </p>
      )}
    </div>
  );
}
