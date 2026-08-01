"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { getJob, postJobTranslate, startAnalyze } from "@/lib/api";
import { applyJobTranslation } from "@/lib/applyTranslation";
import type { Job, TranslateResponse } from "@/lib/types";
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
  const { t, locale } = useI18n();

  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);
  const [translation, setTranslation] = useState<TranslateResponse | null>(
    null,
  );
  const [translating, setTranslating] = useState(false);
  const [translateError, setTranslateError] = useState<string | null>(null);

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
    // Locale switch invalidates a previous translation target.
    setTranslation(null);
    setTranslateError(null);
  }, [locale, jobId]);

  const displayJob = useMemo(
    () => (job ? applyJobTranslation(job, translation) : null),
    [job, translation],
  );

  async function handleTranslate() {
    if (!job || job.status !== "completed") return;
    if (translation && translation.locale === locale) {
      setTranslation(null);
      setTranslateError(null);
      return;
    }
    if (locale === "en") {
      setTranslation(null);
      return;
    }
    setTranslating(true);
    setTranslateError(null);
    try {
      const next = await postJobTranslate(job.id, locale);
      setTranslation(next);
    } catch (err) {
      setTranslateError(
        err instanceof Error ? err.message : t("translateError"),
      );
    } finally {
      setTranslating(false);
    }
  }

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

  if (!job || !displayJob) return null;

  const isNone = displayJob.roadmap_source === "none";
  const running = displayJob.status === "queued" || displayJob.status === "running";
  const failed = displayJob.status === "failed";
  const completed = displayJob.status === "completed";
  const reviewWindow = formatReviewWindow(
    displayJob.stats.review_window_start,
    displayJob.stats.review_window_end,
  );
  const degraded = Array.isArray(displayJob.stats.degraded)
    ? displayJob.stats.degraded
    : [];
  const translationActive =
    !!translation && translation.locale === locale && locale !== "en";

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-10 sm:py-14">
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <Link
          href="/"
          className="text-sm text-[var(--muted)] transition hover:text-[var(--foreground)]"
        >
          {t("newAnalysis")}
        </Link>
        <p className="font-mono text-xs text-[var(--muted)]">{displayJob.id}</p>
      </div>

      <header className="animate-fade-up space-y-4">
        <p className="font-mono text-xs uppercase tracking-[0.22em] text-[var(--accent)]">
          {t("jobLabel")}
        </p>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-4xl font-semibold tracking-tight text-[var(--foreground)] sm:text-5xl">
              {displayJob.app.display_name}
            </h1>
            <p className="mt-1 font-mono text-sm text-[var(--muted)]">
              {displayJob.app.package_name}
            </p>
            <p className="mt-3 font-mono text-sm text-[var(--muted)]">
              {reviewWindow
                ? `reviews: ${reviewWindow}`
                : "reviews: window unknown"}
              {" · "}
              roadmap: live
              {displayJob.stats.review_provenance
                ? ` · source ${displayJob.stats.review_provenance}`
                : ""}
            </p>
          </div>
          <RoadmapSourceBadge source={displayJob.roadmap_source} />
        </div>
      </header>

      {running && (
        <section className="animate-fade-up mt-10 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 sm:p-8">
          <StagePipeline
            stage={displayJob.stage}
            progress={displayJob.progress}
          />
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
            {displayJob.error ?? error ?? "Unknown error"}
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
          {displayJob.summary && (
            <p className="max-w-3xl text-lg leading-relaxed text-[var(--muted)]">
              {displayJob.summary}
            </p>
          )}

          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              [t("reviews"), displayJob.stats.total_reviews],
              [t("clusters"), displayJob.stats.clusters],
              [t("roadmapItems"), displayJob.stats.roadmap_items],
              [t("elapsed"), `${displayJob.stats.elapsed_s}s`],
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
            charts={displayJob.stats.charts}
            gaps={displayJob.gaps}
            reviewsNeedBearing={displayJob.stats.reviews_need_bearing}
            totalReviews={
              displayJob.stats.reviews_total ?? displayJob.stats.total_reviews
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

          <div className="space-y-3">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div className="space-y-2">
                <h2 className="text-2xl font-semibold tracking-tight text-[var(--foreground)] sm:text-3xl">
                  {isNone ? t("gapTitleNone") : t("gapTitle")}
                </h2>
                <p className="text-sm text-[var(--muted)]">
                  {displayJob.gaps.length} gaps · embedding{" "}
                  {displayJob.stats.embedding_backend}
                  {displayJob.stats.llm_used
                    ? " · LLM extract on"
                    : " · deterministic extract (no LLM)"}
                </p>
              </div>
              {locale !== "en" && (
                <div className="flex max-w-md flex-col items-end gap-2">
                  <button
                    type="button"
                    onClick={handleTranslate}
                    disabled={translating}
                    className="inline-flex items-center gap-2 rounded-xl border border-[var(--border-strong)] bg-[var(--surface)] px-4 py-2.5 text-sm font-semibold text-[var(--foreground)] shadow-sm transition hover:border-[var(--accent)] hover:text-[var(--accent)] disabled:opacity-60"
                  >
                    <span
                      className="inline-flex h-5 w-5 items-center justify-center rounded-md bg-[var(--surface-muted)] font-mono text-[10px]"
                      aria-hidden
                    >
                      AI
                    </span>
                    {translating
                      ? t("translatingAnalysis")
                      : translationActive
                        ? t("showOriginalAnalysis")
                        : t("translateAnalysis")}
                  </button>
                  {!translationActive && (
                    <p className="text-right text-xs leading-snug text-[var(--muted)]">
                      {t("translateHint")}
                    </p>
                  )}
                  {translationActive && (
                    <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-[var(--accent)]">
                      {t("translatedBadge")}
                      {translation?.model ? ` · ${translation.model}` : ""}
                    </p>
                  )}
                  {translateError && (
                    <p className="text-right text-xs text-red-600 dark:text-red-300">
                      {t("translateError")}: {translateError}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>

          {displayJob.gaps.length === 0 ? (
            <div className="rounded-xl border border-dashed border-[var(--border)] px-5 py-10 text-center text-[var(--muted)]">
              {t("noGaps")}
            </div>
          ) : (
            <div className="space-y-5">
              {displayJob.gaps.map((gap, i) => (
                <GapCard
                  key={gap.id}
                  gap={gap}
                  roadmapSource={displayJob.roadmap_source}
                  llmUsed={displayJob.stats.llm_used}
                  defaultExpanded={i === 0}
                />
              ))}
            </div>
          )}

          <EvidenceChat jobId={displayJob.id} />
        </div>
      )}

      {error && displayJob && (
        <p className="mt-6 text-sm text-amber-700 dark:text-amber-300">
          Poll warning: {error}
        </p>
      )}
    </div>
  );
}
