import type { GapMetrics } from "@/lib/types";
import { formatMonthYear, formatReviewWindow } from "@/lib/dates";

/** Runtime guard — fields may be absent while the backend is mid-rollout. */
export function hasRetrospectiveFields(
  metrics: GapMetrics,
): metrics is GapMetrics & { validated_by_later_roadmap: boolean } {
  return typeof metrics.validated_by_later_roadmap === "boolean";
}

export function RetrospectiveValidation({ metrics }: { metrics: GapMetrics }) {
  if (!hasRetrospectiveFields(metrics)) return null;

  const windowLabel = formatReviewWindow(
    metrics.review_window_start,
    metrics.review_window_end,
  );
  const later = metrics.later_addressed_by;
  const laterDate = later ? formatMonthYear(later.date) : null;

  if (metrics.validated_by_later_roadmap && later) {
    return (
      <div className="rounded-lg border border-emerald-400/45 bg-emerald-500/10 px-4 py-3.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-400/20 px-2.5 py-1 text-xs font-bold uppercase tracking-wide text-emerald-200 ring-1 ring-inset ring-emerald-400/40">
            <span className="h-1.5 w-1.5 rounded-sm bg-emerald-300" aria-hidden />
            Validated in hindsight
          </span>
          {windowLabel && (
            <span className="font-mono text-[11px] text-emerald-200/70">
              review window {windowLabel}
            </span>
          )}
        </div>
        <p className="mt-2 text-sm leading-relaxed text-emerald-50/90">
          Surfaced from historical reviews alone — the team later shipped a
          matching roadmap item. Corroboration only; does not change the
          verdict computed at the review window.
        </p>
        <div className="mt-3 rounded-md border border-emerald-400/25 bg-black/25 px-3 py-2.5">
          {later.url ? (
            <a
              href={later.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-emerald-200 hover:text-emerald-100"
            >
              {later.title} ↗
            </a>
          ) : (
            <p className="text-sm font-medium text-emerald-100">{later.title}</p>
          )}
          <p className="mt-1 font-mono text-xs text-emerald-200/65">
            {[
              laterDate,
              later.state,
              `similarity ${later.similarity.toFixed(2)}`,
            ]
              .filter(Boolean)
              .join(" · ")}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-rose-400/35 bg-rose-500/[0.08] px-4 py-3.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-md bg-rose-400/15 px-2.5 py-1 text-xs font-bold uppercase tracking-wide text-rose-200 ring-1 ring-inset ring-rose-400/35">
          <span className="h-1.5 w-1.5 rounded-sm bg-rose-300" aria-hidden />
          Still unaddressed
        </span>
        {windowLabel && (
          <span className="font-mono text-[11px] text-rose-200/70">
            since review window {windowLabel}
          </span>
        )}
      </div>
      <p className="mt-2 text-sm leading-relaxed text-rose-50/85">
        No later roadmap item matches this need after the review window. The
        latent demand has stood without a documented response — retrospective
        check only; verdict unchanged.
      </p>
    </div>
  );
}
