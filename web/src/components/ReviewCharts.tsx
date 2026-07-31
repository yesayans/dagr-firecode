"use client";

import type { Gap, JobCharts } from "@/lib/types";
import { useI18n } from "@/lib/i18n";

interface Props {
  charts?: JobCharts | null;
  gaps: Gap[];
  reviewsNeedBearing?: number;
  totalReviews?: number;
}

function VerticalBars({
  items,
  emptyLabel,
  accent = "var(--accent)",
}: {
  items: { label: string; value: number; hint?: string }[];
  emptyLabel: string;
  accent?: string;
}) {
  const max = Math.max(1, ...items.map((i) => i.value));
  if (items.length === 0 || items.every((i) => i.value === 0)) {
    return (
      <p className="py-8 text-center text-sm text-[var(--muted)]">{emptyLabel}</p>
    );
  }
  return (
    <div className="flex h-44 items-end gap-1.5 sm:gap-2.5">
      {items.map((item) => {
        const h = Math.max(6, Math.round((item.value / max) * 100));
        return (
          <div
            key={item.label}
            className="flex min-w-0 flex-1 flex-col items-center justify-end gap-1.5"
            title={item.hint ?? `${item.label}: ${item.value}`}
          >
            <span className="font-mono text-[11px] font-semibold tabular-nums text-[var(--foreground)]">
              {item.value > 0 ? item.value.toLocaleString() : ""}
            </span>
            <div
              className="w-full max-w-12 rounded-t-md"
              style={{
                height: `${h}%`,
                background: accent,
                minHeight: item.value > 0 ? 8 : 4,
              }}
            />
            <span className="w-full truncate text-center font-mono text-[11px] font-medium text-[var(--muted)]">
              {item.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function StarDistribution({
  items,
  emptyLabel,
}: {
  items: { stars: number; count: number }[];
  emptyLabel: string;
}) {
  const total = items.reduce((a, r) => a + r.count, 0);
  const max = Math.max(1, ...items.map((i) => i.count));
  if (total === 0) {
    return (
      <p className="py-6 text-center text-sm text-[var(--muted)]">{emptyLabel}</p>
    );
  }

  return (
    <div className="space-y-3">
      {[5, 4, 3, 2, 1].map((stars) => {
        const row = items.find((r) => r.stars === stars);
        const count = row?.count ?? 0;
        const pct = total > 0 ? (count / total) * 100 : 0;
        const barPct = (count / max) * 100;
        return (
          <div key={stars} className="grid grid-cols-[4.5rem_1fr_4.5rem] items-center gap-3">
            <span
              className="font-mono text-sm font-semibold tabular-nums"
              style={{ color: "var(--star)" }}
              aria-label={`${stars} stars`}
            >
              {"★".repeat(stars)}
              <span className="text-[var(--muted)]">
                {"☆".repeat(5 - stars)}
              </span>
            </span>
            <div
              className="h-3.5 overflow-hidden rounded-full"
              style={{ background: "var(--chart-track)" }}
            >
              <div
                className="h-full rounded-full transition-[width] duration-500"
                style={{
                  width: `${Math.max(count > 0 ? 3 : 0, barPct)}%`,
                  background:
                    stars <= 2
                      ? "color-mix(in srgb, #dc2626 80%, var(--accent))"
                      : stars === 3
                        ? "color-mix(in srgb, #d97706 70%, var(--accent))"
                        : "var(--accent)",
                }}
              />
            </div>
            <div className="text-right font-mono text-xs tabular-nums text-[var(--foreground)]">
              <span className="font-semibold">{count.toLocaleString()}</span>
              <span className="ml-1 text-[var(--muted)]">
                {pct.toFixed(0)}%
              </span>
            </div>
          </div>
        );
      })}
      <p className="pt-1 text-right font-mono text-[11px] text-[var(--muted)]">
        n = {total.toLocaleString()}
      </p>
    </div>
  );
}

function fromGaps(
  gaps: Gap[],
  reviewsNeedBearing?: number,
  totalReviews?: number,
): JobCharts {
  const ratingBuckets = [0, 0, 0, 0, 0];
  for (const g of gaps) {
    const r = Math.round(g.metrics.mean_rating || 0);
    if (r >= 1 && r <= 5) ratingBuckets[r - 1] += g.metrics.cluster_size || 1;
  }
  const need = reviewsNeedBearing ?? 0;
  const total = totalReviews ?? need;
  return {
    period: "year",
    reviews_by_period: [],
    rating_histogram: ratingBuckets.map((count, i) => ({
      stars: i + 1,
      count,
    })),
    need_bearing: {
      need_bearing: need,
      other: Math.max(0, total - need),
    },
  };
}

export function ReviewCharts({
  charts,
  gaps,
  reviewsNeedBearing,
  totalReviews,
}: Props) {
  const { t } = useI18n();
  const data =
    charts &&
    ((charts.reviews_by_period?.length ?? 0) > 0 ||
      (charts.rating_histogram?.some((r) => r.count > 0) ?? false))
      ? charts
      : fromGaps(gaps, reviewsNeedBearing, totalReviews);

  // Always show year buckets (backend period is "year").
  const timeItems = (data.reviews_by_period || []).map((p) => {
    const year = p.period.slice(0, 4);
    return {
      label: year,
      value: p.count,
      hint: `${year}: ${p.count.toLocaleString()} reviews`,
    };
  });

  const ratingRows = data.rating_histogram || [];
  const need = data.need_bearing?.need_bearing ?? 0;
  const other = data.need_bearing?.other ?? 0;
  const needTotal = Math.max(1, need + other);
  const needPct = Math.round((need / needTotal) * 100);

  return (
    <section className="space-y-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-[var(--shadow)] sm:p-6">
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-[var(--foreground)]">
          {t("chartsTitle")}
        </h2>
        <p className="mt-1 text-sm text-[var(--muted)]">{t("chartsSubtitle")}</p>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-muted)]/70 p-4 sm:p-5">
          <h3 className="text-xs font-bold uppercase tracking-[0.14em] text-[var(--muted)]">
            {t("reviewsOverTime")}
          </h3>
          <div className="mt-4">
            <VerticalBars items={timeItems} emptyLabel={t("noChartData")} />
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-muted)]/70 p-4 sm:p-5">
          <h3 className="text-xs font-bold uppercase tracking-[0.14em] text-[var(--muted)]">
            {t("ratingMix")}
          </h3>
          <div className="mt-4">
            <StarDistribution
              items={ratingRows}
              emptyLabel={t("noChartData")}
            />
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-[var(--border)] bg-[var(--surface-muted)]/70 p-4 sm:p-5">
        <h3 className="text-xs font-bold uppercase tracking-[0.14em] text-[var(--muted)]">
          {t("needBearing")}
        </h3>
        <div className="mt-4 space-y-3">
          <div
            className="flex h-4 overflow-hidden rounded-full"
            style={{ background: "var(--chart-track)" }}
          >
            <div
              className="bg-[var(--accent)]"
              style={{ width: `${needPct}%` }}
              title={`${t("needBearingLabel")}: ${need}`}
            />
            <div
              className="bg-[color-mix(in_srgb,var(--muted)_45%,transparent)]"
              style={{ width: `${100 - needPct}%` }}
              title={`${t("otherReviews")}: ${other}`}
            />
          </div>
          <div className="flex flex-wrap justify-between gap-3 font-mono text-sm text-[var(--foreground)]">
            <span>
              <span className="font-semibold text-[var(--accent)]">
                {t("needBearingLabel")}
              </span>
              : {need.toLocaleString()} ({needPct}%)
            </span>
            <span>
              <span className="font-semibold">{t("otherReviews")}</span>:{" "}
              {other.toLocaleString()} ({100 - needPct}%)
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
