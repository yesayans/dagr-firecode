import type { ComponentKey, GapMetrics } from "@/lib/types";
import { COMPONENT_KEYS } from "@/lib/types";
import { formatDecimals, formatSigFigs } from "@/lib/format";

const LABELS: Record<ComponentKey, string> = {
  volume: "Volume",
  novelty: "Novelty",
  consistency: "Consistency",
  severity: "Severity",
  spread: "Spread",
};

const SEGMENT_COLORS: Record<ComponentKey, string> = {
  volume: "bg-teal-400",
  novelty: "bg-sky-400",
  consistency: "bg-violet-400",
  severity: "bg-rose-400",
  spread: "bg-amber-400",
};

const TEXT_COLORS: Record<ComponentKey, string> = {
  volume: "text-teal-800 dark:text-teal-300",
  novelty: "text-sky-800 dark:text-sky-300",
  consistency: "text-violet-800 dark:text-violet-300",
  severity: "text-rose-800 dark:text-rose-300",
  spread: "text-amber-800 dark:text-amber-300",
};

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function contribution(metrics: GapMetrics, key: ComponentKey): number {
  return round2(100 * metrics.weights[key] * metrics.components[key]);
}

function formatRawInput(label: string, value: number | null): string {
  if (value === null) return "—";
  switch (label) {
    case "cluster_size":
      return String(Math.round(value));
    case "cluster_share":
    case "best_similarity":
      return formatSigFigs(value, 3);
    case "mean_rating":
    case "rating_spread":
    case "cohesion":
      return formatDecimals(value, 1);
    default:
      return formatSigFigs(value, 3);
  }
}

export function ConfidenceBreakdown({
  metrics,
  confidence,
}: {
  metrics: GapMetrics;
  confidence: number;
}) {
  const segments = COMPONENT_KEYS.map((key) => ({
    key,
    points: contribution(metrics, key),
    weight: metrics.weights[key],
    component: metrics.components[key],
  })).filter((s) => s.weight > 0 || s.points > 0);

  // Sum of per-segment contributions (2-dp); display at 1 decimal so judges
  // can follow the arithmetic without mistaking float noise for precision.
  const weightedSum = round1(
    segments.reduce((acc, s) => acc + s.points, 0),
  );
  const displayedConfidence = round1(confidence);

  const hasLlm = metrics.llm_confidence !== null;
  const blended = hasLlm
    ? round1(
        0.6 * metrics.deterministic_confidence +
          0.4 * (metrics.llm_confidence as number),
      )
    : null;

  return (
    <div className="space-y-5">
      <div>
        <div className="mb-2 flex items-end justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
              Confidence reconstruction
            </p>
            <p className="mt-1 text-sm text-[var(--muted)]">
              Σ (weight × component) × 100 ={" "}
              <span className="font-mono text-[var(--foreground)]">
                {formatDecimals(weightedSum, 1)}
              </span>
              {hasLlm ? " (deterministic)" : " → displayed"}
            </p>
          </div>
          <p className="font-mono text-3xl font-semibold tabular-nums text-[var(--foreground)]">
            {formatDecimals(displayedConfidence, 1)}
            <span className="text-lg text-[var(--muted)]">%</span>
          </p>
        </div>

        <div
          className="flex h-10 w-full overflow-hidden rounded-md ring-1 ring-[var(--border)]"
          role="img"
          aria-label={`Confidence breakdown totaling ${formatDecimals(weightedSum, 1)}`}
        >
          {segments.map((s) => {
            const width =
              weightedSum > 0 ? (s.points / weightedSum) * 100 : 0;
            if (width <= 0) return null;
            return (
              <div
                key={s.key}
                className={`${SEGMENT_COLORS[s.key]} relative flex items-center justify-center transition-all`}
                style={{ width: `${width}%` }}
                title={`${LABELS[s.key]}: ${formatDecimals(s.points, 1)}`}
              >
                {width >= 12 && (
                  <span className="px-1 text-[10px] font-semibold uppercase tracking-wide text-black/80">
                    {LABELS[s.key]}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <ul className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
          {segments.map((s) => (
            <li
              key={s.key}
              className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-sm ${SEGMENT_COLORS[s.key]}`}
                />
                <span
                  className={`text-xs font-semibold uppercase tracking-wide ${TEXT_COLORS[s.key]}`}
                >
                  {LABELS[s.key]}
                </span>
              </div>
              <p className="mt-1 font-mono text-lg tabular-nums text-[var(--foreground)]">
                {formatDecimals(s.points, 1)}
              </p>
              <p className="font-mono text-[11px] text-[var(--muted)]">
                {formatDecimals(s.weight, 2)} ×{" "}
                {formatDecimals(s.component, 1)}
              </p>
            </li>
          ))}
        </ul>
      </div>

      {hasLlm && blended !== null && (
        <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
            Deterministic vs LLM blend
          </p>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <div>
              <p className="text-xs text-[var(--muted)]">Deterministic (60%)</p>
              <p className="font-mono text-xl tabular-nums text-teal-800 dark:text-teal-300">
                {formatDecimals(metrics.deterministic_confidence, 1)}
              </p>
            </div>
            <div>
              <p className="text-xs text-[var(--muted)]">LLM (40%)</p>
              <p className="font-mono text-xl tabular-nums text-violet-800 dark:text-violet-300">
                {formatDecimals(metrics.llm_confidence as number, 1)}
              </p>
            </div>
            <div>
              <p className="text-xs text-[var(--muted)]">Displayed</p>
              <p className="font-mono text-xl tabular-nums text-[var(--foreground)]">
                {formatDecimals(blended, 1)}
              </p>
            </div>
          </div>
          <div className="mt-3 flex h-3 overflow-hidden rounded-full ring-1 ring-[var(--border)]">
            <div className="bg-teal-500" style={{ width: "60%" }} />
            <div className="bg-violet-500" style={{ width: "40%" }} />
          </div>
          <p className="mt-2 font-mono text-xs text-[var(--muted)]">
            0.6 × {formatDecimals(metrics.deterministic_confidence, 1)} + 0.4 ×{" "}
            {formatDecimals(metrics.llm_confidence as number, 1)} ={" "}
            {formatDecimals(blended, 1)}
          </p>
        </div>
      )}

      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
          Raw inputs
        </p>
        <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          {(
            [
              ["cluster_size", metrics.cluster_size],
              ["cluster_share", metrics.cluster_share],
              ["best_similarity", metrics.best_similarity],
              ["mean_rating", metrics.mean_rating],
              ["rating_spread", metrics.rating_spread],
              ["cohesion", metrics.cohesion],
            ] as const
          ).map(([label, value]) => (
            <div
              key={label}
              className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2"
            >
              <dt className="text-[10px] uppercase tracking-wider text-[var(--muted)]">
                {label}
              </dt>
              <dd className="mt-0.5 font-mono text-sm tabular-nums text-[var(--foreground)]">
                {formatRawInput(label, value)}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
