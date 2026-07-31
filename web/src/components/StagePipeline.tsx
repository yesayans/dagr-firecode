import type { Stage } from "@/lib/types";
import { STAGE_SEQUENCE } from "@/lib/types";

const LABELS: Record<Stage, string> = {
  queued: "Queued",
  resolving_roadmap: "Resolving roadmap",
  fetching_reviews: "Fetching reviews",
  embedding: "Embedding",
  clustering: "Clustering",
  matching: "Matching",
  extracting: "Extracting",
  persisting: "Persisting",
  done: "Done",
  failed: "Failed",
};

export function StagePipeline({
  stage,
  progress,
}: {
  stage: Stage;
  progress: number;
}) {
  const activeIndex =
    stage === "failed"
      ? -1
      : Math.max(0, STAGE_SEQUENCE.indexOf(stage));

  return (
    <div className="space-y-5">
      <div>
        <div className="mb-2 flex items-end justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
              Pipeline
            </p>
            <p className="mt-1 text-lg font-medium text-[var(--foreground)]">
              {LABELS[stage]}
            </p>
          </div>
          <p className="font-mono text-3xl font-semibold tabular-nums text-[var(--accent)]">
            {Math.round(progress)}%
          </p>
        </div>
        <div
          className="h-2.5 overflow-hidden rounded-full"
          style={{ background: "var(--chart-track)" }}
        >
          <div
            className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-500 ease-out"
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
        </div>
      </div>

      <ol className="grid gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {STAGE_SEQUENCE.map((s, i) => {
          const done = stage === "done" || (activeIndex >= 0 && i < activeIndex);
          const current = stage !== "failed" && i === activeIndex;
          return (
            <li
              key={s}
              className={`rounded-md border px-3 py-2.5 text-sm transition ${
                current
                  ? "border-[var(--accent)] bg-[color-mix(in_srgb,var(--accent)_14%,transparent)] text-[var(--accent)]"
                  : done
                    ? "border-[var(--border)] bg-[var(--surface-muted)] text-[var(--foreground)]"
                    : "border-[var(--border)] bg-[var(--input-bg)] text-[var(--muted)]"
              }`}
            >
              <span className="font-mono text-[10px] text-[var(--muted)]">
                {String(i + 1).padStart(2, "0")}
              </span>
              <p className="mt-0.5 font-medium leading-snug">{LABELS[s]}</p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
