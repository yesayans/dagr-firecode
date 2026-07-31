import type { RoadmapSource } from "@/lib/types";

const CONFIG: Record<
  RoadmapSource,
  { label: string; className: string; dot: string }
> = {
  github: {
    label: "GitHub roadmap verified",
    className:
      "bg-emerald-500/15 text-emerald-800 ring-emerald-600/35 dark:text-emerald-300 dark:ring-emerald-500/40",
    dot: "bg-emerald-600 dark:bg-emerald-400",
  },
  web: {
    label: "Web roadmap verified",
    className:
      "bg-sky-500/15 text-sky-900 ring-sky-600/35 dark:text-sky-300 dark:ring-sky-500/40",
    dot: "bg-sky-600 dark:bg-sky-400",
  },
  hybrid: {
    label: "Hybrid roadmap verified",
    className:
      "bg-cyan-500/15 text-cyan-900 ring-cyan-600/35 dark:text-cyan-300 dark:ring-cyan-500/40",
    dot: "bg-cyan-700 dark:bg-cyan-400",
  },
  none: {
    label: "No public roadmap — needs shown unverified",
    className:
      "bg-amber-500/15 text-amber-900 ring-amber-600/40 dark:text-amber-300 dark:ring-amber-500/40",
    dot: "bg-amber-600 dark:bg-amber-400",
  },
};

export function RoadmapSourceBadge({
  source,
  className = "",
}: {
  source: RoadmapSource;
  className?: string;
}) {
  const cfg = CONFIG[source];
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium ring-1 ring-inset ${cfg.className} ${className}`}
    >
      <span className={`h-2 w-2 rounded-sm ${cfg.dot}`} aria-hidden />
      {cfg.label}
    </span>
  );
}
