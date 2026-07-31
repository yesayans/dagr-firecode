import type { RoadmapSource } from "@/lib/types";

const CONFIG: Record<
  RoadmapSource,
  { label: string; className: string }
> = {
  github: {
    label: "GitHub roadmap verified",
    className: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/40",
  },
  web: {
    label: "Web roadmap verified",
    className: "bg-sky-500/15 text-sky-300 ring-sky-500/40",
  },
  hybrid: {
    label: "Hybrid roadmap verified",
    className: "bg-cyan-500/15 text-cyan-300 ring-cyan-500/40",
  },
  none: {
    label: "No public roadmap — needs shown unverified",
    className: "bg-amber-500/15 text-amber-300 ring-amber-500/40",
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
      <span
        className={`h-2 w-2 rounded-sm ${
          source === "github"
            ? "bg-emerald-400"
            : source === "web"
              ? "bg-sky-400"
              : source === "hybrid"
                ? "bg-cyan-400"
                : "bg-amber-400"
        }`}
        aria-hidden
      />
      {cfg.label}
    </span>
  );
}
