import type { EvidenceItem } from "@/lib/types";

const GROUP_ORDER = [
  "review",
  "github_issue",
  "github_milestone",
  "web_page",
  "interview",
  "other",
] as const;

const GROUP_LABELS: Record<(typeof GROUP_ORDER)[number], string> = {
  review: "Review snippets",
  github_issue: "GitHub issues",
  github_milestone: "GitHub milestones",
  web_page: "Web pages",
  interview: "Interviews",
  other: "Other",
};

function starsFromPayload(payload: Record<string, unknown>): number | null {
  const stars = payload.stars;
  return typeof stars === "number" ? stars : null;
}

function reviewIdFrom(item: EvidenceItem): string {
  const fromPayload = item.payload.review_id;
  if (typeof fromPayload === "string") return fromPayload;
  return item.evidence_id;
}

function EvidenceRow({ item }: { item: EvidenceItem }) {
  if (item.source_type === "review") {
    const stars = starsFromPayload(item.payload);
    return (
      <li className="rounded-md border border-white/8 bg-white/[0.03] px-4 py-3">
        <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
          {stars !== null && (
            <span className="font-mono text-amber-300/90">{stars}★</span>
          )}
          <span className="font-mono text-zinc-500">
            review {reviewIdFrom(item)}
          </span>
        </div>
        {item.snippet && (
          <p className="mt-2 text-sm leading-relaxed text-zinc-200">
            “{item.snippet}”
          </p>
        )}
      </li>
    );
  }

  const isLink =
    item.source_type === "github_issue" ||
    item.source_type === "github_milestone" ||
    item.source_type === "web_page";

  const body = (
    <>
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm font-medium text-zinc-100">
          {item.title ?? "Untitled"}
        </p>
        {isLink && item.url && (
          <span className="shrink-0 text-xs text-teal-400">↗</span>
        )}
      </div>
      {item.snippet && (
        <p className="mt-1.5 text-sm leading-relaxed text-zinc-400">
          {item.snippet}
        </p>
      )}
    </>
  );

  if (isLink && item.url) {
    return (
      <li>
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block rounded-md border border-white/8 bg-white/[0.03] px-4 py-3 transition hover:border-teal-500/40 hover:bg-teal-500/5"
        >
          {body}
        </a>
      </li>
    );
  }

  return (
    <li className="rounded-md border border-white/8 bg-white/[0.03] px-4 py-3">
      {body}
    </li>
  );
}

export function EvidenceTrace({ evidence }: { evidence: EvidenceItem[] }) {
  if (evidence.length === 0) {
    return (
      <p className="text-sm text-red-300">
        No evidence attached — this gap should not have been emitted.
      </p>
    );
  }

  const groups = GROUP_ORDER.map((type) => ({
    type,
    items: evidence.filter((e) => e.source_type === type),
  })).filter((g) => g.items.length > 0);

  return (
    <div className="space-y-5">
      {groups.map((group) => (
        <div key={group.type}>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-zinc-500">
            {GROUP_LABELS[group.type]}
          </h4>
          <ul className="space-y-2">
            {group.items.map((item) => (
              <EvidenceRow key={item.evidence_id} item={item} />
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
