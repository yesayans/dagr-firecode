/** Format an ISO date as a short month–year label for projector UI. */
export function formatMonthYear(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

/** Format a review window as "Apr–Sep 2016" (or a single month if equal). */
export function formatReviewWindow(
  start: string | null | undefined,
  end: string | null | undefined,
): string | null {
  if (!start || !end) return null;
  const a = new Date(start);
  const b = new Date(end);
  if (Number.isNaN(a.getTime()) || Number.isNaN(b.getTime())) return null;

  const sameYear = a.getUTCFullYear() === b.getUTCFullYear();
  const sameMonth =
    sameYear && a.getUTCMonth() === b.getUTCMonth();

  if (sameMonth) {
    return formatMonthYear(start);
  }

  const startMonth = a.toLocaleDateString("en-US", {
    month: "short",
    timeZone: "UTC",
  });
  const endMonth = b.toLocaleDateString("en-US", {
    month: "short",
    timeZone: "UTC",
  });
  const endYear = b.getUTCFullYear();

  if (sameYear) {
    return `${startMonth}–${endMonth} ${endYear}`;
  }

  return `${formatMonthYear(start)} – ${formatMonthYear(end)}`;
}
