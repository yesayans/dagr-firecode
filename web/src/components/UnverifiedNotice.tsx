export function UnverifiedNotice() {
  return (
    <div
      role="status"
      className="rounded-lg border border-amber-600/35 bg-amber-500/12 px-5 py-4 text-amber-950 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-100"
    >
      <p className="text-base font-semibold tracking-tight text-amber-900 dark:text-amber-200">
        No public roadmap discoverable
      </p>
      <p className="mt-1.5 max-w-3xl text-sm leading-relaxed text-amber-950/85 dark:text-amber-100/85">
        These needs are surfaced from user review evidence alone. Without a
        public roadmap to compare against, no IGNORED, UNDER-PRIORITIZED, or
        MISUNDERSTOOD verdict can be assigned — every gap is marked UNVERIFIED.
      </p>
    </div>
  );
}
