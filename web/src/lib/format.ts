/** Format a number to `figs` significant figures for projector-friendly display. */
export function formatSigFigs(n: number, figs = 3): string {
  if (!Number.isFinite(n)) return String(n);
  if (n === 0) return "0";
  return Number(n.toPrecision(figs)).toString();
}

/** Format a number to a fixed count of decimal places. */
export function formatDecimals(n: number, places: number): string {
  if (!Number.isFinite(n)) return String(n);
  return n.toFixed(places);
}
