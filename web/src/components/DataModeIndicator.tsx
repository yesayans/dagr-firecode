"use client";

import { useEffect, useState } from "react";
import { getDataMode, subscribeDataMode, type DataMode } from "@/lib/api";

export function DataModeIndicator() {
  const [mode, setMode] = useState<DataMode>(() => getDataMode());

  useEffect(() => {
    setMode(getDataMode());
    return subscribeDataMode(setMode);
  }, []);

  // Mock mode must be impossible to miss: these are invented needs and invented
  // reviews, and they must never be mistaken for analysis of real user feedback.
  if (mode === "mock") {
    return (
      <div className="fixed inset-x-0 bottom-0 z-50 border-t-2 border-amber-400 bg-amber-500/95 px-4 py-2 text-center font-mono text-xs font-bold uppercase tracking-widest text-amber-950 shadow-2xl">
        Demo fixtures — not real analysis. Fabricated apps, reviews and gaps.
      </div>
    );
  }

  if (process.env.NODE_ENV === "production") {
    return null;
  }

  return (
    <div
      className="fixed bottom-4 right-4 z-50 rounded-md bg-emerald-500/20 px-3 py-1.5 font-mono text-[11px] font-medium uppercase tracking-wider text-emerald-200 shadow-lg ring-1 ring-emerald-500/40"
      title="Talking to live API"
    >
      data: live
    </div>
  );
}
