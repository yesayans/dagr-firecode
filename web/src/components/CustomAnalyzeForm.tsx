"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { createCustomApp } from "@/lib/api";
import { RoadmapSourceBadge } from "@/components/RoadmapSourceBadge";
import type { RoadmapSource } from "@/lib/types";

const fieldClass =
  "w-full rounded-lg border border-[var(--border)] bg-[var(--input-bg)] px-3 py-2.5 text-sm text-[var(--foreground)] outline-none focus:border-[var(--accent)] focus:shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_18%,transparent)]";

export function CustomAnalyzeForm() {
  const router = useRouter();
  const [, startTransition] = useTransition();
  const [appName, setAppName] = useState("");
  const [packageName, setPackageName] = useState("");
  const [roadmapUrls, setRoadmapUrls] = useState("");
  const [roadmapText, setRoadmapText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<{
    rowsKept: number;
    rowsRaw: number;
    mapping: Record<string, string | null>;
    warnings: string[];
    roadmapSource: RoadmapSource;
    roadmapItems: number;
  } | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setInfo(null);
    if (!appName.trim()) {
      setError("App name is required.");
      return;
    }
    if (!file) {
      setError("Upload a reviews CSV.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await createCustomApp({
        appName: appName.trim(),
        packageName: packageName.trim() || undefined,
        roadmapUrls: roadmapUrls.trim() || undefined,
        roadmapText: roadmapText.trim() || undefined,
        reviewsFile: file,
      });
      setInfo({
        rowsKept: res.rows_kept,
        rowsRaw: res.rows_raw,
        mapping: res.column_mapping,
        warnings: res.warnings,
        roadmapSource: res.roadmap_source,
        roadmapItems: res.roadmap_item_count,
      });
      startTransition(() => {
        router.push(`/jobs/${res.job_id}`);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setSubmitting(false);
    }
  }

  return (
    <form
      onSubmit={onSubmit}
      className="space-y-5 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 shadow-[var(--shadow)] sm:p-8"
    >
      <div>
        <h2 className="text-xl font-semibold text-[var(--foreground)]">
          Analyze your own data
        </h2>
        <p className="mt-1 text-sm text-[var(--muted)]">
          Upload a reviews CSV for any app — including closed-source. Optionally
          add changelog/roadmap URLs or paste a feature list.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block space-y-1.5 sm:col-span-1">
          <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
            App name *
          </span>
          <input
            required
            value={appName}
            onChange={(e) => setAppName(e.target.value)}
            placeholder="Acme Notes"
            className={fieldClass}
          />
        </label>
        <label className="block space-y-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
            Package / id (optional)
          </span>
          <input
            value={packageName}
            onChange={(e) => setPackageName(e.target.value)}
            placeholder="com.acme.notes"
            className={`${fieldClass} font-mono`}
          />
        </label>
      </div>

      <label className="block space-y-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
          Reviews CSV *
        </span>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm text-[var(--foreground)] file:mr-4 file:rounded-lg file:border-0 file:bg-[color-mix(in_srgb,var(--accent)_18%,transparent)] file:px-3 file:py-2 file:text-sm file:font-semibold file:text-[var(--accent)] hover:file:opacity-90"
        />
        <span className="block text-xs text-[var(--muted)]">
          Auto-detects columns like review/text/body, rating/stars, date.
          5-star and under-10-word reviews are dropped.
        </span>
      </label>

      <label className="block space-y-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
          Roadmap / changelog URLs (optional)
        </span>
        <textarea
          value={roadmapUrls}
          onChange={(e) => setRoadmapUrls(e.target.value)}
          rows={2}
          placeholder={"https://example.com/changelog\nhttps://example.com/roadmap"}
          className={`${fieldClass} font-mono`}
        />
      </label>

      <label className="block space-y-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
          Paste roadmap / upcoming features (optional)
        </span>
        <textarea
          value={roadmapText}
          onChange={(e) => setRoadmapText(e.target.value)}
          rows={4}
          placeholder={"Offline sync\nShared workspaces\nExport to PDF"}
          className={fieldClass}
        />
        <span className="block text-xs text-[var(--muted)]">
          One item per line (or blank-line paragraphs). Used when the app has no
          public GitHub roadmap.
        </span>
      </label>

      {error && (
        <div
          role="alert"
          className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-200"
        >
          {error}
        </div>
      )}

      {info && (
        <div className="space-y-2 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-4 py-3 text-sm text-[var(--foreground)]">
          <div className="flex flex-wrap items-center gap-3">
            <RoadmapSourceBadge source={info.roadmapSource} />
            <span className="font-mono text-xs text-[var(--muted)]">
              {info.rowsKept}/{info.rowsRaw} reviews · {info.roadmapItems}{" "}
              roadmap items
            </span>
          </div>
          <p className="font-mono text-[11px] text-[var(--muted)]">
            columns: text={info.mapping.review_text ?? "—"} · rating=
            {info.mapping.rating ?? "—"} · date={info.mapping.created_at ?? "—"}
          </p>
          {info.warnings.length > 0 && (
            <ul className="list-inside list-disc text-xs text-[var(--muted)]">
              {info.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <button
        type="submit"
        disabled={submitting}
        className="inline-flex w-full items-center justify-center rounded-xl bg-[var(--accent)] px-6 py-3.5 text-base font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto dark:text-zinc-950"
      >
        {submitting ? "Uploading & analyzing…" : "Upload & analyze"}
      </button>
    </form>
  );
}
