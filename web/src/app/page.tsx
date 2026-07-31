"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { resolveApp, searchApps, startAnalyze } from "@/lib/api";
import type { App } from "@/lib/types";
import { CustomAnalyzeForm } from "@/components/CustomAnalyzeForm";
import { RoadmapSourceBadge } from "@/components/RoadmapSourceBadge";
import { useI18n } from "@/lib/i18n";

export default function HomePage() {
  const router = useRouter();
  const { t } = useI18n();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<App[]>([]);
  const [open, setOpen] = useState(false);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [selected, setSelected] = useState<App | null>(null);
  const [resolving, setResolving] = useState(false);
  const [resolveError, setResolveError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [, startTransition] = useTransition();
  const wrapRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (!wrapRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const q = query.trim();
    // Empty query: keep prior browse results; only search when typing.
    if (!q) {
      setSearching(false);
      setSearchError(null);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      setSearchError(null);
      try {
        const apps = await searchApps(q, 40);
        setResults(apps);
        setOpen(true);
      } catch (err) {
        setSearchError(
          err instanceof Error ? err.message : "Failed to search apps",
        );
        setResults([]);
        setOpen(true);
      } finally {
        setSearching(false);
      }
    }, 220);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  async function handleSelect(app: App) {
    setOpen(false);
    setQuery(app.display_name);
    setResolveError(null);
    setAnalyzeError(null);
    setResolving(true);
    try {
      const resolved = await resolveApp({
        app_name: app.display_name,
        package_name: app.package_name,
        github_repo: app.github_repo,
        refresh: false,
      });
      setSelected(resolved);
    } catch (err) {
      setSelected(null);
      setResolveError(
        err instanceof Error ? err.message : "Failed to resolve app",
      );
    } finally {
      setResolving(false);
    }
  }

  async function handleAnalyze() {
    if (!selected) return;
    setAnalyzing(true);
    setAnalyzeError(null);
    try {
      const { job_id } = await startAnalyze({
        app_id: selected.id,
        max_reviews: 2000,
        force: false,
      });
      startTransition(() => {
        router.push(`/jobs/${job_id}`);
      });
    } catch (err) {
      setAnalyzeError(
        err instanceof Error ? err.message : "Failed to start analysis",
      );
      setAnalyzing(false);
    }
  }

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-5xl flex-col px-6 py-10 sm:px-10 sm:py-14">
      <header className="animate-fade-up">
        <p className="font-mono text-xs font-medium uppercase tracking-[0.22em] text-[var(--accent)]">
          {t("brandEyebrow")}
        </p>
        <h1 className="mt-3 text-5xl font-semibold tracking-tight text-[var(--foreground)] sm:text-6xl md:text-7xl">
          dagr
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-[var(--muted)] sm:text-xl">
          {t("tagline")}
        </p>
      </header>

      <section
        className="animate-fade-up relative z-10 mt-12 space-y-6"
        style={{ animationDelay: "80ms" }}
      >
        <div ref={wrapRef} className="space-y-2">
          <label
            htmlFor="app-search"
            className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]"
          >
            {t("findApp")}
          </label>
          <div className="relative">
            <input
              id="app-search"
              type="search"
              autoComplete="off"
              placeholder={t("searchPlaceholder")}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setSelected(null);
              }}
              onFocus={() => {
                setOpen(true);
                if (!query.trim() && results.length === 0 && !searching) {
                  void searchApps("", 40)
                    .then((apps) => {
                      setResults(apps);
                      setSearchError(null);
                    })
                    .catch((err) => {
                      setSearchError(
                        err instanceof Error
                          ? err.message
                          : "Failed to search apps",
                      );
                    });
                }
              }}
              className="w-full rounded-xl border border-[var(--border)] bg-[var(--surface)] px-5 py-4 text-lg text-[var(--foreground)] outline-none ring-0 placeholder:text-[var(--muted)] focus:border-[var(--accent)] focus:shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_18%,transparent)]"
            />
            {searching && (
              <span className="absolute right-4 top-1/2 -translate-y-1/2 font-mono text-xs text-[var(--muted)] animate-pulse-soft">
                {t("searching")}
              </span>
            )}
          </div>

          {/* In-flow panel (not absolute) so it pushes the custom-upload section
              down instead of covering it. */}
          {open && (results.length > 0 || searchError || searching) && (
            <ul
              role="listbox"
              className="max-h-72 w-full overflow-auto rounded-xl border border-[var(--border)] bg-[var(--surface)] py-2 shadow-xl"
            >
              {!query.trim() && !searchError && (
                <li className="px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                  Top apps in catalog · github + closed-source (no public roadmap)
                </li>
              )}
              {searchError && (
                <li className="px-4 py-3 text-sm text-red-600 dark:text-red-300">
                  {searchError}
                </li>
              )}
              {!searchError && !searching && results.length === 0 && query.trim() && (
                <li className="px-4 py-3 text-sm text-[var(--muted)]">
                  {t("noApps", { q: query })}
                </li>
              )}
              {results.map((app) => (
                <li key={app.id}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={selected?.id === app.id}
                    onClick={() => handleSelect(app)}
                    className="flex w-full items-start justify-between gap-4 px-4 py-3 text-left transition hover:bg-[var(--surface-muted)]"
                  >
                    <div>
                      <p className="font-medium text-[var(--foreground)]">
                        {app.display_name}
                      </p>
                      <p className="font-mono text-xs text-[var(--muted)]">
                        {app.package_name}
                      </p>
                    </div>
                    <div className="text-right">
                      <p
                        className={`font-mono text-xs uppercase ${
                          app.roadmap_source === "none"
                            ? "text-amber-800 dark:text-amber-300"
                            : "text-[var(--muted)]"
                        }`}
                      >
                        {app.roadmap_source === "none"
                          ? "closed / no roadmap"
                          : app.roadmap_source}
                      </p>
                      <p className="text-xs text-[var(--muted)]">
                        {app.review_count.toLocaleString()} {t("reviews").toLowerCase()}
                      </p>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {resolving && (
          <div className="rounded-xl border border-[var(--border)] bg-[var(--surface)] px-5 py-6 text-sm text-[var(--muted)] animate-pulse-soft">
            {t("resolving")}
          </div>
        )}

        {resolveError && (
          <div
            role="alert"
            className="rounded-xl border border-red-500/40 bg-red-500/10 px-5 py-4 text-sm text-red-700 dark:text-red-200"
          >
            {resolveError}
          </div>
        )}

        {selected && !resolving && (
          <div className="animate-fade-up space-y-6 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-3xl font-semibold tracking-tight text-[var(--foreground)]">
                  {selected.display_name}
                </h2>
                <p className="mt-1 font-mono text-sm text-[var(--muted)]">
                  {selected.package_name}
                </p>
              </div>
              <RoadmapSourceBadge source={selected.roadmap_source} />
            </div>

            <dl className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-4 py-3">
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                  {t("reviews")}
                </dt>
                <dd className="mt-1 font-mono text-2xl tabular-nums text-[var(--foreground)]">
                  {selected.review_count.toLocaleString()}
                </dd>
              </div>
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-4 py-3">
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                  {t("avgStars")}
                </dt>
                <dd className="mt-1 font-mono text-2xl tabular-nums text-[var(--foreground)]">
                  {selected.avg_stars ?? "—"}
                </dd>
              </div>
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] px-4 py-3">
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
                  {t("roadmapItems")}
                </dt>
                <dd className="mt-1 font-mono text-2xl tabular-nums text-[var(--foreground)]">
                  {selected.roadmap_item_count}
                </dd>
              </div>
            </dl>

            <div className="space-y-2 text-sm">
              {selected.github_repo && (
                <p>
                  <span className="text-[var(--muted)]">GitHub · </span>
                  <a
                    href={`https://github.com/${selected.github_repo}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-mono text-[var(--accent)] hover:opacity-80"
                  >
                    {selected.github_repo}
                  </a>
                </p>
              )}
              {!selected.github_repo && selected.roadmap_source === "web" && (
                <p className="text-[var(--muted)]">
                  No repo linked — web changelog / roadmap pages will be used.
                </p>
              )}
              {selected.roadmap_source === "none" && (
                <p className="text-amber-800 dark:text-amber-200/90">
                  Closed-source / no public roadmap — analysis will surface
                  UNVERIFIED needs from reviews alone.
                </p>
              )}
              {selected.sample_review && (
                <p className="border-l-2 border-[var(--border)] pl-3 italic text-[var(--muted)]">
                  “{selected.sample_review}”
                </p>
              )}
            </div>

            {analyzeError && (
              <div
                role="alert"
                className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-200"
              >
                {analyzeError}
              </div>
            )}

            <button
              type="button"
              onClick={handleAnalyze}
              disabled={analyzing}
              className="inline-flex w-full items-center justify-center rounded-xl bg-[var(--accent)] px-6 py-4 text-lg font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto sm:min-w-[220px] dark:text-zinc-950"
            >
              {analyzing ? t("analyzing") : t("analyze")}
            </button>
          </div>
        )}

        {!selected && !resolving && !resolveError && (
          <div className="rounded-xl border border-dashed border-[var(--border)] bg-[var(--surface)]/50 px-5 py-8 text-center text-sm text-[var(--muted)]">
            Select an app to resolve its roadmap source, then run analysis.
            Catalog covers{" "}
            <span className="text-[var(--foreground)]">github</span>,{" "}
            <span className="text-[var(--foreground)]">web</span>,{" "}
            <span className="text-[var(--foreground)]">hybrid</span>, and{" "}
            <span className="text-[var(--foreground)]">none</span> modes.
          </div>
        )}
      </section>

      <section
        className="animate-fade-up relative z-0 mt-14"
        style={{ animationDelay: "140ms" }}
      >
        <div className="mb-4 flex items-center gap-3">
          <div className="h-px flex-1 bg-[var(--border)]" />
          <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">
            or bring your own data
          </p>
          <div className="h-px flex-1 bg-[var(--border)]" />
        </div>
        <CustomAnalyzeForm />
      </section>
    </div>
  );
}
