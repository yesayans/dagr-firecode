"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { resolveApp, searchApps, startAnalyze } from "@/lib/api";
import type { App } from "@/lib/types";
import { RoadmapSourceBadge } from "@/components/RoadmapSourceBadge";

export default function HomePage() {
  const router = useRouter();
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
    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      setSearchError(null);
      try {
        const apps = await searchApps(query);
        setResults(apps);
        setOpen(true);
      } catch (err) {
        setSearchError(
          err instanceof Error ? err.message : "Failed to search apps",
        );
        setResults([]);
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
        <p className="font-mono text-xs font-medium uppercase tracking-[0.22em] text-teal-400/90">
          Silent Stakeholder
        </p>
        <h1 className="mt-3 text-5xl font-semibold tracking-tight text-white sm:text-6xl md:text-7xl">
          dagr
        </h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-zinc-400 sm:text-xl">
          Cross-reference app-store reviews against a product roadmap — for{" "}
          <span className="text-zinc-200">any</span> app, including closed-source
          ones with no GitHub repo — and surface latent needs the roadmap misses.
        </p>
      </header>

      <section
        className="animate-fade-up mt-12 space-y-6"
        style={{ animationDelay: "80ms" }}
      >
        <div ref={wrapRef} className="relative">
          <label
            htmlFor="app-search"
            className="mb-2 block text-xs font-semibold uppercase tracking-[0.14em] text-zinc-500"
          >
            Find an app
          </label>
          <div className="relative">
            <input
              id="app-search"
              type="search"
              autoComplete="off"
              placeholder="Search by name or package — try AntennaPod, Signal, Instagram…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setSelected(null);
              }}
              onFocus={() => setOpen(true)}
              className="w-full rounded-xl border border-white/12 bg-zinc-900/80 px-5 py-4 text-lg text-white outline-none ring-0 placeholder:text-zinc-600 focus:border-teal-400/50 focus:shadow-[0_0_0_3px_rgba(45,212,191,0.12)]"
            />
            {searching && (
              <span className="absolute right-4 top-1/2 -translate-y-1/2 font-mono text-xs text-zinc-500 animate-pulse-soft">
                searching…
              </span>
            )}
          </div>

          {open && (results.length > 0 || searchError || (!searching && query)) && (
            <ul
              role="listbox"
              className="absolute z-20 mt-2 max-h-80 w-full overflow-auto rounded-xl border border-white/10 bg-zinc-950/95 py-2 shadow-2xl backdrop-blur"
            >
              {searchError && (
                <li className="px-4 py-3 text-sm text-red-300">{searchError}</li>
              )}
              {!searchError && results.length === 0 && (
                <li className="px-4 py-3 text-sm text-zinc-500">
                  No apps matched “{query}”.
                </li>
              )}
              {results.map((app) => (
                <li key={app.id}>
                  <button
                    type="button"
                    role="option"
                    aria-selected={selected?.id === app.id}
                    onClick={() => handleSelect(app)}
                    className="flex w-full items-start justify-between gap-4 px-4 py-3 text-left transition hover:bg-white/5"
                  >
                    <div>
                      <p className="font-medium text-white">
                        {app.display_name}
                      </p>
                      <p className="font-mono text-xs text-zinc-500">
                        {app.package_name}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-xs uppercase text-zinc-400">
                        {app.roadmap_source}
                      </p>
                      <p className="text-xs text-zinc-600">
                        {app.review_count.toLocaleString()} reviews
                      </p>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {resolving && (
          <div className="rounded-xl border border-white/10 bg-zinc-900/50 px-5 py-6 text-sm text-zinc-400 animate-pulse-soft">
            Resolving roadmap source…
          </div>
        )}

        {resolveError && (
          <div
            role="alert"
            className="rounded-xl border border-red-500/40 bg-red-500/10 px-5 py-4 text-sm text-red-200"
          >
            {resolveError}
          </div>
        )}

        {selected && !resolving && (
          <div className="animate-fade-up space-y-6 rounded-xl border border-white/10 bg-zinc-900/60 p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <h2 className="text-3xl font-semibold tracking-tight text-white">
                  {selected.display_name}
                </h2>
                <p className="mt-1 font-mono text-sm text-zinc-500">
                  {selected.package_name}
                </p>
              </div>
              <RoadmapSourceBadge source={selected.roadmap_source} />
            </div>

            <dl className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg border border-white/8 bg-black/25 px-4 py-3">
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-500">
                  Reviews
                </dt>
                <dd className="mt-1 font-mono text-2xl tabular-nums text-white">
                  {selected.review_count.toLocaleString()}
                </dd>
              </div>
              <div className="rounded-lg border border-white/8 bg-black/25 px-4 py-3">
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-500">
                  Avg stars
                </dt>
                <dd className="mt-1 font-mono text-2xl tabular-nums text-white">
                  {selected.avg_stars ?? "—"}
                </dd>
              </div>
              <div className="rounded-lg border border-white/8 bg-black/25 px-4 py-3">
                <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-zinc-500">
                  Roadmap items
                </dt>
                <dd className="mt-1 font-mono text-2xl tabular-nums text-white">
                  {selected.roadmap_item_count}
                </dd>
              </div>
            </dl>

            <div className="space-y-2 text-sm">
              {selected.github_repo && (
                <p>
                  <span className="text-zinc-500">GitHub · </span>
                  <a
                    href={`https://github.com/${selected.github_repo}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-mono text-teal-300 hover:text-teal-200"
                  >
                    {selected.github_repo}
                  </a>
                </p>
              )}
              {!selected.github_repo && selected.roadmap_source === "web" && (
                <p className="text-zinc-400">
                  No repo linked — web changelog / roadmap pages will be used.
                </p>
              )}
              {selected.roadmap_source === "none" && (
                <p className="text-amber-200/90">
                  Closed-source / no public roadmap — analysis will surface
                  UNVERIFIED needs from reviews alone.
                </p>
              )}
              {selected.sample_review && (
                <p className="border-l-2 border-white/15 pl-3 italic text-zinc-400">
                  “{selected.sample_review}”
                </p>
              )}
            </div>

            {analyzeError && (
              <div
                role="alert"
                className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200"
              >
                {analyzeError}
              </div>
            )}

            <button
              type="button"
              onClick={handleAnalyze}
              disabled={analyzing}
              className="inline-flex w-full items-center justify-center rounded-xl bg-teal-400 px-6 py-4 text-lg font-semibold text-zinc-950 transition hover:bg-teal-300 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto sm:min-w-[220px]"
            >
              {analyzing ? "Starting analysis…" : "Analyze"}
            </button>
          </div>
        )}

        {!selected && !resolving && !resolveError && (
          <div className="rounded-xl border border-dashed border-white/12 bg-white/[0.02] px-5 py-8 text-center text-sm text-zinc-500">
            Select an app to resolve its roadmap source, then run analysis.
            Demo fixtures cover{" "}
            <span className="text-zinc-300">github</span>,{" "}
            <span className="text-zinc-300">web</span>,{" "}
            <span className="text-zinc-300">hybrid</span>, and{" "}
            <span className="text-zinc-300">none</span> modes.
          </div>
        )}
      </section>
    </div>
  );
}
