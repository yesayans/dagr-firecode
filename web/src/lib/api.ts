import {
  MOCK_HEALTH,
  jobForApp,
  resolveMockApp,
  searchMockApps,
} from "./mocks";
import type {
  AnalyzeRequest,
  AnalyzeResponse,
  App,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  Job,
  ResolveAppRequest,
  Stage,
} from "./types";
import { STAGE_SEQUENCE } from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

const FORCE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "1";

export type DataMode = "live" | "mock";

let activeMode: DataMode | null = FORCE_MOCK ? "mock" : null;
let modeResolved = FORCE_MOCK;

type Listener = (mode: DataMode) => void;
const listeners = new Set<Listener>();

export function getDataMode(): DataMode {
  return activeMode ?? (FORCE_MOCK ? "mock" : "live");
}

export function subscribeDataMode(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function setMode(mode: DataMode): void {
  if (activeMode === mode && modeResolved) return;
  activeMode = mode;
  modeResolved = true;
  listeners.forEach((l) => l(mode));
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function tryLive<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      body || `API ${res.status} ${res.statusText} for ${path}`,
    );
  }
  return res.json() as Promise<T>;
}

// Mock fixtures are served only when explicitly opted into. A live request that fails
// must surface as an error: silently substituting fabricated gaps for a dead backend
// would present invented user needs as real analysis.
async function request<T>(
  live: () => Promise<T>,
  mock: () => Promise<T>,
): Promise<T> {
  if (FORCE_MOCK) {
    setMode("mock");
    await delay(280 + Math.random() * 220);
    return mock();
  }

  const result = await live();
  setMode("live");
  return result;
}

// --- Mock job progression ---

interface MockRun {
  job: Job;
  startedAt: number;
}

const mockRuns = new Map<string, MockRun>();

const STAGE_DURATION_MS = 8000 / (STAGE_SEQUENCE.length - 1);

function advanceMockJob(run: MockRun): Job {
  const elapsed = Date.now() - run.startedAt;
  const step = Math.min(
    STAGE_SEQUENCE.length - 1,
    Math.floor(elapsed / STAGE_DURATION_MS),
  );
  const stage: Stage = STAGE_SEQUENCE[step];
  const progress = Math.min(
    100,
    Math.round((elapsed / 8000) * 100),
  );

  if (stage === "done" || elapsed >= 8000) {
    return {
      ...run.job,
      status: "completed",
      stage: "done",
      progress: 100,
      completed_at: new Date().toISOString(),
    };
  }

  return {
    ...run.job,
    status: stage === "queued" ? "queued" : "running",
    stage,
    progress,
    gaps: [],
    summary: null,
    completed_at: null,
  };
}

export async function getHealth(): Promise<HealthResponse> {
  return request(
    () => tryLive<HealthResponse>("/health"),
    async () => MOCK_HEALTH,
  );
}

export async function searchApps(q: string, limit = 25): Promise<App[]> {
  const qs = new URLSearchParams();
  if (q) qs.set("q", q);
  qs.set("limit", String(limit));
  return request(
    () => tryLive<App[]>(`/apps?${qs.toString()}`),
    async () => searchMockApps(q, limit),
  );
}

export async function resolveApp(
  body: ResolveAppRequest,
): Promise<App> {
  return request(
    () =>
      tryLive<App>("/apps/resolve", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    async () =>
      resolveMockApp({
        app_name: body.app_name,
        package_name: body.package_name,
        github_repo: body.github_repo,
      }),
  );
}

export interface CustomAppResponse {
  app: App;
  job_id: string;
  status: "queued" | "completed";
  column_mapping: {
    review_text: string | null;
    rating: string | null;
    created_at: string | null;
  };
  warnings: string[];
  rows_kept: number;
  rows_raw: number;
  roadmap_source: App["roadmap_source"];
  roadmap_item_count: number;
}

export async function createCustomApp(input: {
  appName: string;
  packageName?: string;
  roadmapUrls?: string;
  roadmapText?: string;
  maxReviews?: number;
  reviewsFile: File;
}): Promise<CustomAppResponse> {
  if (FORCE_MOCK || getDataMode() === "mock") {
    setMode("mock");
    await delay(400);
    const id = `custom-mock-${Date.now()}`;
    const jobId = `mock-${id}`;
    const app: App = {
      id,
      package_name: input.packageName || `custom.${input.appName.toLowerCase().replace(/\W+/g, ".")}`,
      display_name: input.appName,
      review_count: 42,
      avg_stars: 2.8,
      github_repo: null,
      roadmap_source: input.roadmapText || input.roadmapUrls ? "web" : "none",
      roadmap_item_count: input.roadmapText ? 4 : 0,
      sample_review: "Mock uploaded review for custom analyze.",
    };
    const template = jobForApp("app-none-instagram");
    mockRuns.set(jobId, {
      job: {
        ...template,
        id: jobId,
        app,
        roadmap_source: app.roadmap_source,
        created_at: new Date().toISOString(),
      },
      startedAt: Date.now(),
    });
    return {
      app,
      job_id: jobId,
      status: "queued",
      column_mapping: {
        review_text: "review",
        rating: "stars",
        created_at: "date",
      },
      warnings: ["mock mode — CSV not parsed"],
      rows_kept: 42,
      rows_raw: 50,
      roadmap_source: app.roadmap_source,
      roadmap_item_count: app.roadmap_item_count,
    };
  }

  const form = new FormData();
  form.append("app_name", input.appName);
  if (input.packageName) form.append("package_name", input.packageName);
  if (input.roadmapUrls) form.append("roadmap_urls", input.roadmapUrls);
  if (input.roadmapText) form.append("roadmap_text", input.roadmapText);
  if (input.maxReviews != null) {
    form.append("max_reviews", String(input.maxReviews));
  }
  form.append("reviews", input.reviewsFile);

  const res = await fetch(`${BASE_URL}/apps/custom`, {
    method: "POST",
    body: form,
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(body || `API ${res.status} for /apps/custom`);
  }
  setMode("live");
  return res.json() as Promise<CustomAppResponse>;
}

export async function startAnalyze(
  body: AnalyzeRequest,
): Promise<AnalyzeResponse> {
  return request(
    () =>
      tryLive<AnalyzeResponse>("/analyze", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    async () => {
      const template = jobForApp(body.app_id);
      const jobId = `mock-${body.app_id}-${Date.now()}`;
      const created_at = new Date().toISOString();
      mockRuns.set(jobId, {
        job: { ...template, id: jobId, created_at },
        startedAt: Date.now(),
      });
      return { job_id: jobId, status: "queued" as const };
    },
  );
}

export async function getJob(id: string): Promise<Job> {
  return request(
    () => tryLive<Job>(`/jobs/${id}`),
    async () => {
      const run = mockRuns.get(id);
      if (run) {
        const advanced = advanceMockJob(run);
        if (advanced.status === "completed") {
          mockRuns.set(id, {
            ...run,
            job: advanced,
            startedAt: run.startedAt - 9000,
          });
        }
        return advanced;
      }

      // Direct fixture ids for deep-linking demos
      const { MOCK_JOBS } = await import("./mocks");
      if (MOCK_JOBS[id]) {
        return structuredClone(MOCK_JOBS[id]);
      }

      throw new Error(`Unknown mock job: ${id}`);
    },
  );
}

export async function postJobChat(
  jobId: string,
  body: ChatRequest,
): Promise<ChatResponse> {
  if (FORCE_MOCK || getDataMode() === "mock") {
    setMode("mock");
    await delay(350);
    const gap = (
      mockRuns.get(jobId)?.job ??
      (await import("./mocks")).MOCK_JOBS[jobId]
    )?.gaps?.[0];
    return {
      answer:
        "From this job’s evidence, the strongest signal is the top-ranked gap. " +
        "Prioritize it if confidence and review density stay high — this is a mock reply.",
      citations: gap
        ? [
            {
              gap_rank: gap.rank,
              evidence_id: gap.evidence[0]?.evidence_id ?? null,
              quote: gap.evidence[0]?.snippet ?? gap.one_sentence_summary,
            },
          ]
        : [],
      model: "mock",
    };
  }

  const res = await fetch(`${BASE_URL}/jobs/${jobId}/chat`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `API ${res.status} for /jobs/${jobId}/chat`);
  }
  setMode("live");
  return res.json() as Promise<ChatResponse>;
}
