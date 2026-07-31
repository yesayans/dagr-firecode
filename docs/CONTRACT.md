# dagr — build contract

Single source of truth shared by the backend and frontend workstreams. Neither side may
change a shape here without updating this file first.

## 1. Product statement

dagr cross-references user reviews against a product's roadmap to surface latent, unmet
user needs that the roadmap misses or under-serves. It must work for **any** app,
including closed-source ones with no GitHub repo.

## 2. Roadmap modes

| mode | meaning | verdicts allowed |
|---|---|---|
| `github` | repo known/discoverable; issues + milestones fetched | IGNORED, UNDER-PRIORITIZED, MISUNDERSTOOD |
| `web` | no repo, but changelog/release-notes/roadmap page found | IGNORED, UNDER-PRIORITIZED, MISUNDERSTOOD |
| `hybrid` | both sources contributed | same as above |
| `none` | nothing discoverable | `UNVERIFIED` only |

In `none` mode the UI heading is **"Surfaced Needs (no public roadmap found to verify
against)"** and no IGNORED/UNDER-PRIORITIZED/MISUNDERSTOOD badge may ever be shown.

## 3. Verdict rules (backend, pre-LLM)

Let `s` = best cosine similarity between a review-cluster centroid and any roadmap item.
`MATCH_THRESHOLD` defaults to `0.45` and is per-embedding-backend configurable.

- `s < MATCH_THRESHOLD` → **IGNORED** — nothing on the roadmap addresses this theme.
- `s >= MATCH_THRESHOLD`, matched item is **closed/shipped/released** yet the cluster's
  reviews are still complaining and are more recent than the item's close date →
  **MISUNDERSTOOD** — the team believes this is solved; users disagree.
- `s >= MATCH_THRESHOLD`, matched item is open but **stale** (`updated_at` older than 365
  days) or has **no milestone** → **UNDER-PRIORITIZED**.
- `s >= MATCH_THRESHOLD`, matched item is open, fresh, and milestoned → **well covered,
  drop the cluster** (do not emit a gap).
- mode `none` → **UNVERIFIED**, no similarity step at all.

The LLM may *confirm or adjust* a verdict within the allowed set for the mode; it may
never invent one outside it, and in `none` mode it is forced to `UNVERIFIED`.

## 4. Confidence (0–100, must be reconstructable)

Weighted sum of five named components, each normalised to 0–1:

| component | definition | weight (github/web) | weight (none) |
|---|---|---|---|
| `volume` | `log1p(cluster_size) / log1p(max_cluster_size)` | 0.30 | 0.35 |
| `novelty` | `1 - best_similarity` (1.0 when mode is `none`) | 0.25 | 0.00 |
| `consistency` | mean cosine of members to their centroid | 0.20 | 0.30 |
| `severity` | `(5 - mean_rating) / 4` | 0.15 | 0.20 |
| `spread` | share of distinct star ratings present in the cluster | 0.10 | 0.15 |

`confidence = round(100 * Σ(weight_i * component_i), 2)`.

Every component value, every weight, `cluster_size`, `total_reviews`, `best_similarity`,
`mean_rating`, `rating_spread` and `cohesion` must be persisted so the number can be
recomputed live in front of judges. Store them on `gaps.metrics` (jsonb) **and** mirror
them into the `payload` of each `gap_evidence` row of `source_type='review'`.

The LLM returns its own 0–100 confidence; the stored value is
`0.6 * deterministic + 0.4 * llm` when the LLM ran, and the deterministic value alone
otherwise. `confidence_rationale` must state which formula was used.

## 5. Hard rules

1. No gap is written without **at least one** linked `gap_evidence` row. Enforce in code
   before insert; if evidence is empty, drop the gap and re-rank.
2. The LLM may only cite review IDs from the sample it was given. Validate
   `cited_review_ids ⊆ provided_ids`; drop unknown IDs, and if none survive, fall back to
   the highest-cohesion members of the cluster.
3. No placeholder code, no TODOs, no stubbed returns.
4. Every external dependency has an offline fallback so a live demo cannot fail:
   reviews → local parquet cache; roadmap → `data/roadmaps/*.json` cache; LLM → deterministic
   template extractor with `llm_used: false`; Supabase → local JSON store.
5. Anything degraded must be visible in `/health` and in the job's `stats`, never silently
   faked.

## 6. HTTP API

Base URL `http://127.0.0.1:8000`. All responses JSON. CORS open to `http://localhost:3000`.

### `GET /health`
```json
{
  "ok": true,
  "store": "supabase",
  "llm_enabled": true,
  "llm_model": "openai/gpt-4o-mini",
  "github_token": true,
  "embedding_backend": "minilm",
  "match_threshold": 0.45
}
```

### `GET /apps?q=<substring>&limit=25`
Returns `App[]` from the catalog (seeded from `data/discovery/candidates_sealuzh.json`).

### `POST /apps/resolve`
Request: `{ "app_name": "AntennaPod", "package_name": "de.danoeh.antennapod", "github_repo": null, "refresh": false }`
Response: `App`.

### `POST /analyze`
Request: `{ "app_id": "<uuid>", "max_reviews": 2000, "force": false }`
Response: `{ "job_id": "<uuid>", "status": "queued" }`

### `GET /jobs/{job_id}`
Response: `Job`.

### Types

```ts
type RoadmapSource = "github" | "web" | "hybrid" | "none";
type Verdict = "IGNORED" | "UNDER-PRIORITIZED" | "MISUNDERSTOOD" | "UNVERIFIED";
type JobStatus = "queued" | "running" | "completed" | "failed";
type Stage =
  | "queued" | "resolving_roadmap" | "fetching_reviews" | "embedding"
  | "clustering" | "matching" | "extracting" | "persisting" | "done" | "failed";

interface App {
  id: string;
  package_name: string;
  display_name: string;
  review_count: number;
  avg_stars: number | null;
  github_repo: string | null;
  roadmap_source: RoadmapSource;
  roadmap_item_count: number;
  sample_review: string | null;
}

interface EvidenceItem {
  evidence_id: string;
  source_type: "review" | "github_issue" | "github_milestone" | "web_page" | "interview" | "other";
  title: string | null;
  snippet: string | null;
  url: string | null;
  payload: Record<string, unknown>;
}

interface GapMetrics {
  cluster_size: number;
  total_reviews: number;
  cluster_share: number;
  best_similarity: number | null;
  matched_item_title: string | null;
  matched_item_url: string | null;
  matched_item_state: string | null;
  matched_item_age_days: number | null;
  mean_rating: number;
  rating_spread: number;
  cohesion: number;
  components: { volume: number; novelty: number; consistency: number; severity: number; spread: number };
  weights: { volume: number; novelty: number; consistency: number; severity: number; spread: number };
  deterministic_confidence: number;
  llm_confidence: number | null;
  keywords: string[];
}

interface Gap {
  id: string;
  rank: number;
  need: string;
  one_sentence_summary: string;
  verdict: Verdict;
  confidence: number;            // 0–100
  confidence_rationale: string;
  latent_reasoning: string;
  metrics: GapMetrics;
  evidence: EvidenceItem[];
}

interface Job {
  id: string;
  app: App;
  status: JobStatus;
  stage: Stage;
  progress: number;              // 0–100
  error: string | null;
  roadmap_source: RoadmapSource;
  summary: string | null;
  stats: {
    total_reviews: number;
    clusters: number;
    roadmap_items: number;
    llm_used: boolean;
    embedding_backend: string;
    elapsed_s: number;
    degraded: string[];          // human-readable notes, e.g. "no GITHUB_TOKEN"
  };
  gaps: Gap[];
  created_at: string;
  completed_at: string | null;
}
```

Polling: frontend calls `GET /jobs/{id}` every 1500 ms until `status` is `completed` or
`failed`, rendering `stage`/`progress` meanwhile.

## 7. Backend module map (`api/src/`)

| module | responsibility |
|---|---|
| `config.py` | pydantic-settings; reads `.env`; exposes feature flags used by `/health` |
| `store.py` | `Store` protocol + `SupabaseStore` + `LocalJsonStore` (writes `data/cache/store.json`) |
| `resolver.py` | `RoadmapResolver.resolve()` — wraps existing `silent_stakeholder` package, cache-first |
| `data_ingestion.py` | `ReviewScraper`, `GitHubScraper`, `WebRoadmapScraper` |
| `embedding_engine.py` | `EmbeddingBackend` protocol, `MiniLMBackend`, `TfidfSvdBackend`, clustering |
| `gap_analyzer.py` | `GapMatrix` — similarity, verdicts, deterministic confidence |
| `llm_extractor.py` | `LatentNeedExtractor` — OpenRouter, strict JSON, retry, citation validation |
| `pipeline.py` | orchestration + stage callbacks + result assembly |
| `main.py` | FastAPI app, routes above, background execution |

Each module exposes a `python -m src.<module>` smoke entrypoint that runs standalone.

## 8. Frontend map (`web/src/`)

- `lib/api.ts` — typed client mirroring section 6 exactly, base URL from `NEXT_PUBLIC_API_URL`.
- `lib/types.ts` — the TypeScript interfaces above, verbatim.
- `app/page.tsx` — app search/select, resolve, roadmap-source badge, analyze trigger.
- `app/jobs/[id]/page.tsx` — stage progress, results.
- `components/RoadmapSourceBadge.tsx` — "GitHub roadmap verified" / "Web roadmap verified" /
  "No public roadmap — needs shown unverified".
- `components/GapCard.tsx` — need, confidence %, verdict badge, expandable evidence trace.
- `components/ConfidenceBreakdown.tsx` — renders `metrics.components` × `metrics.weights` as
  a bar chart that visibly sums to the stored confidence.
- `components/EvidenceTrace.tsx` — review snippets and issue/milestone/web links.
- `components/UnverifiedNotice.tsx` — `none`-mode explanation banner.
