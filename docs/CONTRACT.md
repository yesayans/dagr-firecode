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

Let `s` = best cosine similarity between a review-cluster centroid and any roadmap item
that is **contemporaneous with the review corpus** (item `created_at` ≤
`review_window_end`). Items created after the corpus window never count as covering a
complaint — they are reserved for retrospective validation (section 4).

`MATCH_THRESHOLD` is per-embedding-backend configurable (`MATCH_THRESHOLD_TFIDF`
defaults to `0.18` for char_wb TF-IDF; `MATCH_THRESHOLD_MINILM` defaults to `0.45`).
A match also requires a relative margin: top-1 similarity must exceed the runner-up
by at least `MATCH_MARGIN_TFIDF` (default `0.015`) / `MATCH_MARGIN_MINILM`.

**What the threshold is and is not.** Calibrated by `scripts/calibrate_retrieval.py`
against 192 live AntennaPod roadmap items with five hand-labelled review probes
(fixture: `api/tests/fixtures/retrieval_calibration.json`): char_wb retrieves the
correct top-1 for 5/5 probes, and the weakest true match scores `0.181`. The
threshold is therefore a **recall floor** — it is set just low enough to admit every
known true match. It is **not** a precision guarantee: an intentionally ambiguous
negative probe ("Podcasts won't start playing") reaches `0.191`, and the runner-up
within a correct query reaches `0.331`. Absolute similarity distributions for true
matches and plausible-but-wrong items overlap. Precision comes from taking top-1 and
requiring `MATCH_MARGIN`, not from the absolute number. Consequence for verdicts: the
`IGNORED` branch (`s < threshold`) is conservative — a theme must be lexically distant
from *every* contemporaneous item before we claim the roadmap ignores it.

All recency / staleness comparisons are anchored to the **review corpus window**, not
wall-clock `now`. Compute `review_window_start` / `review_window_end` from the
`created_at` range of reviews used in the job; the reference date is
`review_window_end`. Persist both bounds (and the reference date) on every gap's
`metrics` and on the job `stats`.

- `s < MATCH_THRESHOLD` → **IGNORED** — nothing on the contemporaneous roadmap addresses
  this theme.
- `s >= MATCH_THRESHOLD`, matched item is **closed/shipped/released as of
  `review_window_end`** (its `closed_at` ≤ window end) yet reviews inside the window
  still complain about it → **MISUNDERSTOOD** — the team believed it was done; users
  in the corpus disagree.
- `s >= MATCH_THRESHOLD`, matched item is **open as of `review_window_end`** but
  **stale** (last touch more than 365 days before `review_window_end`) or has
  **no milestone** → **UNDER-PRIORITIZED**.
- `s >= MATCH_THRESHOLD`, matched item is open as of the window, fresh within 365 days
  of `review_window_end`, and milestoned → **well covered, drop the cluster**.
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

### Retrospective validation (corroboration only — does not change verdict)

When the review corpus predates the live roadmap (e.g. 2016 reviews vs 2026 GitHub),
each gap may carry an optional signal that the need was later addressed:

- `later_addressed_by`: best roadmap item created or closed **after**
  `review_window_end` whose embedding similarity to the cluster exceeds
  `MATCH_THRESHOLD`, or `null` when none exist. Shape:
  `{ title, url, state, date, similarity }`.
- `validated_by_later_roadmap`: `true` iff `later_addressed_by` is non-null.

This is **not** a verdict. A gap that is `IGNORED` against the 2016-era roadmap can
still show `validated_by_later_roadmap: true` if the product shipped a fix years
later — evidence the method predicted real roadmap movement. Conversely, a decade
with no post-window match strengthens the claim of a standing ignored need.

Also persist on every `GapMetrics` (and job `stats`):

- `review_window_start` / `review_window_end` / `reference_date` (ISO-8601;
  `reference_date === review_window_end`)

### Need-bearing selection (before clustering)

Clustering runs only on **need-bearing** reviews — reviews that express an unmet want,
a problem, or a low rating. Pure praise with no want/problem language is excluded at
the *review* level (not by cluster mean rating), so polite four-star feedback that
mentions a latent goal remains eligible. Job `stats` reports:

- `reviews_total` — all reviews loaded for the job
- `reviews_need_bearing` — count that survived the filter and were clustered

Each gap's `metrics.need_bearing_share` is the share of that cluster's members that
were classified need-bearing (1.0 when clustering the filtered set only).

## 5. Hard rules

1. No gap is written without **at least one** linked `gap_evidence` row. Enforce in code
   before insert; if evidence is empty, drop the gap and re-rank.
2. The LLM may only cite review IDs from the sample it was given. Validate
   `cited_review_ids ⊆ provided_ids`; drop unknown IDs, and if none survive, fall back to
   the highest-cohesion members of the cluster.
3. No placeholder code, no TODOs, no stubbed returns.
4. Every external dependency has an offline fallback so a live demo cannot fail:
   reviews → local parquet cache; roadmap → `data/roadmaps/*.json` cache; LLM → quote the
   cluster's most representative need-bearing review verbatim (`need_source:
   "representative_review"`, `llm_used: false`) — never synthesise a fake need statement;
   Supabase → local JSON store. GitHub auth may come from `GITHUB_TOKEN` or, when
   `GITHUB_TOKEN_FROM_GIT_CREDENTIAL=true` (default), from `git credential fill` for
   `github.com` (resolved once into memory; never written to `.env`).
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
  "github_token_source": "git_credential",
  "embedding_backend": "minilm",
  "match_threshold": 0.45
}
```

`github_token_source` is `"env" | "git_credential" | "none"`. The boolean
`github_token` remains `true` iff the source is not `"none"`.

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

interface LaterAddressedBy {
  title: string;
  url: string | null;
  state: string | null;
  date: string | null;       // ISO-8601 created/closed date of the later item
  similarity: number;
}

interface GapMetrics {
  cluster_size: number;
  total_reviews: number;
  cluster_share: number;
  best_similarity: number | null;
  matched_item_title: string | null;
  matched_item_url: string | null;
  matched_item_state: string | null;
  matched_item_age_days: number | null;  // age as of review_window_end, not wall-clock
  mean_rating: number;
  rating_spread: number;
  cohesion: number;
  components: { volume: number; novelty: number; consistency: number; severity: number; spread: number };
  weights: { volume: number; novelty: number; consistency: number; severity: number; spread: number };
  deterministic_confidence: number;
  llm_confidence: number | null;
  keywords: string[];
  review_window_start: string;   // ISO-8601
  review_window_end: string;     // ISO-8601 — also the temporal reference_date
  reference_date: string;        // === review_window_end
  later_addressed_by: LaterAddressedBy | null;
  validated_by_later_roadmap: boolean;
  need_bearing_share: number;    // 0–1 share of cluster members classified need-bearing
}

type NeedSource = "llm" | "representative_review";

interface Gap {
  id: string;
  rank: number;
  need: string;
  one_sentence_summary: string;
  verdict: Verdict;
  confidence: number;            // 0–100
  confidence_rationale: string;
  latent_reasoning: string;
  /** Distinguishes an LLM-inferred latent need from a verbatim user quote. */
  need_source: NeedSource;
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
    review_provenance: "hf" | "parquet_cache" | "fixture";
    reviews_total: number;         // all loaded reviews
    reviews_need_bearing: number;  // clustered after per-review need filter
    review_window_start: string; // ISO-8601
    review_window_end: string;
    reference_date: string;      // === review_window_end
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
