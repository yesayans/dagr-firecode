# AI Product Manager Assistant — Architecture

Streamlit app that analyses Google Play reviews to surface **hidden user needs**,
with evidence and computed confidence — not sentiment scores.

## 1. Core architectural decisions

| Decision | Choice | Why |
|---|---|---|
| Analysis timing | Offline batch, precomputed | A PM waiting 90s on a spinner is a broken demo |
| Unit of analysis | Review *segments*, not whole reviews | One review contains several complaints; segmenting roughly doubles cluster quality |
| Who produces numbers | Deterministic Python, never the LLM | LLM-invented scores destroy credibility |
| Grounding | Every need cites review IDs; citations validated | Uncited needs are dropped before reaching the UI |
| Storage | Repository interface; Parquet+SQLite default, Postgres swappable | Postgres is a demo liability |
| Contracts | Pydantic models on hour one | Lets the team work in parallel against stubs |

## 2. Project structure

```
ai-pm-assistant/
├── app/                      # Streamlit UI ONLY — no business logic
│   ├── Home.py               # app catalog + selection
│   ├── pages/                # 1_Dashboard 2_Needs 3_Evidence 4_Chat 5_Upload
│   ├── components/           # kpi_cards, trend_charts, cluster_map,
│   │                         # need_card, confidence_meter, evidence_table
│   └── state.py              # typed session_state accessors
├── src/aipm/
│   ├── config.py             # pydantic-settings, thresholds, weights
│   ├── schemas.py            # ALL contracts
│   ├── ingest/               # loaders, validators, normalize
│   ├── preprocess/           # clean, dedupe, language, segment, quality
│   ├── embeddings/           # provider, cache, store
│   ├── clustering/           # reduce, cluster, keywords, representatives, metrics
│   ├── llm/                  # client, structured, prompts/, guards
│   ├── analysis/             # pipeline, stats, trends, needs, confidence,
│   │                         # prioritize, hiddenness
│   ├── chat/                 # retriever, agent
│   ├── storage/              # repository ABC, sqlite_repo, postgres_repo, models
│   └── utils/                # hashing, logging, timing
├── scripts/                  # seed_db.py, precompute.py, eval_clusters.py
├── data/                     # raw/ processed/ cache/embeddings/ runs/ fixtures/
└── tests/
```

**Hard rule:** `app/` imports from `src/aipm/`, never the reverse. No `st.` calls
inside `src/`. The pipeline must be runnable from a plain script.

## 3. Data flow

```
CSV (upload or seed) → ingest → validate → normalize → [reviews]
                                                          ↓
                        clean → dedupe → language filter → segment
                                                          ↓
                                                   [review_units]
                        ┌─────────────────────────────────┤
                        ↓                                 ↓
                 embed (cached)                   stats + trends
                        ↓                         (no LLM — instant)
                 UMAP → HDBSCAN                           │
                        ↓                                 │
                 clusters + c-TF-IDF keywords              │
                        ↓                                 │
                 representative sampling (medoid + MMR)    │
                        ↓                                 │
                 LLM: label + summarise cluster            │
                        ↓                                 │
                 LLM: complaint → latent NEED              │
                        ↓                                 │
                 LLM: cross-cluster merge / dedupe         │
                        ↓                                 │
                 citation validation (drop ungrounded)     │
                        ↓                                 │
                 confidence · hiddenness · priority        │
                        ↓                                 ↓
                   [analysis_run persisted] ←──────────────┘
                        ↓
              Streamlit reads run → Dashboard / Needs / Evidence / Chat
```

The two paths diverge deliberately: **stats and trends never touch the LLM**, so
the dashboard renders even if OpenAI is down.

## 4. AI pipeline

1. **Segmentation** — split reviews into clauses. Drop segments under ~4 tokens
   and pure praise from clustering; keep them in statistics. Retain `review_id`
   so evidence always resolves back to the full review.
2. **Embedding** — `text-embedding-3-small`, batches of 256, disk cache keyed by
   `sha256(text)`. Re-runs should cost ~$0.
3. **Reduction** — UMAP to 5–15 dims, pinned `random_state`. HDBSCAN degrades
   badly in 1536 dims; this step is not optional. Fallback: `TruncatedSVD(50)`.
4. **Clustering** — HDBSCAN, `min_cluster_size = max(15, 0.01 * n_units)`,
   `cluster_selection_method='eom'`. Handle the `-1` noise label explicitly.
   **If `n_clusters < 3`, auto-fall back to KMeans** and show a banner.
5. **Characterisation** — c-TF-IDF for distinguishing keywords (free,
   deterministic). Sample 8–12 representatives: medoid + MMR-diverse, biased
   toward high `helpful_count`. Only these reach the LLM, never the raw cluster.
6. **Need extraction** — prompt must push past the complaint to the
   job-to-be-done. Require three separate fields: (a) surface complaint,
   (b) workaround users describe, (c) latent need. Requiring the *workaround*
   field is a cheap trick that reliably surfaces hidden needs — users describing
   hacks are describing unmet needs.
7. **Cross-cluster synthesis** — merge duplicates, find needs spanning clusters.
   Cross-cluster needs get a hiddenness boost: they're invisible to anyone
   reading reviews linearly.
8. **Guards** — validate every cited review_id exists and is in-cluster. Compute
   cosine(need_statement, cited_review); below threshold the citation is
   fabricated → drop. Zero surviving citations → discard the need. Log the drop
   rate; it's a great thing to show a judge.

## 5. Confidence model (the differentiator)

Six computed components, each 0..1, weighted:

| Component | Computation | Weight |
|---|---|---|
| Support | `log(1+n_units) / log(1+n_units_max)` | 0.25 |
| Cohesion | mean pairwise cosine sim within cluster | 0.20 |
| Separation | normalised distance to nearest other centroid | 0.10 |
| Temporal | fraction of months the theme appears in | 0.15 |
| Diversity | 1 − max near-duplicate group share | 0.15 |
| Grounding | share of citations surviving validation | 0.15 |

Rendered as a stacked bar with per-component hover, plus one plain sentence:
*"High confidence: 214 reviews across 9 months, tightly clustered, all 6
citations verified."* Show low-confidence needs too, labelled as hypotheses.

**Hiddenness** (separate): `1 − (explicit_feature_requests / total_mentions)`,
boosted for cross-cluster needs. Sort the Needs page by
`hiddenness × confidence` — that ordering *is* the product.

## 6. Prioritisation

- **Reach** = share of reviews touching the need
- **Impact** = `app_avg_rating − avg_rating_of_affected_reviews`
- **Confidence** = composite above
- **Value** = reach × impact × confidence

**Effort is deliberately not computed.** No engineering context exists, and a
fabricated estimate is what gets an AI tool distrusted in a real PM meeting.
Give the PM a slider and let the ranking update live.

## 7. Database schema

Postgres-shaped, runs on SQLite. Everything versioned by `run_id`.

```sql
apps(app_id PK, name, description, score, ratings_count,
     downloads_raw, downloads_numeric, categories, source, created_at)

reviews(review_id PK, app_id FK, text, score, review_date, helpful_count,
        lang, is_duplicate, quality_weight)                -- IDX (app_id, review_date)

review_units(unit_id PK, review_id FK, app_id FK, text, position,
             embedding_hash)                               -- IDX (app_id)

embeddings(embedding_hash PK, model, dim, vector)          -- pgvector if Postgres

analysis_runs(run_id PK, app_id FK, params_hash, embed_model, llm_model,
              status, n_reviews, n_units, n_clusters, noise_ratio,
              clustering_fallback, citations_dropped, cost_usd,
              started_at, finished_at, error)
              -- UNIQUE (app_id, params_hash)  ← makes reruns free

clusters(cluster_id PK, run_id FK, label, summary, keywords, size,
         persistence, cohesion, separation, medoid_unit_id, centroid)

cluster_members(cluster_id FK, unit_id FK, membership_prob, PK(cluster_id, unit_id))

needs(need_id PK, run_id FK, statement, underlying_goal, category,
      surface_complaints, workarounds, cluster_ids, hiddenness,
      confidence_total, confidence_components, reach, impact,
      value_score, rank)

need_evidence(id PK, need_id FK, review_id FK, quote, relevance, validated)

chat_sessions(session_id PK, app_id FK, run_id FK, created_at)
chat_messages(id PK, session_id FK, role, content, citations, created_at)
```

## 8. Streamlit pages

- **Home — App Catalog.** Grid of apps with name, category, rating, review count,
  and an "analysis available" badge. Selecting sets `session_state.app_id`/`run_id`.
- **1 · Dashboard.** KPI row, rating distribution, volume + rolling-average
  score time series with annotated inflection points, UMAP 2D cluster scatter
  (coloured by cluster, sized by helpful_count), cluster table.
- **2 · Hidden Needs.** The centrepiece. Cards sorted by hiddenness × confidence:
  statement, underlying goal, surface complaints, confidence meter with
  breakdown, reach/impact/value chips, 2–3 quotes with dates and stars. Below:
  value-vs-effort quadrant with the effort slider. Low-confidence needs in a
  collapsed "Hypotheses — weak evidence" section.
- **3 · Evidence.** Drill-down to every supporting review with full text, score,
  date, helpful count, membership probability. Filters, CSV export. One click
  from every need card. This page is what turns scepticism into trust.
- **4 · Chat.** Scoped to the selected app. Hybrid BM25 + vector retrieval over
  units, cluster summaries and needs. Answers cite review IDs as expandable
  quotes. Refuses to answer beyond retrieved context.
- **5 · Upload.** CSV → validation with per-column diagnostics → preview →
  cost/time estimate → run with `st.status` stage progress.

**Mechanics:** `@st.cache_resource` for the OpenAI client and repository;
`@st.cache_data` keyed by `(app_id, run_id)` for run loads; all cross-page state
behind typed accessors in `app/state.py`. Never run the pipeline inline in a
page render — trigger, persist, then read.

## 9. Implementation plan

| Phase | Time | Deliverable |
|---|---|---|
| 0 · Skeleton | 2h | Contracts, repo ABC, Streamlit shell on fixture data — **a clickable demo with fake data** |
| 1 · Data spine | 2h | Ingest, clean, dedupe, segment, seed_db. Stats + trends live, zero AI |
| 2 · Clustering | 3h | Embeddings + cache, UMAP, HDBSCAN, c-TF-IDF, KMeans fallback |
| 3 · Needs | 3h | LLM client, 3 prompts, guards, confidence, hiddenness, prioritisation |
| 4 · Chat | 2h | Retriever, agent with citations |
| 5 · Precompute & polish | 2h | Run all seeded apps, verify it demos with the network unplugged |

Build the vertical slice for **one** app end-to-end before broadening.
Budget real time for prompt iteration in Phase 3 — the need-extraction prompt
needs 5–10 revisions to stop producing generic restatements of complaints.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Demo wifi dies / rate limits | Precomputed runs committed; app reads from cache |
| HDBSCAN returns all noise | Auto-detect, fall back to KMeans, banner |
| Non-English reviews fragment clusters | Detect, filter to dominant language, state it in UI |
| LLM invents needs | Citation validation; show the drop rate as a feature |
| Bot/duplicate reviews inflate clusters | Near-dup detection feeds the diversity component |
| Cost blowout | Embedding cache + per-run cost meter |

**Cut in this order:** Postgres → chat → upload → UMAP scatter → cross-cluster
synthesis. **Never cut:** evidence drill-down, confidence breakdown. Those two
are the entire argument that this is a PM tool and not a sentiment classifier.
