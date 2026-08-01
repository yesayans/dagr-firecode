# Architecture decision: `main` vs `alternative` → `best-solution`

**Status:** Accepted on branch `best-solution`  
**Date:** 2026-07-31  
**Decision:** Keep **Next.js + FastAPI** (main) as the product shell; selectively port analysis ideas from **Streamlit / `aipm`** (alternative). Do **not** make Streamlit the primary UI.

---

## Phase 1 — Compare

### Product thesis (most important difference)

| | **main (`dagr`)** | **alternative (`aipm`)** |
|---|---|---|
| Question answered | “Which latent needs does the **roadmap** miss or mishandle?” | “Which **hidden needs** live in reviews?” |
| Roadmap | First-class (GitHub / web / none) + gap verdicts | Absent |
| Primary artifact | Gap with verdict + confidence + evidence | Need with hiddenness × confidence + citations |

**Tradeoff:** main differentiates against the hackathon “Silent Stakeholder / reviews × roadmap” framing; alternative is a stronger pure review-mining PM tool. Preferring alternative wholesale would abandon the roadmap story even though matching is currently disabled (all `UNVERIFIED`).

### Architecture & structure

| Dimension | main | alternative |
|---|---|---|
| UI | Next.js 15 App Router (`web/`) | Streamlit multipage (`app/`) |
| Backend | FastAPI HTTP API (`api/`) | Library `src/aipm/` called in-process by Streamlit |
| Contracts | `docs/CONTRACT.md` + TS types | Pydantic `schemas.py` + `ARCHITECTURE.md` / `HANDOFF.md` |
| Persistence | Local JSON + parquet (+ Supabase option) | Repository ABC → SQLite (Postgres swappable) |
| Demo strategy | Live jobs + caching | Precomputed analysis runs emphasized |

**main advantages:** clear client/server boundary; UI can evolve independently; CORS API demoable; richer navigation/i18n/theme path.  
**main disadvantages:** looser package layering than `aipm`; some analysis modules sit flat under `api/src/`.  
**alternative advantages:** hard UI/logic split (`app/` never owns business logic); excellent design docs; offline-batch first.  
**alternative disadvantages:** Streamlit owns session/routing; no reusable HTTP surface for other clients; nested product UX fights the framework.

### Scalability

- **main:** HTTP API scales horizontally for request handling; heavy work is per-job sync/async in one process — fine for hackathon, needs a worker queue later. Parquet cache helps ingest.
- **alternative:** Precompute scales *demo* well (instant pages); live upload path still heavy. Embedding/UMAP/HDBSCAN cost is higher than main’s TF-IDF/MiniLM clustering path.

**Tradeoff:** alternative’s cluster quality stack is heavier and better for research demos; main’s lighter stack is more reliable live.

### Maintainability & separation of concerns

- **alternative wins** on package boundaries and “numbers never invented by the LLM.”
- **main wins** on a single CONTRACT shared with the frontend and explicit roadmap modes/verdict rules.
- Streamlit encourages UI-coupled state; Next.js encourages typed API consumption (already present).

### AI pipeline

| Step | main | alternative |
|---|---|---|
| Filter | Need-bearing lexical filter | Quality + segment units |
| Unit | Whole review | Review *segments* |
| Cluster | Embeddings / TF-IDF path | UMAP → HDBSCAN (+ KMeans fallback), c-TF-IDF |
| LLM role | Latent need + optional confidence blend | Label / need / merge; **confidence deterministic** |
| Roadmap | Match + temporal window (disabled → UNVERIFIED) | None |
| Extra scores | volume/novelty/consistency/severity/spread | support/cohesion/separation/temporal/diversity/grounding + **hiddenness** |
| Grounding | Citation ⊆ provided IDs | Citation audit; drop ungrounded needs |
| Chat | Job evidence pack | Hybrid BM25 + vector RRF over units/clusters/needs |

**Tradeoff:** alternative’s segmentation + hiddenness + grounding discipline are stronger for “hidden need” storytelling; main’s roadmap/temporal story and reconstructable confidence formula are stronger for judge auditability of *gaps*.

### UX & performance

- **main:** App-like navigation, job polling, charts, theme, i18n, evidence chat — better long-term product UX. Live analysis latency is the demo risk.
- **alternative:** Fast dashboards on precomputed runs; Streamlit limits nested navigation, branding control, and multi-surface product growth.

### Code quality & hackathon suitability

- Both have real tests. Alternative’s docs/handoff are unusually strong. Main’s CONTRACT + null-model honesty (matching off) is strong science communication.
- For a **live demo with judges clicking around**, Next.js + API is more controllable. For a **precomputed science demo**, alternative’s philosophy is excellent — and can be adopted without Streamlit.

### Future extensibility

Next.js + FastAPI extends to auth, multi-tenant SaaS, mobile clients, and richer viz. Streamlit extends poorly past internal tools. The **`aipm` library shape** is still worth mimicking inside `api/`.

---

## Phase 2 — Hackathon requirements

| Requirement | main | alternative | Notes |
|---|---|---|---|
| Hidden user needs | Yes (latent need / representative quote) | Stronger (surface → workaround → latent need + hiddenness) | Port hiddenness + workaround fields |
| Confidence score | Reconstructable 5-component + LLM blend | Multi-component; LLM never invents total | Keep main formula; add insight = hiddenness × confidence |
| Evidence trace | Per-gap review (+ roadmap) evidence | Citations + evidence table | Keep main; tighten citation drop later |
| Gap verdict | IGNORED / UNDER-PRIORITIZED / MISUNDERSTOOD / UNVERIFIED | Missing | **Keep main** — core differentiator |
| AI reasoning | `latent_reasoning` + rationale | Cluster/need rationales | Keep; enrich prompts |
| Explainability | ConfidenceBreakdown UI + CONTRACT | Confidence meter + written architecture | Keep Next UI; borrow language |
| Demo quality | Live jobs, catalog, closed-source CSV | Precomputed runs, analytics pages | Adopt precompute *option* on main stack |

### Missing / weak on each

**main:** roadmap matching disabled (all UNVERIFIED); no explicit hiddenness; LLM can blend confidence; ingest drops 5★ (skew); weaker segmentation; flatter module layout.  
**alternative:** no roadmap cross-ref / verdicts; Streamlit UX ceiling; product thesis drift from “gap vs roadmap.”

#### Resolved since this decision

The two lists above record the state on 2026-07-31 and are left unedited. What has
changed since:

- **“no explicit hiddenness”** — resolved in `c914472`. `api/src/hiddenness.py`
  annotates every gap and `insight_score` drives ranking.
- **“ingest drops 5★ (skew)”** — resolved in `499b06c`, and it was worse than
  “skew”. `need_filter.is_need_bearing` already decides what is analysable, and
  decides it better: it keeps a 5★ review voicing a want and rejects pure praise
  even at 4★. Dropping by rating first ran *before* that filter, so it destroyed
  the polite unmet-want signal the product exists to surface, and left the UI
  rendering a 1–5★ histogram whose 5★ bar was permanently zero. The committed
  parquet caches were themselves post-filter, so refreshing them took the demo
  corpus from 2,218 to 5,515 reviews — 60% had been discarded — recovering 246
  reviews with unambiguous want language. Rating is a statistic; need-bearing is
  the analysis filter.
- **Related, found while fixing the above:** a missing rating defaulted to 3.0,
  pinning `severity` at 0.50 and `spread` at 0.00 and freezing 35% of the
  confidence weight, so confidence became a pure function of cluster size.
  Unknown ratings now stay unknown and `compute_confidence` renormalises over
  the components it can actually measure.

Still open from the deferred list: segmentation, optional HDBSCAN backend,
BM25+vector chat retrieval, precomputed demo jobs, package layering under
`api/src/`. Two of those were re-assessed against the current code and are lower
value than they look here: the chat builds a *complete* evidence pack from a
job's gaps rather than retrieving, so hybrid retrieval would add a miss-the-passage
failure mode at a scale that does not need it; and a live job now completes in
~12s, so precompute solves a latency problem this stack does not have.

---

## Phase 3 — Recommendation

### Frontend: stay with Next.js

Streamlit is excellent for internal analytics and precomputed demos, but is the wrong long-term shell for nested product navigation, branding, i18n, and post-hackathon extensibility. **Do not migrate the primary UI to Streamlit.**

### Backend: stay with FastAPI; absorb `aipm` ideas

Keep the HTTP contract and job model. Over time, reshape `api/src` toward `aipm`-like packages (`analysis/`, `preprocess/`, …) without a big-bang rewrite.

### Product thesis: keep main’s roadmap × reviews story

Even while matching is off, modes, temporal window, retrospective validation, and verdict vocabulary remain the hackathon narrative. Alternative’s review-only path becomes the **enriched engine under `UNVERIFIED` / mode `none`**, not a replacement thesis.

### Hybrid (chosen)

| Keep from main | Port from alternative | Discard / defer |
|---|---|---|
| Next.js UI, FastAPI, CONTRACT, jobs, catalog, CSV upload | Hiddenness + insight ranking | Streamlit app as primary UI |
| Gap verdicts + roadmap modes | `surface_complaint` / `workaround` in LLM extract | Full UMAP/HDBSCAN swap (defer — cost/risk) |
| Reconstructable confidence components | Explicit-request markers (auditable) | Replacing confidence formula wholesale |
| Evidence chat on jobs | Precompute-for-demo philosophy (later) | Postgres-first |
| Null-model honesty | Stricter “drop ungrounded” posture (later) | — |
| Theme / i18n / charts | Stronger design-doc culture | — |

**Not chosen:** stay entirely Streamlit — loses extensibility and the existing Next demo surface.  
**Not chosen:** ignore alternative — leaves hidden-need storytelling on the table.

---

## Phase 4 — Migration plan

### Keep

- `web/` Next.js app (pages, GapCard, ConfidenceBreakdown, EvidenceChat, i18n, theme)
- `api/` FastAPI + LocalJsonStore + parquet cache
- `docs/CONTRACT.md` as source of truth (extend, don’t replace)
- Roadmap resolver, temporal window, retrospective validation hooks
- Need-bearing filter, null-model scripts, matching flags

### Discard (for primary product)

- Streamlit `app/` as the user-facing product
- Dual parallel UIs long-term (one demo path only)
- Any implication that LLM alone invents the *ranking* of “most hidden”

### Rewrite / do not rewrite

- **Do not rewrite** clustering stack in this pass (HDBSCAN/UMAP is a later experiment behind a flag).
- **Do not rewrite** confidence weights; add parallel **hiddenness** and **insight_score**.
- **Rewrite lightly** LLM prompt/schema to require surface complaint + workaround.

### Move / port

| From alternative | Into |
|---|---|
| `analysis/confidence.hiddenness` + markers | `api/src/hiddenness.py` |
| Need fields surface/workaround/hiddenness | Gap `metrics` + UI |
| Insight ranking `hiddenness × confidence` | Candidate selection + gap `rank` |
| Citation discipline ideas | Future hardening of `llm_extractor` |
| Precompute script pattern | Future `scripts/precompute_demo_jobs.py` |
| Package layering | Incremental folders under `api/src/` |

### Refactor (incremental, this branch starts it)

1. Add hiddenness annotation in `GapMatrix` (both roadmap and none paths).
2. Rank candidates / extracted gaps by `insight_score`.
3. Extend LLM JSON + GapCard to show surface complaint, workaround, hiddenness.
4. Document decision in this file; CONTRACT sections for new metrics.
5. Later: segmenter module; optional HDBSCAN backend; BM25+vector chat retrieval; precomputed demo jobs.

### Out of scope for first `best-solution` slice

- Porting the entire Streamlit UI
- Replacing MiniLM/TF-IDF with UMAP/HDBSCAN
- Replacing the five-component confidence formula
- Re-enabling roadmap matching without a discriminative matcher

---

## Phase 5 — Implementation notes (this branch)

Branch `best-solution` is cut from `main` and applies the hybrid above: Next.js + FastAPI retained; hiddenness / insight / surface+workaround fields wired through API → UI.
