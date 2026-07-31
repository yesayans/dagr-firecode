# dagr-firecode

Surfaces latent user needs by mapping app reviews against developer roadmaps or web signals, backed by traceable evidence.

---

**AI Product Manager Assistant** — a Streamlit app that analyses Google Play
reviews to surface **hidden user needs**, with evidence and computed confidence.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the design; this file covers running
the app and the demo precompute.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[umap,local-embeddings,dev]'
cp .env.example .env   # then fill in LLM_BASE_URL / LLM_API_KEY
```

Place `apps_info.csv` and `apps_reviews.csv` in `data/raw/`.

## Running the app

```bash
streamlit run app/main.py
```

Five views, all reading precomputed runs:

| View | URL | What it does |
|---|---|---|
| **Catalogue** | `/` | App catalogue with run-level KPIs, search, category filter, and why each app was selected |
| **Dashboard** | `/dashboard` | KPI row, rating distribution, volume and rating over time, confidence bands, helpful-vote concentration, reach-vs-impact map |
| **Needs & Evidence** | `/details` | Three tabs — ranked need cards with confidence breakdowns, the theme explorer with a segment map, and the full evidence drill-down with CSV export |
| **Chat** | `/chat` | Grounded Q&A over one app, hybrid BM25 + vector retrieval, answers cite review ids |
| **Upload dataset** | `/upload` | Validate → preview → estimate → run, with per-column diagnostics and staged progress |

**Layer rule:** `app/` imports from `src/aipm/`, never the reverse, and there is no
`st.` call anywhere under `src/`. The pipeline runs from a plain script.

### Navigation

Streamlit's built-in page tree is **switched off** — `app/main.py` registers the
views with `st.navigation(..., position="hidden")`, which also makes Streamlit
ignore any `pages/` directory. `app/components/nav.py` draws the sidebar instead.

The built-in tree was a flat list of five siblings, but three of those views only
mean anything *inside* a selected app, so moving between levels looked like
moving between peers. The replacement shows the actual shape:

```
← All applications
─────────────
Google Wallet              ← only once an app is chosen
   Dashboard
   Needs & Evidence
   Chat
─────────────
Upload dataset
```

The current view renders as a disabled item — you cannot navigate to where you
already are, and disabling marks it without relying on colour alone.

### Routing

All navigation happens **in the same tab**, and the selected app travels in the
**query string** rather than living only in `st.session_state`:

```
/Dashboard?app=15&run=run_e8ab18f2bb11d2d6
```

That makes a dashboard URL refresh-safe, bookmarkable and shareable — none of
which hold if the selection is session-only, because a refresh or a new tab is a
new Streamlit session with empty state.

**Every navigation goes through `switch_to()` in `app/state.py`** — there is
exactly one `st.switch_page` call in the codebase, and a test enforces that.
The reason is a sharp edge in the API:

> `st.switch_page(page)` **clears all query parameters** when `query_params` is
> omitted — "all non-embed query parameters are cleared during navigation."

So a bare `st.switch_page("pages/1_Dashboard.py")` silently drops `?app=` and the
destination URL stops identifying anything. `switch_to` always passes them;
`switch_to(page, Selection(None, None))` clears them deliberately, which is what
navigating back to the catalogue should do.

`switch_to(page, Selection(None, None))` clears the session keys as well as the
query string. Dropping only the URL would let `get_selection()` resurrect the app
from session state, and the sidebar would still show an app section on a
catalogue URL that says there is none.

`require_selection()` repairs the URL if it arrives without `?app=` but session
state knows the app. That lives there, not in `get_selection()`, because only
app-scoped views call it — so the catalogue is never stamped with a selection it
does not have.

For reference, Streamlit's four navigation APIs:

| API | Tab | Notes |
|---|---|---|
| `st.switch_page(page, query_params=…)` | same | what this app uses |
| `st.page_link(page, label=…)` | same (new tab for external URLs) | renders an anchor |
| `st.link_button(label, url)` | **always new tab** | no `target` argument |
| Built-in sidebar nav | same | bare href, drops query params — disabled here |

### UI conventions

- **State** goes through typed accessors in `app/state.py`, never raw
  `st.session_state["…"]` strings scattered across pages.
- **Caching** follows the two rules that matter: `@st.cache_resource` for things
  holding a connection or a model (repository, LLM client, embedder, retrieval
  index), `@st.cache_data` for values derived from them, keyed by
  `(app_id, run_id)`.
- **The repository uses one SQLite connection per thread.** Streamlit runs each
  rerun on a script-runner thread and caches the repository across all of them; a
  single shared connection raises `ProgrammingError` on the second page load.
- **Colours are validated, not chosen.** `app/theme.py` holds the palette that
  passed the lightness-band, chroma, CVD-separation, normal-vision and contrast
  checks against these exact surfaces. Changing a hex means re-running that
  validation.
- **Charts follow the data's job**: diverging for the ordered 1–5★ scale, two
  separate charts for volume and rating (never a dual axis), and the *emphasis*
  form for the segment map — with 12–20 themes, twenty hues is confetti, so one
  theme lights up and the rest stay context grey.
- Status never travels as colour alone: confidence bands ship an icon and a label
  (`●●● high confidence · 0.73`), and every chart has a table view.

## Precomputing the demo

The Streamlit app must never run AI analysis during startup. This script does all
of it ahead of time and writes the results into the app's storage.

```bash
python scripts/precompute_demo.py
```

Afterwards the UI reads `data/aipm.db` and renders a dashboard from a single row
read — no embedding, no clustering, no LLM calls, and it works with the network
unplugged.

### What it does

| Step | Service | Notes |
|---|---|---|
| 1. Load + validate | `ingest.loaders`, `ingest.validators` | Streams the 94 MB review CSV in chunks; per-column diagnostics |
| 2. Select demo apps | `demo.selection` | Scored, filtered, category-diverse. Configurable |
| 3. Filter reviews | `ingest.loaders` | One pass for all selected apps, most recent first |
| 4. Clean + validate | `preprocess.*` | Clean, language filter, near-dup flagging, quality weights, segmentation |
| 5. Embed | `embeddings.*` | Pluggable backend, disk-cached by `sha256(text)` |
| 6. Cluster | `clustering.*` | UMAP → HDBSCAN, c-TF-IDF keywords, medoid + MMR representatives |
| 7. Statistics | `analysis.stats`, `analysis.trends` | Counts, ratings, distribution, helpful votes, monthly trends. **Never touches the LLM** |
| 8. Needs | `analysis.needs`, `llm.*` | Per cluster: title, summary, hidden need, confidence + explanation |
| 9. Persist | `storage.sqlite_repo` | Runs, clusters, needs, evidence, and a demo manifest |

The script itself only parses arguments, wires dependencies and reports progress.
Every step above lives in a service and is unit-tested on its own.

### Options

```bash
python scripts/precompute_demo.py --dry-run              # show the selection, compute nothing
python scripts/precompute_demo.py --n-apps 6             # 5..10
python scripts/precompute_demo.py --max-reviews 2000     # cap per app; controls runtime
python scripts/precompute_demo.py --force                # ignore cached runs
python scripts/precompute_demo.py --require-llm          # fail rather than degrade
python scripts/precompute_demo.py --strategy config/demo_strategy.example.json
```

Exit codes: `0` all apps succeeded, `1` some failed, `2` all failed, `3` bad input.

### Selection strategy

Apps are scored on five normalised signals — volume, text quality, recency,
star-level coverage and helpful-vote engagement — then filtered and picked greedily
under a per-category cap. Copy `config/demo_strategy.example.json` and pass it with
`--strategy` to retune without touching code.

Category diversity is a hard constraint rather than a scoring term, so one crowded
category cannot quietly dominate the demo.

## Configuration

Secrets go in `.env` (gitignored), never in source.

### LLM

Any OpenAI-compatible endpoint. Server-side JSON-schema enforcement is **not**
assumed: the shape is specified in the prompt, stripped of markdown fences, parsed,
validated with pydantic, and retried once with the validation error on failure.
Token usage is read from the response, not estimated locally.

```
LLM_BASE_URL=https://autorouter.io/v1
LLM_API_KEY=...
LLM_MODEL=gemini-2.5-flash
HTTP_USER_AGENT=aipm/0.1.0
```

A preflight healthcheck runs before the batch, so a bad key fails in one second
rather than after retry backoff on every cluster of every app.

**`HTTP_USER_AGENT` is load-bearing.** autorouter.io fronts its API with a WAF
that returns `403 Your request was blocked` to anything advertising itself as
`OpenAI/Python*` — the exact agent the SDK sends by default. The same request
succeeds from `curl`. Overriding the header fixes it; the `x-stainless-*` headers
the SDK also sends are not a problem. Set it empty to restore the SDK default.

### Embeddings

Configured independently of the LLM. `EMBED_MODEL` and `EMBED_DIM` are unset by
default so each backend resolves its own identity.

| `EMBED_BACKEND` | Model | Dim |
|---|---|---|
| `local` (default) | `all-MiniLM-L6-v2`, falling back to TF-IDF → TruncatedSVD | 384 / 256 |
| `api` | Any OpenAI-compatible `/embeddings` endpoint | resolved from the response |
| `fixture` | Deterministic hashed vectors — offline, for tests | 128 |

All backends share one disk cache keyed by `sha256(text)`, so re-runs are near-free.

## Two things about this dataset

**It is a quota-capped scrape.** Some apps hold exactly 3,000 reviews per star
level; others (Priceline, Burger King) hold *only* 1-star reviews. A review-derived
average rating is therefore a scraping artefact, not user sentiment. Consequently:

- `OverviewStats.store_score` carries the real store rating separately, and
  `sample_is_quota_capped` flags the artefact for the UI to disclose;
- **impact** is computed against the store rating, not the sample mean, which would
  otherwise sit at exactly 3.0;
- selection rejects apps covering fewer than three star levels.

**Only 42 of the 217 apps have reviews at all.** Selection operates on those 42.

## Division of labour: Python computes, the LLM reasons

| Produced by Python (measured) | Produced by the LLM (language) |
|---|---|
| Cluster membership, size, cohesion, separation, medoid | Cluster title and summary |
| c-TF-IDF keywords | Hidden need, underlying goal, surface complaint, workaround |
| All six confidence components and the total | Qualitative rationale (`confidence.llm_rationale`) |
| Reach, impact, value, rank, hiddenness | Need category (a label, not a number) |
| Citation validity and relevance scores | Which reviews to cite (then validated) |
| Review counts, ratings, distributions, helpful votes, trends | — |

`ClusterInsight` — the only schema the model can write into — exposes exactly one
numeric field, `evidence_strength`, and it is **advisory**: it never reaches a
displayed number. It exists so the pipeline can log divergence between the model's
read and the computed score as a prompt-quality signal.

`tests/test_pipeline.py::TestLlmProducesNoNumbers` enforces this mechanically:
flipping `evidence_strength` from 0.0 to 1.0 must not move any component, every
stored number must be reproducible from measured inputs alone, and the insight
schema must expose exactly one numeric field. Adding a new numeric field for the
model to fill fails the suite.

## Confidence

Six computed components — support, cohesion, separation, temporal spread,
diversity, grounding — combined with the weights in `config.py`. The plain-English
explanation is assembled from the same numbers, so it cannot contradict the meter.
The model's qualitative rationale is carried separately in `llm_rationale`:

```
computed  : High confidence: 2057 supporting review segments, across 12 months,
            moderately clustered, 4 of 4 citations verified.
            Weakest signal: separation (0.24).
llm reason: The reviews consistently point to a breakdown in the core utility of
            the app, with users explicitly reverting to physical alternatives.
```

`ARCHITECTURE.md` requires that the LLM never produce a number, while the demo spec
asks for a per-cluster confidence score. Both are satisfied: the model returns an
advisory `evidence_strength` and a qualitative rationale, the displayed score is
computed in Python, and a divergence above 0.35 between the two is logged as a
prompt-quality signal.

## Tests

```bash
python -m pytest
```

284 tests, fully offline — the `fixture` embedding backend and a stub LLM client
mean no network and no API key are needed. Notable guards:

| Test | What it protects |
|---|---|
| `TestLlmProducesNoNumbers` | Flipping the model's `evidence_strength` must move no component; the insight schema must expose exactly one numeric field |
| `TestThreadSafety` | The repository survives Streamlit's script-runner threads |
| `TestFrameMapping` | The batch loader and the upload page derive identical review ids |
| `TestDominantClusterHandling` | A cluster swallowing the corpus triggers leaf re-selection |
| `TestChatAgent::test_refuses_when_nothing_retrieves` | The agent refuses instead of answering from irrelevant context |
