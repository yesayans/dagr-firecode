# dagr API

Backend for the Silent Stakeholder / dagr hackathon project. Cross-references app-store reviews against a product roadmap and surfaces latent unmet needs.

See [`docs/CONTRACT.md`](../docs/CONTRACT.md) for HTTP shapes, verdict rules, and confidence formula.

## Setup (Windows PowerShell)

```powershell
cd C:\Users\Grigor\Desktop\Hackathon\api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
# Optional MiniLM backend:
# pip install -r requirements-embeddings.txt
copy .env.example .env
# Edit .env if you have OPENROUTER_API_KEY / GITHUB_TOKEN / Supabase keys
```

Seed **test-only** synthetic fixtures (never writes to `data/reviews/`):

```powershell
cd C:\Users\Grigor\Desktop\Hackathon
.\api\.venv\Scripts\python.exe scripts\seed_test_fixtures.py
```

Real demo review caches live under `data/reviews/{package}.parquet` and must come from
HuggingFace (`ReviewScraper.fetch_reviews(..., force_refresh=True)`). Job `stats.review_provenance`
is `hf` | `parquet_cache` | `fixture` — never attribute `fixture` quotes to real users.

## Run the server

```powershell
cd C:\Users\Grigor\Desktop\Hackathon\api
.\.venv\Scripts\Activate.ps1
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Vertical-slice smoke (PowerShell):

```powershell
$resolve = Invoke-RestMethod -Method POST http://127.0.0.1:8000/apps/resolve -ContentType 'application/json' -Body '{"app_name":"AntennaPod","package_name":"de.danoeh.antennapod","github_repo":null,"refresh":false}'
$analyze = Invoke-RestMethod -Method POST http://127.0.0.1:8000/analyze -ContentType 'application/json' -Body (@{app_id=$resolve.id; max_reviews=200; force=$true} | ConvertTo-Json)
Start-Sleep -Seconds 8
Invoke-RestMethod "http://127.0.0.1:8000/jobs/$($analyze.job_id)" | ConvertTo-Json -Depth 8
```

## Tests

```powershell
cd C:\Users\Grigor\Desktop\Hackathon\api
.\.venv\Scripts\Activate.ps1
pytest -q
```

## Module smoke entrypoints

From `api/` with the venv active:

```powershell
python -m src.config
python -m src.store
python -m src.resolver
python -m src.data_ingestion
python -m src.embedding_engine
python -m src.gap_analyzer
python -m src.llm_extractor
python -m src.pipeline
```

## Offline fallbacks

| Dependency | Fallback | Visible in |
|---|---|---|
| HuggingFace reviews | `data/reviews/{package}.parquet` | `stats.degraded` |
| GitHub API | `data/roadmaps/*.json` cache + relevance filter | `/health.github_token`, `stats.degraded` |
| OpenRouter LLM | deterministic template extractor (`llm_used: false`) | `/health.llm_enabled` |
| Supabase | `data/cache/store.json` (`LocalJsonStore`) | `/health.store` |
| MiniLM embeddings | TF-IDF+SVD | `/health.embedding_backend` |

## Embedding thresholds

- `ROADMAP_MATCHING_ENABLED` default `false` — null-model test showed lexical
  matching cannot separate real apps from controls; gaps emit as `UNVERIFIED`
- Matcher code (review-level agreement + `NULL_PERCENTILE=95`) remains for a
  future embedding backend; see `scripts/null_test_validation.py`
- `MATCH_THRESHOLD_TFIDF` / `MATCH_MARGIN_*` apply only when matching is re-enabled

Recalibrate against live GitHub data (rewrites the regression fixture):

```powershell
cd C:\Users\Grigor\Desktop\Hackathon
.\api\.venv\Scripts\python.exe scripts\calibrate_retrieval.py
```

The TF-IDF threshold is a **recall floor**, not a precision guarantee: it admits all
five labelled true matches (min true top-1 ≈ `0.181`), but an ambiguous probe still
reaches ≈ `0.191`. Ranking (top-1 + margin) does the disambiguation. See
[`docs/CONTRACT.md`](../docs/CONTRACT.md) section 3.
