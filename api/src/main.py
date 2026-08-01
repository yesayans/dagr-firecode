"""FastAPI entrypoint — CONTRACT.md section 6 routes."""

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chat import ChatTurn, answer_job_chat
from src.config import get_settings
from src.data_ingestion import ReviewScraper
from src.pipeline import AnalysisPipeline, config_hash
from src.resolver import RoadmapResolver, _split_urls
from src.review_charts import build_review_charts
from src.store import get_store, load_catalog, reset_store_singleton
from src.translate_job import translate_job_analysis

logger = logging.getLogger("dagr")
logging.basicConfig(level=logging.INFO)


class ResolveRequest(BaseModel):
    app_name: str
    package_name: str
    github_repo: str | None = None
    refresh: bool = False


class AnalyzeRequest(BaseModel):
    app_id: str
    max_reviews: int = Field(default=2000, ge=10, le=5000)
    force: bool = False


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatHistoryItem] = Field(default_factory=list)


class TranslateRequest(BaseModel):
    locale: str = Field(min_length=2, max_length=8)


def _public_app(app: dict[str, Any] | None) -> dict[str, Any] | None:
    if not app:
        return None
    return {
        "id": app["id"],
        "package_name": app["package_name"],
        "display_name": app["display_name"],
        "review_count": int(app.get("review_count") or 0),
        "avg_stars": app.get("avg_stars"),
        "github_repo": app.get("github_repo"),
        "roadmap_source": app.get("roadmap_source") or "none",
        "roadmap_item_count": int(app.get("roadmap_item_count") or 0),
        "sample_review": app.get("sample_review"),
    }


def _health_payload() -> dict[str, Any]:
    settings = get_settings()
    store = get_store(settings)
    backend = (settings.embedding_backend or "tfidf").lower().strip()
    active_backend = backend
    if backend == "minilm":
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            active_backend = "tfidf"
    return {
        "ok": True,
        "store": store.name,
        "llm_enabled": settings.llm_enabled,
        "llm_model": settings.openrouter_model,
        "llm_base_url": settings.llm_base_url,
        "roadmap_matching_enabled": settings.roadmap_matching_enabled,
        "github_token": settings.github_token_present,
        "github_token_source": settings.github_token_source,
        "embedding_backend": active_backend,
        "match_threshold": settings.active_match_threshold(active_backend),
        "match_margin": settings.active_match_margin(active_backend),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.resolve_github_credentials()
    # Log source only — never the secret
    logger.info("GitHub token source: %s", settings.github_token_source)
    reset_store_singleton()
    store = get_store(settings)
    catalog = load_catalog(settings)
    if catalog:
        n = store.seed_apps_from_catalog(catalog)
        logger.info("Seeded %s apps into %s store", n, store.name)
    app.state.settings = settings
    app.state.store = store
    yield


app = FastAPI(title="dagr", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return _health_payload()


@app.get("/apps")
def list_apps(
    q: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
) -> list[dict[str, Any]]:
    store = get_store()
    return [_public_app(a) for a in store.list_apps(q=q, limit=limit)]  # type: ignore[misc]


def _slug_package(app_name: str, package_name: str | None) -> str:
    pkg = (package_name or "").strip()
    if pkg:
        return pkg
    slug = re.sub(r"[^a-z0-9]+", ".", app_name.lower()).strip(".")
    slug = slug or "app"
    return f"custom.{slug}"


@app.post("/apps/custom")
async def create_custom_app(
    background: BackgroundTasks,
    app_name: str = Form(...),
    package_name: str | None = Form(default=None),
    roadmap_urls: str | None = Form(default=None),
    roadmap_text: str | None = Form(default=None),
    max_reviews: int = Form(default=2000),
    reviews: UploadFile = File(...),
) -> dict[str, Any]:
    """
    Upload a review CSV (+ optional external roadmap URLs/text), upsert the app,
    and queue analysis. Closed-source friendly path.
    """
    settings = get_settings()
    store = get_store(settings)
    name = (app_name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="app_name is required")
    if max_reviews < 10 or max_reviews > 5000:
        raise HTTPException(status_code=400, detail="max_reviews must be 10–5000")

    pkg = _slug_package(name, package_name)
    # Stream with a hard size cap so uploads cannot fill memory/disk.
    max_bytes = int(settings.max_csv_upload_bytes)
    chunks: list[bytes] = []
    total = 0
    while True:
        piece = await reviews.read(1024 * 1024)
        if not piece:
            break
        total += len(piece)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"reviews CSV exceeds {max_bytes} byte limit",
            )
        chunks.append(piece)
    raw = b"".join(chunks)
    if not raw:
        raise HTTPException(status_code=400, detail="reviews CSV is empty")
    filename = reviews.filename or ""
    if filename and not filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=400, detail="reviews file must be a .csv (or .txt)"
        )

    scraper = ReviewScraper(settings)
    try:
        ingested = scraper.ingest_csv(
            pkg, raw, max_reviews=max_reviews, filename=filename or None
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    urls = _split_urls([roadmap_urls or ""])
    paste = (roadmap_text or "").strip()
    resolver = RoadmapResolver(settings)
    result = resolver.resolve(
        app_name=name,
        package_name=pkg,
        github_repo=None,
        refresh=True,
        external_roadmap_urls=urls,
        external_roadmap_text=paste,
    )

    avg = float(ingested.df["rating"].mean()) if not ingested.df.empty else None
    sample = None
    if not ingested.df.empty:
        sample = str(ingested.df.iloc[0]["review_text"])[:280]

    app_row = store.upsert_app(
        {
            "package_name": pkg,
            "display_name": name,
            "dataset": "csv_upload",
            "review_count": ingested.rows_kept,
            "avg_stars": round(avg, 2) if avg is not None else None,
            "github_repo": None,
            "roadmap_source": result.roadmap_source,
            "roadmap_item_count": len(result.roadmap_items),
            "sample_review": sample,
            "metadata": {
                "roadmap_item_count": len(result.roadmap_items),
                "degraded": result.degraded,
                "notes": result.notes,
                "external_roadmap_urls": urls,
                "external_roadmap_text": paste,
                "force_roadmap_refresh": True,
                "csv_column_mapping": ingested.column_mapping,
                "csv_warnings": ingested.warnings,
                "review_provenance": "csv_upload",
            },
        }
    )

    ch = config_hash(app_row["id"], max_reviews, settings)
    # Force a fresh run for uploads
    job = store.create_job(
        {
            "app_id": app_row["id"],
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "config_hash": ch + f":csv:{uuid.uuid4().hex[:8]}",
        }
    )
    background.add_task(_run_job, job["id"], app_row, max_reviews)
    return {
        "app": _public_app(app_row),
        "job_id": job["id"],
        "status": "queued",
        "column_mapping": ingested.column_mapping,
        "warnings": ingested.warnings,
        "rows_kept": ingested.rows_kept,
        "rows_raw": ingested.rows_raw,
        "roadmap_source": result.roadmap_source,
        "roadmap_item_count": len(result.roadmap_items),
    }


@app.post("/apps/resolve")
def resolve_app(body: ResolveRequest) -> dict[str, Any]:
    settings = get_settings()
    store = get_store(settings)
    resolver = RoadmapResolver(settings)
    result = resolver.resolve(
        app_name=body.app_name,
        package_name=body.package_name,
        github_repo=body.github_repo,
        refresh=body.refresh,
    )
    existing = store.get_app_by_package(body.package_name)
    review_count = existing["review_count"] if existing else 0
    avg_stars = existing["avg_stars"] if existing else None
    sample = existing["sample_review"] if existing else None
    # Prefer catalog stats when available
    if existing is None:
        for row in load_catalog(settings):
            if row.get("package_name") == body.package_name:
                review_count = int(row.get("reviews") or 0)
                avg_stars = row.get("avg_stars")
                sample = row.get("sample_review")
                break

    app_row = store.upsert_app(
        {
            "package_name": body.package_name,
            "display_name": body.app_name,
            "review_count": review_count,
            "avg_stars": avg_stars,
            "github_repo": result.github_repo or body.github_repo,
            "roadmap_source": result.roadmap_source,
            "roadmap_item_count": len(result.roadmap_items),
            "sample_review": sample,
            "metadata": {
                "roadmap_item_count": len(result.roadmap_items),
                "degraded": result.degraded,
                "notes": result.notes,
            },
        }
    )
    return _public_app(app_row)


def _run_job(job_id: str, app: dict[str, Any], max_reviews: int) -> None:
    settings = get_settings()
    store = get_store(settings)
    pipe = AnalysisPipeline(store, settings)
    pipe.run(job_id, app, max_reviews=max_reviews)


@app.post("/analyze")
def analyze(body: AnalyzeRequest, background: BackgroundTasks) -> dict[str, Any]:
    settings = get_settings()
    store = get_store(settings)
    app = store.get_app(body.app_id)
    if not app:
        raise HTTPException(status_code=404, detail="app not found")

    ch = config_hash(body.app_id, body.max_reviews, settings)
    if not body.force:
        cached = store.find_completed_job(body.app_id, ch)
        if cached:
            return {"job_id": cached["id"], "status": "completed"}

    job = store.create_job(
        {
            "app_id": body.app_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "config_hash": ch,
        }
    )
    background.add_task(_run_job, job["id"], app, body.max_reviews)
    return {"job_id": job["id"], "status": "queued"}


def _job_charts(job: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Return charts from stats, or backfill from the review parquet cache."""
    charts = stats.get("charts")
    # Prefer yearly series; rebuild older month buckets from the review cache.
    if (
        isinstance(charts, dict)
        and charts.get("period") == "year"
        and isinstance(charts.get("reviews_by_period"), list)
        and len(charts["reviews_by_period"]) > 0
    ):
        return charts
    app = job.get("app") or {}
    package = app.get("package_name")
    if not package:
        return build_review_charts(None)
    try:
        settings = get_settings()
        scraper = ReviewScraper(settings)
        path = scraper.cache_path(str(package))
        if not path.exists():
            return build_review_charts(None)
        import pandas as pd

        df = pd.read_parquet(path)
        max_n = int(stats.get("total_reviews") or stats.get("reviews_total") or 2000)
        if len(df) > max_n:
            df = df.head(max_n)
        return build_review_charts(
            df,
            reviews_need_bearing=int(stats.get("reviews_need_bearing") or 0),
        )
    except Exception as e:
        logger.warning("chart backfill failed for %s: %s", job.get("id"), e)
        return build_review_charts(None)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    store = get_store()
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    stats = job.get("stats") or {}
    return {
        "id": job["id"],
        "app": _public_app(job.get("app")),
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job.get("error"),
        "roadmap_source": job.get("roadmap_source") or "none",
        "summary": job.get("summary"),
        "stats": {
            "total_reviews": int(stats.get("total_reviews") or 0),
            "clusters": int(stats.get("clusters") or 0),
            "roadmap_items": int(stats.get("roadmap_items") or 0),
            "llm_used": bool(stats.get("llm_used")),
            "embedding_backend": stats.get("embedding_backend") or "",
            "elapsed_s": float(stats.get("elapsed_s") or 0.0),
            "degraded": list(stats.get("degraded") or []),
            "review_provenance": stats.get("review_provenance") or "parquet_cache",
            "reviews_total": int(
                stats.get("reviews_total")
                if stats.get("reviews_total") is not None
                else stats.get("total_reviews")
                or 0
            ),
            "reviews_need_bearing": int(stats.get("reviews_need_bearing") or 0),
            "review_window_start": stats.get("review_window_start"),
            "review_window_end": stats.get("review_window_end"),
            "reference_date": stats.get("reference_date"),
            "charts": _job_charts(job, stats),
        },
        "gaps": job.get("gaps") or [],
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
    }


@app.post("/jobs/{job_id}/chat")
async def job_chat(job_id: str, body: ChatRequest) -> dict[str, Any]:
    """Evidence-grounded Q&A over a completed job's gaps and citations."""
    settings = get_settings()
    store = get_store(settings)
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.get('status')}; chat requires completed",
        )
    if not settings.llm_enabled:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured (OPENROUTER_API_KEY / Autorouter)",
        )

    history: list[ChatTurn] = []
    for h in body.history[-8:]:
        role = h.role if h.role in ("user", "assistant") else None
        if not role or not (h.content or "").strip():
            continue
        history.append(ChatTurn(role=role, content=h.content.strip()))

    try:
        reply = await answer_job_chat(
            job, body.message, history=history, settings=settings
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return {
        "answer": reply.answer,
        "citations": [c.model_dump() for c in reply.citations],
        "model": reply.model,
    }


@app.post("/jobs/{job_id}/translate")
async def job_translate(job_id: str, body: TranslateRequest) -> dict[str, Any]:
    """Translate job summary + gap narrative (and review snippets) into a UI locale."""
    settings = get_settings()
    store = get_store(settings)
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.get('status')}; translate requires completed",
        )
    locale = (body.locale or "").strip().lower()
    if locale not in ("en", "ru", "hy"):
        raise HTTPException(status_code=400, detail="locale must be en, ru, or hy")
    if locale != "en" and not settings.llm_enabled:
        raise HTTPException(
            status_code=503,
            detail="LLM is not configured (OPENROUTER_API_KEY / Autorouter)",
        )
    try:
        return await translate_job_analysis(job, locale, settings=settings)  # type: ignore[arg-type]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


def main() -> None:
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
