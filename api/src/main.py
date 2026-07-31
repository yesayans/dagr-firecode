"""FastAPI entrypoint — CONTRACT.md section 6 routes."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.pipeline import AnalysisPipeline, config_hash
from src.resolver import RoadmapResolver
from src.store import get_store, load_catalog, reset_store_singleton

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
        "github_token": settings.github_token_present,
        "embedding_backend": active_backend,
        "match_threshold": settings.active_match_threshold(active_backend),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
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
        },
        "gaps": job.get("gaps") or [],
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
    }


def main() -> None:
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
