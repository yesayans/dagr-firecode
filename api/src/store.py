"""Persistence: SupabaseStore or LocalJsonStore with identical method surface."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.config import Settings, get_settings


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


@runtime_checkable
class Store(Protocol):
    name: str

    def upsert_app(self, app: dict[str, Any]) -> dict[str, Any]: ...
    def get_app(self, app_id: str) -> dict[str, Any] | None: ...
    def get_app_by_package(self, package_name: str) -> dict[str, Any] | None: ...
    def list_apps(self, q: str | None = None, limit: int = 25) -> list[dict[str, Any]]: ...
    def create_job(self, job: dict[str, Any]) -> dict[str, Any]: ...
    def update_job(self, job_id: str, fields: dict[str, Any]) -> dict[str, Any]: ...
    def write_gaps_with_evidence(
        self, job_id: str, gaps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]: ...
    def get_job(self, job_id: str) -> dict[str, Any] | None: ...
    def find_completed_job(self, app_id: str, config_hash: str) -> dict[str, Any] | None: ...
    def seed_apps_from_catalog(self, catalog: list[dict[str, Any]]) -> int: ...


def _normalize_app(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    return {
        "id": row["id"],
        "package_name": row["package_name"],
        "display_name": row["display_name"],
        "review_count": int(row.get("review_count") or 0),
        "avg_stars": float(row["avg_stars"]) if row.get("avg_stars") is not None else None,
        "github_repo": row.get("github_repo"),
        "roadmap_source": row.get("roadmap_source") or "none",
        "roadmap_item_count": int(
            row.get("roadmap_item_count")
            if row.get("roadmap_item_count") is not None
            else meta.get("roadmap_item_count") or 0
        ),
        "sample_review": row.get("sample_review"),
        "dataset": row.get("dataset") or "sealuzh/app_reviews",
        "metadata": meta,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


class LocalJsonStore:
    """Atomic JSON file store at data/cache/store.json."""

    name = "local"

    def __init__(self, path: Path | None = None, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self.path = path or (settings.data_dir / "cache" / "store.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write(
                {
                    "apps": {},
                    "jobs": {},
                    "gaps": {},
                    "gap_evidence": {},
                    "job_reviews": {},
                }
            )

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=".store-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def upsert_app(self, app: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            pkg = app["package_name"]
            existing = None
            for a in data["apps"].values():
                if a["package_name"] == pkg:
                    existing = a
                    break
            now = _utcnow()
            if existing:
                row = {**existing, **app, "id": existing["id"], "updated_at": now}
            else:
                row = {
                    "id": app.get("id") or _new_id(),
                    "created_at": now,
                    "updated_at": now,
                    **app,
                }
            data["apps"][row["id"]] = row
            self._write(data)
            return _normalize_app(row)

    def get_app(self, app_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._read()["apps"].get(app_id)
            return _normalize_app(row) if row else None

    def get_app_by_package(self, package_name: str) -> dict[str, Any] | None:
        with self._lock:
            for a in self._read()["apps"].values():
                if a["package_name"] == package_name:
                    return _normalize_app(a)
        return None

    def list_apps(self, q: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock:
            apps = list(self._read()["apps"].values())
        if q:
            ql = q.lower()
            apps = [
                a
                for a in apps
                if ql in (a.get("display_name") or "").lower()
                or ql in (a.get("package_name") or "").lower()
            ]
        apps.sort(key=lambda a: int(a.get("review_count") or 0), reverse=True)
        return [_normalize_app(a) for a in apps[:limit]]

    def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            now = _utcnow()
            row = {
                "id": job.get("id") or _new_id(),
                "app_id": job["app_id"],
                "status": job.get("status") or "queued",
                "stage": job.get("stage") or "queued",
                "progress": int(job.get("progress") or 0),
                "error": job.get("error"),
                "roadmap_snapshot": job.get("roadmap_snapshot") or {},
                "summary": job.get("summary"),
                "stats": job.get("stats")
                or {
                    "total_reviews": 0,
                    "clusters": 0,
                    "roadmap_items": 0,
                    "llm_used": False,
                    "embedding_backend": "",
                    "elapsed_s": 0.0,
                    "degraded": [],
                },
                "config_hash": job.get("config_hash"),
                "created_at": now,
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "gaps": [],
            }
            data["jobs"][row["id"]] = row
            self._write(data)
            return deepcopy(row)

    def update_job(self, job_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            row = data["jobs"].get(job_id)
            if not row:
                raise KeyError(f"job not found: {job_id}")
            row.update(fields)
            data["jobs"][job_id] = row
            self._write(data)
            return deepcopy(row)

    def write_gaps_with_evidence(
        self, job_id: str, gaps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            if job_id not in data["jobs"]:
                raise KeyError(f"job not found: {job_id}")
            # Drop prior gaps for this job
            old_gap_ids = [gid for gid, g in data["gaps"].items() if g["job_id"] == job_id]
            for gid in old_gap_ids:
                del data["gaps"][gid]
            data["gap_evidence"] = {
                eid: e
                for eid, e in data["gap_evidence"].items()
                if e["gap_id"] not in old_gap_ids
            }

            written: list[dict[str, Any]] = []
            for g in gaps:
                evidence = g.get("evidence") or []
                if not evidence:
                    continue
                gap_id = g.get("id") or _new_id()
                metrics = dict(g.get("metrics") or {})
                need_source = g.get("need_source") or "representative_review"
                metrics["need_source"] = need_source
                gap_row = {
                    "id": gap_id,
                    "job_id": job_id,
                    "rank": g["rank"],
                    "need": g["need"],
                    "one_sentence_summary": g.get("one_sentence_summary") or "",
                    "confidence": g["confidence"],
                    "confidence_rationale": g.get("confidence_rationale") or "",
                    "verdict": g["verdict"],
                    "latent_reasoning": g.get("latent_reasoning") or "",
                    "need_source": need_source,
                    "metrics": metrics,
                    "created_at": _utcnow(),
                }
                data["gaps"][gap_id] = gap_row
                ev_rows = []
                for ev in evidence:
                    eid = _new_id()
                    ev_row = {
                        "id": eid,
                        "gap_id": gap_id,
                        "evidence_id": ev["evidence_id"],
                        "source_type": ev["source_type"],
                        "title": ev.get("title"),
                        "snippet": ev.get("snippet"),
                        "url": ev.get("url"),
                        "payload": ev.get("payload") or {},
                        "created_at": _utcnow(),
                    }
                    data["gap_evidence"][eid] = ev_row
                    ev_rows.append(
                        {
                            "evidence_id": ev_row["evidence_id"],
                            "source_type": ev_row["source_type"],
                            "title": ev_row["title"],
                            "snippet": ev_row["snippet"],
                            "url": ev_row["url"],
                            "payload": ev_row["payload"],
                        }
                    )
                out = {**gap_row, "evidence": ev_rows}
                written.append(out)

            data["jobs"][job_id]["gaps"] = [g["id"] for g in written]
            self._write(data)
            return written

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            row = data["jobs"].get(job_id)
            if not row:
                return None
            app = data["apps"].get(row["app_id"])
            gap_ids = row.get("gaps") or []
            # Also find by job_id scan in case gaps list stale
            gaps_out: list[dict[str, Any]] = []
            for g in data["gaps"].values():
                if g["job_id"] != job_id:
                    continue
                if gap_ids and g["id"] not in gap_ids:
                    continue
                evidence = [
                    {
                        "evidence_id": e["evidence_id"],
                        "source_type": e["source_type"],
                        "title": e.get("title"),
                        "snippet": e.get("snippet"),
                        "url": e.get("url"),
                        "payload": e.get("payload") or {},
                    }
                    for e in data["gap_evidence"].values()
                    if e["gap_id"] == g["id"]
                ]
                gaps_out.append(
                    {
                        "id": g["id"],
                        "rank": g["rank"],
                        "need": g["need"],
                        "one_sentence_summary": g.get("one_sentence_summary") or "",
                        "verdict": g["verdict"],
                        "confidence": float(g["confidence"]),
                        "confidence_rationale": g.get("confidence_rationale") or "",
                        "latent_reasoning": g.get("latent_reasoning") or "",
                        "need_source": g.get("need_source") or "representative_review",
                        "metrics": g.get("metrics") or {},
                        "evidence": evidence,
                    }
                )
            gaps_out.sort(key=lambda x: x["rank"])
            return {
                "id": row["id"],
                "app": _normalize_app(app) if app else None,
                "status": row["status"],
                "stage": row.get("stage") or "queued",
                "progress": int(row.get("progress") or 0),
                "error": row.get("error"),
                "roadmap_source": (row.get("roadmap_snapshot") or {}).get(
                    "roadmap_source",
                    (app or {}).get("roadmap_source") or "none",
                ),
                "summary": row.get("summary"),
                "stats": row.get("stats")
                or {
                    "total_reviews": 0,
                    "clusters": 0,
                    "roadmap_items": 0,
                    "llm_used": False,
                    "embedding_backend": "",
                    "elapsed_s": 0.0,
                    "degraded": [],
                },
                "gaps": gaps_out,
                "created_at": row.get("created_at"),
                "completed_at": row.get("completed_at"),
                "config_hash": row.get("config_hash"),
                "roadmap_snapshot": row.get("roadmap_snapshot") or {},
            }

    def find_completed_job(self, app_id: str, config_hash: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read()
            candidates = [
                j
                for j in data["jobs"].values()
                if j.get("app_id") == app_id
                and j.get("config_hash") == config_hash
                and j.get("status") == "completed"
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda j: j.get("completed_at") or "", reverse=True)
        return self.get_job(candidates[0]["id"])

    def seed_apps_from_catalog(self, catalog: list[dict[str, Any]]) -> int:
        n = 0
        for row in catalog:
            pkg = row.get("package_name")
            if not pkg:
                continue
            display = row.get("display_name") or row.get("app_name") or _display_from_package(pkg)
            self.upsert_app(
                {
                    "package_name": pkg,
                    "display_name": display,
                    "review_count": int(row.get("reviews") or row.get("review_count") or 0),
                    "avg_stars": row.get("avg_stars"),
                    "github_repo": row.get("github_repo") or None,
                    "roadmap_source": row.get("roadmap_source") or "none",
                    "sample_review": row.get("sample_review"),
                    "dataset": row.get("dataset") or "sealuzh/app_reviews",
                    "metadata": {
                        "roadmap_item_count": int(row.get("roadmap_item_count") or 0),
                        **{
                            k: row[k]
                            for k in ("gh_stars", "gh_open_issues", "likely_applicable")
                            if k in row
                        },
                    },
                    "roadmap_item_count": int(row.get("roadmap_item_count") or 0),
                }
            )
            n += 1
        return n


class SupabaseStore:
    """Supabase-backed store using the service role key."""

    name = "supabase"

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        from supabase import create_client

        self._client = create_client(settings.supabase_url, settings.supabase_service_key)

    def upsert_app(self, app: dict[str, Any]) -> dict[str, Any]:
        meta = dict(app.get("metadata") or {})
        if "roadmap_item_count" in app:
            meta["roadmap_item_count"] = app["roadmap_item_count"]
        payload = {
            "package_name": app["package_name"],
            "display_name": app["display_name"],
            "review_count": int(app.get("review_count") or 0),
            "avg_stars": app.get("avg_stars"),
            "github_repo": app.get("github_repo"),
            "roadmap_source": app.get("roadmap_source") or "none",
            "sample_review": app.get("sample_review"),
            "dataset": app.get("dataset") or "sealuzh/app_reviews",
            "metadata": meta,
            "updated_at": _utcnow(),
        }
        if app.get("id"):
            payload["id"] = app["id"]
        res = (
            self._client.table("apps")
            .upsert(payload, on_conflict="package_name")
            .execute()
        )
        return _normalize_app(res.data[0])

    def get_app(self, app_id: str) -> dict[str, Any] | None:
        res = self._client.table("apps").select("*").eq("id", app_id).limit(1).execute()
        return _normalize_app(res.data[0]) if res.data else None

    def get_app_by_package(self, package_name: str) -> dict[str, Any] | None:
        res = (
            self._client.table("apps")
            .select("*")
            .eq("package_name", package_name)
            .limit(1)
            .execute()
        )
        return _normalize_app(res.data[0]) if res.data else None

    def list_apps(self, q: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        query = (
            self._client.table("apps")
            .select("*")
            .order("review_count", desc=True)
            .limit(limit)
        )
        if q:
            # PostgREST or-filter on ilike
            query = query.or_(f"display_name.ilike.%{q}%,package_name.ilike.%{q}%")
        res = query.execute()
        return [_normalize_app(r) for r in (res.data or [])]

    def create_job(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "app_id": job["app_id"],
            "status": job.get("status") or "queued",
            "stage": job.get("stage") or "queued",
            "progress": int(job.get("progress") or 0),
            "error": job.get("error"),
            "roadmap_snapshot": job.get("roadmap_snapshot") or {},
            "summary": job.get("summary"),
            "stats": job.get("stats")
            or {
                "total_reviews": 0,
                "clusters": 0,
                "roadmap_items": 0,
                "llm_used": False,
                "embedding_backend": "",
                "elapsed_s": 0.0,
                "degraded": [],
            },
            "config_hash": job.get("config_hash"),
        }
        if job.get("id"):
            payload["id"] = job["id"]
        res = self._client.table("analysis_jobs").insert(payload).execute()
        row = res.data[0]
        return {**row, "gaps": []}

    def update_job(self, job_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "status",
            "stage",
            "progress",
            "error",
            "roadmap_snapshot",
            "summary",
            "stats",
            "config_hash",
            "started_at",
            "completed_at",
        }
        payload = {k: v for k, v in fields.items() if k in allowed}
        res = (
            self._client.table("analysis_jobs")
            .update(payload)
            .eq("id", job_id)
            .execute()
        )
        return res.data[0]

    def write_gaps_with_evidence(
        self, job_id: str, gaps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # Clear previous
        existing = (
            self._client.table("gaps").select("id").eq("job_id", job_id).execute()
        )
        for g in existing.data or []:
            self._client.table("gap_evidence").delete().eq("gap_id", g["id"]).execute()
        self._client.table("gaps").delete().eq("job_id", job_id).execute()

        written: list[dict[str, Any]] = []
        for g in gaps:
            evidence = g.get("evidence") or []
            if not evidence:
                continue
            metrics = dict(g.get("metrics") or {})
            metrics["need_source"] = g.get("need_source") or "representative_review"
            gap_payload = {
                "job_id": job_id,
                "rank": g["rank"],
                "need": g["need"],
                "one_sentence_summary": g.get("one_sentence_summary") or "",
                "confidence": g["confidence"],
                "confidence_rationale": g.get("confidence_rationale") or "",
                "verdict": g["verdict"],
                "latent_reasoning": g.get("latent_reasoning") or "",
                "metrics": metrics,
            }
            gres = self._client.table("gaps").insert(gap_payload).execute()
            gap_row = gres.data[0]
            ev_out = []
            for ev in evidence:
                ev_payload = {
                    "gap_id": gap_row["id"],
                    "evidence_id": ev["evidence_id"],
                    "source_type": ev["source_type"],
                    "title": ev.get("title"),
                    "snippet": ev.get("snippet"),
                    "url": ev.get("url"),
                    "payload": ev.get("payload") or {},
                }
                eres = self._client.table("gap_evidence").insert(ev_payload).execute()
                er = eres.data[0]
                ev_out.append(
                    {
                        "evidence_id": er["evidence_id"],
                        "source_type": er["source_type"],
                        "title": er.get("title"),
                        "snippet": er.get("snippet"),
                        "url": er.get("url"),
                        "payload": er.get("payload") or {},
                    }
                )
            written.append(
                {
                    "id": gap_row["id"],
                    "rank": gap_row["rank"],
                    "need": gap_row["need"],
                    "one_sentence_summary": gap_row.get("one_sentence_summary") or "",
                    "verdict": gap_row["verdict"],
                    "confidence": float(gap_row["confidence"]),
                    "confidence_rationale": gap_row.get("confidence_rationale") or "",
                    "latent_reasoning": gap_row.get("latent_reasoning") or "",
                    "need_source": (gap_row.get("metrics") or {}).get(
                        "need_source", "representative_review"
                    ),
                    "metrics": gap_row.get("metrics") or {},
                    "evidence": ev_out,
                }
            )
        return written

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        res = (
            self._client.table("analysis_jobs")
            .select("*")
            .eq("id", job_id)
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        row = res.data[0]
        app = self.get_app(row["app_id"])
        gaps_res = (
            self._client.table("gaps")
            .select("*")
            .eq("job_id", job_id)
            .order("rank")
            .execute()
        )
        gaps_out = []
        for g in gaps_res.data or []:
            ev_res = (
                self._client.table("gap_evidence")
                .select("*")
                .eq("gap_id", g["id"])
                .execute()
            )
            evidence = [
                {
                    "evidence_id": e["evidence_id"],
                    "source_type": e["source_type"],
                    "title": e.get("title"),
                    "snippet": e.get("snippet"),
                    "url": e.get("url"),
                    "payload": e.get("payload") or {},
                }
                for e in (ev_res.data or [])
            ]
            m = g.get("metrics") or {}
            gaps_out.append(
                {
                    "id": g["id"],
                    "rank": g["rank"],
                    "need": g["need"],
                    "one_sentence_summary": g.get("one_sentence_summary") or "",
                    "verdict": g["verdict"],
                    "confidence": float(g["confidence"]),
                    "confidence_rationale": g.get("confidence_rationale") or "",
                    "latent_reasoning": g.get("latent_reasoning") or "",
                    "need_source": m.get("need_source") or "representative_review",
                    "metrics": m,
                    "evidence": evidence,
                }
            )
        snap = row.get("roadmap_snapshot") or {}
        return {
            "id": row["id"],
            "app": app,
            "status": row["status"],
            "stage": row.get("stage") or "queued",
            "progress": int(row.get("progress") or 0),
            "error": row.get("error"),
            "roadmap_source": snap.get("roadmap_source")
            or (app or {}).get("roadmap_source")
            or "none",
            "summary": row.get("summary"),
            "stats": row.get("stats")
            or {
                "total_reviews": 0,
                "clusters": 0,
                "roadmap_items": 0,
                "llm_used": False,
                "embedding_backend": "",
                "elapsed_s": 0.0,
                "degraded": [],
            },
            "gaps": gaps_out,
            "created_at": row.get("created_at"),
            "completed_at": row.get("completed_at"),
            "config_hash": row.get("config_hash"),
            "roadmap_snapshot": snap,
        }

    def find_completed_job(self, app_id: str, config_hash: str) -> dict[str, Any] | None:
        res = (
            self._client.table("analysis_jobs")
            .select("id")
            .eq("app_id", app_id)
            .eq("config_hash", config_hash)
            .eq("status", "completed")
            .order("completed_at", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        return self.get_job(res.data[0]["id"])

    def seed_apps_from_catalog(self, catalog: list[dict[str, Any]]) -> int:
        n = 0
        for row in catalog:
            pkg = row.get("package_name")
            if not pkg:
                continue
            display = row.get("display_name") or row.get("app_name") or _display_from_package(pkg)
            self.upsert_app(
                {
                    "package_name": pkg,
                    "display_name": display,
                    "review_count": int(row.get("reviews") or row.get("review_count") or 0),
                    "avg_stars": row.get("avg_stars"),
                    "github_repo": row.get("github_repo") or None,
                    "roadmap_source": row.get("roadmap_source") or "none",
                    "sample_review": row.get("sample_review"),
                    "dataset": row.get("dataset") or "sealuzh/app_reviews",
                    "metadata": {
                        "roadmap_item_count": int(row.get("roadmap_item_count") or 0),
                    },
                    "roadmap_item_count": int(row.get("roadmap_item_count") or 0),
                }
            )
            n += 1
        return n


def _display_from_package(package_name: str) -> str:
    tail = package_name.rsplit(".", 1)[-1]
    # Camel-ish: antennapod → AntennaPod-ish fallback
    if tail.lower() == "antennapod":
        return "AntennaPod"
    if tail.lower() == "uhabits":
        return "Loop Habit Tracker"
    if tail.lower() == "kerneladiutor":
        return "Kernel Adiutor"
    return tail[:1].upper() + tail[1:] if tail else package_name


def probe_supabase(settings: Settings) -> bool:
    if not settings.supabase_url or not settings.supabase_service_key:
        return False
    try:
        from supabase import create_client

        client = create_client(settings.supabase_url, settings.supabase_service_key)
        client.table("apps").select("id").limit(1).execute()
        return True
    except Exception:
        return False


_store_singleton: Store | None = None
_store_lock = threading.Lock()


def get_store(settings: Settings | None = None, force_local: bool = False) -> Store:
    global _store_singleton
    settings = settings or get_settings()
    with _store_lock:
        if _store_singleton is not None and not force_local:
            return _store_singleton
        if not force_local and probe_supabase(settings):
            _store_singleton = SupabaseStore(settings)
        else:
            _store_singleton = LocalJsonStore(settings=settings)
        return _store_singleton


def reset_store_singleton() -> None:
    global _store_singleton
    with _store_lock:
        _store_singleton = None


def load_catalog(settings: Settings | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    path = settings.data_dir / "discovery" / "candidates_sealuzh.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    # Enrich display names from known roadmaps
    name_map = {
        "de.danoeh.antennapod": "AntennaPod",
        "org.isoron.uhabits": "Loop Habit Tracker",
        "com.grarak.kerneladiutor": "Kernel Adiutor",
    }
    for r in rows:
        pkg = r.get("package_name") or ""
        r.setdefault("display_name", name_map.get(pkg) or _display_from_package(pkg))
    return rows


def main() -> None:
    settings = get_settings()
    store = get_store(settings)
    catalog = load_catalog(settings)
    n = store.seed_apps_from_catalog(catalog)
    apps = store.list_apps(limit=3)
    print({"store": store.name, "seeded": n, "sample": apps})


if __name__ == "__main__":
    main()
