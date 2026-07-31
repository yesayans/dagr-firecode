"""Resolve + analyze AntennaPod against the live API, wait for completion."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=120.0) as c:
        h = c.get("/health").json()
        print("health:", json.dumps(h, indent=2))
        if not h.get("llm_enabled"):
            print("FAIL: LLM not enabled")
            sys.exit(1)

        app = c.post(
            "/apps/resolve",
            json={
                "app_name": "AntennaPod",
                "package_name": "de.danoeh.antennapod",
                "github_repo": "AntennaPod/AntennaPod",
                "refresh": True,
            },
        ).json()
        print(
            "resolved:",
            {
                "id": app.get("id"),
                "roadmap_source": app.get("roadmap_source"),
                "github_repo": app.get("github_repo"),
                "roadmap_item_count": app.get("roadmap_item_count"),
            },
        )

        job_resp = c.post(
            "/analyze",
            json={"app_id": app["id"], "max_reviews": 2000, "force": True},
        ).json()
        job_id = job_resp["job_id"]
        print("job:", job_id, job_resp.get("status"))

        for _ in range(120):
            job = c.get(f"/jobs/{job_id}").json()
            print(
                f"  stage={job.get('stage')} progress={job.get('progress')} "
                f"status={job.get('status')}"
            )
            if job.get("status") in ("completed", "failed"):
                break
            time.sleep(1.5)
        else:
            print("FAIL: timed out")
            sys.exit(1)

        print("\n=== RESULT ===")
        print("status:", job.get("status"))
        print("error:", job.get("error"))
        print("summary:", job.get("summary"))
        stats = job.get("stats") or {}
        print(
            "stats:",
            {
                k: stats.get(k)
                for k in (
                    "total_reviews",
                    "reviews_need_bearing",
                    "clusters",
                    "roadmap_items",
                    "llm_used",
                    "embedding_backend",
                    "roadmap_source",
                    "degraded",
                    "elapsed_s",
                )
            },
        )
        for g in job.get("gaps") or []:
            print(
                f"\n#{g.get('rank')} [{g.get('verdict')}] "
                f"conf={g.get('confidence')} source={g.get('need_source')}"
            )
            print(f"  need: {g.get('need')}")
            print(f"  summary: {g.get('one_sentence_summary')}")
            print(f"  evidence: {len(g.get('evidence') or [])}")

        dump = Path(__file__).resolve().parents[1] / "data" / "cache" / "last_job.json"
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(json.dumps(job, indent=2), encoding="utf-8")
        print(f"\nwrote {dump}")
        if job.get("status") != "completed":
            sys.exit(1)


if __name__ == "__main__":
    main()
