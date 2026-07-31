"""
Seed TEST-ONLY synthetic review fixtures under api/tests/fixtures/.

NEVER writes to data/reviews/ — that path is reserved for real HuggingFace
(or otherwise genuine) review snapshots used in demos.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "api" / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "api"))

import pandas as pd  # noqa: E402


def _rid(pkg: str, text: str, rating: float, date: str) -> str:
    raw = f"{pkg}|{rating}|{date}|{text.strip().lower()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def synthetic_rows() -> list[dict]:
    """Hand-authored themes for offline unit/e2e tests only."""
    pkg = "com.dagr.synthetic"
    themes = {
        "sleep": [
            (
                "The sleep timer does not stop playback when my screen locks at night and I wake up to a dead battery.",
                1,
                "2024-06-12",
            ),
            (
                "Sleep timer fails every single night on bluetooth headphones, audio keeps going for hours.",
                2,
                "2024-07-01",
            ),
            (
                "Cannot get the sleep timer working reliably with wireless earbuds, it ignores the fade out.",
                2,
                "2024-08-15",
            ),
            (
                "Please fix sleep timer, it used to work and now never stops the podcast when time is up.",
                1,
                "2024-09-03",
            ),
            (
                "Sleep timer is broken after the last update, playback continues until morning every time.",
                1,
                "2024-10-20",
            ),
            (
                "I rely on sleep timer and it has not worked for weeks, very frustrating for bedtime listening.",
                2,
                "2025-01-11",
            ),
        ],
        "download": [
            (
                "Downloads get stuck in the queue forever and never finish even on strong wifi connection.",
                1,
                "2024-05-02",
            ),
            (
                "Download queue is broken after update, episodes stay pending with no error message shown.",
                2,
                "2024-06-18",
            ),
            (
                "Episode downloads fail silently and I only notice when I go offline and nothing is there.",
                1,
                "2024-07-22",
            ),
            (
                "Automatic download keeps retrying the same failed episode and blocks the rest of the queue.",
                2,
                "2024-08-30",
            ),
            (
                "Cannot clear stuck downloads, the queue management is unreliable for offline listening.",
                1,
                "2024-11-05",
            ),
            (
                "Large episode downloads abort halfway and restart from zero wasting mobile data every time.",
                2,
                "2025-02-14",
            ),
        ],
        "bluetooth": [
            (
                "Car bluetooth disconnects whenever the phone screen locks and podcast audio drops out.",
                2,
                "2024-04-10",
            ),
            (
                "Android Auto bluetooth keeps dropping podcast playback on every short drive I take.",
                1,
                "2024-05-28",
            ),
            (
                "Bluetooth audio cuts out in the car constantly when switching between maps and podcasts.",
                2,
                "2024-07-09",
            ),
            (
                "Phone call interrupts playback and it never resumes on bluetooth afterward correctly.",
                1,
                "2024-09-19",
            ),
            (
                "Wireless car connection is flaky, I lose my place in the episode after every reconnect.",
                2,
                "2024-12-01",
            ),
            (
                "Bluetooth headphones pause randomly and the app does not resume until I unlock the phone.",
                1,
                "2025-03-08",
            ),
        ],
        "search": [
            (
                "Search cannot find episodes from my subscribed podcasts even when I type the exact title.",
                2,
                "2024-03-15",
            ),
            (
                "Need better search across episode titles and show notes, current search is basically useless.",
                3,
                "2024-06-25",
            ),
            (
                "Searching for guests or topics across podcasts does not work, only finds show names.",
                2,
                "2024-08-08",
            ),
            (
                "Old episodes never show up in search results which makes rediscovering content impossible.",
                2,
                "2024-10-02",
            ),
            (
                "Global episode search beyond subscriptions is missing and I miss that from other podcast apps.",
                3,
                "2025-01-20",
            ),
            (
                "Search ranking is odd and buries relevant episodes under unrelated subscription matches.",
                2,
                "2025-04-04",
            ),
        ],
        "chromecast": [
            (
                "Chromecast support is unreliable, casting often disconnects mid episode without warning.",
                2,
                "2024-05-11",
            ),
            (
                "Cannot cast to my TV reliably, the cast session drops every few minutes during long shows.",
                1,
                "2024-07-17",
            ),
            (
                "Chromecast volume controls from the app do nothing and buffering stalls forever on WiFi.",
                2,
                "2024-09-09",
            ),
            (
                "Please improve casting, it is the main reason I keep considering switching podcast players.",
                2,
                "2024-11-21",
            ),
            (
                "Google Cast integration feels abandoned, other apps handle Chromecast much more smoothly.",
                1,
                "2025-02-02",
            ),
            (
                "Casting resumes at the wrong position after a disconnect which ruins long form listening.",
                2,
                "2025-05-16",
            ),
        ],
        "opml": [
            (
                "OPML import failed halfway and duplicated half my subscriptions with broken feed urls.",
                2,
                "2024-04-22",
            ),
            (
                "Exporting OPML then reimporting loses playback positions and custom queue order completely.",
                3,
                "2024-06-06",
            ),
            (
                "Migrating feeds via OPML was painful, many feeds never refreshed afterward correctly.",
                2,
                "2024-08-19",
            ),
            (
                "OPML import should validate feeds and show which ones failed instead of silent partial success.",
                2,
                "2024-10-30",
            ),
            (
                "Backup and restore via OPML is incomplete for someone switching phones with hundreds of feeds.",
                1,
                "2025-01-07",
            ),
            (
                "Please make OPML sync more robust, I lost several paid podcast feeds during an import.",
                1,
                "2025-03-22",
            ),
        ],
        "ui": [
            (
                "The queue editor is clumsy on small screens and drag reorder regularly drops the wrong episode.",
                3,
                "2024-05-05",
            ),
            (
                "Dark mode still flashes bright white on cold start which hurts eyes during night listening.",
                2,
                "2024-07-28",
            ),
            (
                "Home screen layout wastes space and buries the inbox of new episodes under recommendations.",
                3,
                "2024-09-14",
            ),
            (
                "Notification media controls disappear randomly so I cannot skip chapters from the lock screen.",
                2,
                "2024-12-12",
            ),
            (
                "Tablet layout is basically a stretched phone UI and chapter list is hard to navigate quickly.",
                3,
                "2025-02-25",
            ),
            (
                "Wish the playback screen showed richer chapter art and easier jump controls for long shows.",
                2,
                "2025-04-18",
            ),
        ],
    }
    rows = []
    for items in themes.values():
        for text, rating, date in items:
            rows.append(
                {
                    "review_id": _rid(pkg, text, rating, date),
                    "review_text": text,
                    "rating": float(rating),
                    "created_at": date,
                }
            )
    return rows


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # Guardrail: never touch the real review cache directory
    real_cache = ROOT / "data" / "reviews"
    out = FIXTURE_DIR / "synthetic_reviews.parquet"
    assert real_cache.resolve() not in out.resolve().parents or True
    if out.resolve().is_relative_to(real_cache.resolve()):
        raise SystemExit("Refusing to write fixtures into data/reviews/")

    rows = synthetic_rows()
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(
        {
            "wrote": str(out),
            "rows": len(rows),
            "note": "TEST FIXTURE ONLY — never copy into data/reviews/",
        }
    )


if __name__ == "__main__":
    main()
