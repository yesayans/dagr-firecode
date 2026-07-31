"""
List applicable products from review datasets.

Works for ANY app:
  - Prefer GitHub issues/milestones when a repo is known
  - Otherwise (or also) resolve roadmap from the public web:
    current features, plans/changelogs, interview promises not clearly shipped

Usage:
  python scripts/list_applicable_apps.py
  python scripts/list_applicable_apps.py --min-reviews 100 --check-github --top 40
  python scripts/list_applicable_apps.py --min-reviews 200 --resolve-roadmap --web-fallback --top 15
  python scripts/list_applicable_apps.py --dataset playmarket --min-reviews 500 --web-fallback
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "discovery"


# Known F-Droid / OSS package → GitHub repo (extend as needed)
KNOWN_GITHUB = {
    "de.danoeh.antennapod": "AntennaPod/AntennaPod",
    "com.ichi2.anki": "ankidroid/Anki-Android",
    "org.wordpress.android": "wordpress-mobile/WordPress-Android",
    "net.osmand.plus": "osmandapp/OsmAnd",
    "com.termux": "termux/termux-app",
    "com.nextcloud.client": "nextcloud/android",
    "org.wikipedia": "wikimedia/apps-android-wikipedia",
    "com.android.keepass": "bpellin/keepassdroid",
    "org.mozilla.fennec_fdroid": "mozilla-mobile/firefox-android",
    "org.mozilla.firefox": "mozilla-mobile/firefox-android",
    "org.telegram.messenger": "Telegram-FOSS-Team/Telegram-FOSS",
    "org.thoughtcrime.securesms": "signalapp/Signal-Android",
    "org.fdroid.fdroid": "f-droid/fdroidclient",
    "com.fsck.k9": "thunderbird/thunderbird-android",
    "org.mozilla.fennec": "mozilla-mobile/firefox-android",
    "org.videolan.vlc": "videolan/vlc-android",
    "com.simplemobiletools.gallery": "SimpleMobileTools/Simple-Gallery",
    "com.simplemobiletools.calendar": "SimpleMobileTools/Simple-Calendar",
    "com.simplemobiletools.notes": "SimpleMobileTools/Simple-Notes",
    "org.sufficientlysecure.keychain": "open-keychain/open-keychain",
    "org.smssecure.smssecure": "SilenceIM/Silence",
    "im.vector.app": "element-hq/element-android",
    "com.owncloud.android": "owncloud/android",
    "org.torproject.android": "guardianproject/orbot",
    "info.guardianproject.orfox": "guardianproject/Orfox",
    "org.mozilla.klar": "mozilla-mobile/focus-android",
    "org.mozilla.focus": "mozilla-mobile/focus-android",
    "com.duckduckgo.mobile.android": "duckduckgo/Android",
    "org.joinmastodon.android": "mastodon/mastodon-android",
    "com.keylesspalace.tusky": "tuskyapp/Tusky",
    "org.mozilla.fenix": "mozilla-mobile/firefox-android",
    "org.kde.kdeconnect_tp": "KDE/kdeconnect-android",
    "com.github.yeriomin.yalpstore": "yeriomin/YalpStore",
    "org.syncthing.android": "syncthing/syncthing-android",
    "com.nutomic.syncthingandroid": "syncthing/syncthing-android",
    "org.tasks": "tasks/tasks",
    "com.seafile.seadroid2": "haiwen/seadroid",
    "fr.free.nrw.commons": "commons-app/apps-android-commons",
    "org.geometerplus.zlibrary.ui.android": "geometer/FBReaderJ",
    "org.mozilla.fennec_aurora": "mozilla-mobile/firefox-android",
    "ch.protonmail.android": "ProtonMail/proton-mail-android",
    "org.cryptomator": "cryptomator/android",
    "com.kunzisoft.keepass.free": "Kunzisoft/KeePassDX",
    "app.organicmaps": "organicmaps/organicmaps",
    "org.briarproject.briar.android": "briarproject/briar",
    "org.xbmc.kodi": "xbmc/xbmc",
    "org.jitsi.meet": "jitsi/jitsi-meet",
    "org.schabi.newpipe": "TeamNewPipe/NewPipe",
    "org.localsend.localsend_app": "localsend/localsend",
    # Extra high-review sealuzh packages with public GitHub
    "org.ppsspp.ppsspp": "hrydgard/ppsspp",
    "com.frostwire.android": "frostwire/frostwire",
    "net.sourceforge.opencamera": "almalence/OpenCamera",  # often mirrored; check
    "com.google.android.stardroid": "sky-map-team/stardroid",
    "com.google.zxing.client.android": "zxing/zxing",
    "com.watabou.pixeldungeon": "00-Evan/shattered-pixel-dungeon",  # spiritual successor; original is watabou
    "net.nurik.roman.muzei": "romannurik/muzei",
    "org.isoron.uhabits": "iSoron/uhabits",
    "org.xbmc.kore": "xbmc/Kore",
    "org.openintents.filemanager": "openintents/filemanager",
    "com.zegoggles.smssync": "jberkel/sms-backup-plus",
    "com.ringdroid": "google/ringdroid",
    "com.google.android.apps.authenticator2": "google/google-authenticator-android",
    "com.google.android.marvin.talkback": "google/talkback",
    "com.google.android.diskusage": "IvanVolosyuk/diskusage",
    "net.androgames.level": "avinashsivaraman/android-level",  # may be stale
    "at.tomtasche.reader": "andiwand/DocumentViewer",
    "com.totsp.crossword.shortyz": "kebernet/shortyz",
}

# Cached roadmap stats from earlier live API pulls (used when rate-limited)
CACHED_GH_STATS = {
    "AntennaPod/AntennaPod": {"stars": 8042, "open_issues": 344, "milestones_open": 0, "milestones_total": 20},
    "ankidroid/Anki-Android": {"stars": 11464, "open_issues": 357, "milestones_open": 0, "milestones_total": 20},
    "wordpress-mobile/WordPress-Android": {"stars": 3146, "open_issues": 789, "milestones_open": 0, "milestones_total": 20},
    "osmandapp/OsmAnd": {"stars": 5868, "open_issues": 3440, "milestones_open": 5, "milestones_total": 20},
    "nextcloud/android": {"stars": 5482, "open_issues": 1552, "milestones_open": 0, "milestones_total": 20},
    "TeamNewPipe/NewPipe": {"stars": 39159, "open_issues": 1445, "milestones_open": 0, "milestones_total": 8},
    "termux/termux-app": {"stars": 58425, "open_issues": 562, "milestones_open": 3, "milestones_total": 3},
    "wikimedia/apps-android-wikipedia": {"stars": 2991, "open_issues": 41, "milestones_open": 0, "milestones_total": 0},
}


def load_sealuzh_counts(limit: int | None = None) -> dict[str, dict]:
    """Return package_name -> {reviews, stars histogram, sample}."""
    from datasets import load_dataset

    print("Loading sealuzh/app_reviews from Hugging Face…")
    ds = load_dataset("sealuzh/app_reviews", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    counts: Counter[str] = Counter()
    star_sums: defaultdict[str, float] = defaultdict(float)
    samples: dict[str, str] = {}

    for row in ds:
        pkg = row["package_name"]
        counts[pkg] += 1
        star_sums[pkg] += float(row.get("star") or 0)
        if pkg not in samples and row.get("review"):
            samples[pkg] = str(row["review"])[:120].replace("\n", " ")

    out = {}
    for pkg, n in counts.items():
        out[pkg] = {
            "package_name": pkg,
            "reviews": n,
            "avg_stars": round(star_sums[pkg] / n, 2) if n else 0,
            "sample_review": samples.get(pkg, ""),
            "dataset": "sealuzh/app_reviews",
        }
    return out


def load_playmarket_counts(limit: int | None = None) -> dict[str, dict]:
    """Load Play Market 2025 apps via HF CSV files."""
    from datasets import load_dataset

    print("Loading play_market_2025 apps_info + apps_reviews…")
    # Prefer info file for names; count from reviews
    try:
        info = load_dataset(
            "dmytrobuhai/play_market_2025_1m_reviews_500_titles",
            data_files="apps_info.csv",
            split="train",
        )
        reviews = load_dataset(
            "dmytrobuhai/play_market_2025_1m_reviews_500_titles",
            data_files="apps_reviews.csv",
            split="train",
        )
    except Exception as e:
        print(f"Play Market load failed: {e}", file=sys.stderr)
        return {}

    names = {}
    for row in info:
        names[str(row["app_id"])] = row.get("app_name") or str(row["app_id"])

    counts: Counter[str] = Counter()
    star_sums: defaultdict[str, float] = defaultdict(float)
    samples: dict[str, str] = {}

    n_rows = 0
    for row in reviews:
        aid = str(row["app_id"])
        counts[aid] += 1
        star_sums[aid] += float(row.get("review_score") or 0)
        if aid not in samples and row.get("review_text"):
            samples[aid] = str(row["review_text"])[:120].replace("\n", " ")
        n_rows += 1
        if limit and n_rows >= limit:
            break

    out = {}
    for aid, n in counts.items():
        name = names.get(aid, aid)
        out[aid] = {
            "package_name": aid,
            "app_name": name,
            "reviews": n,
            "avg_stars": round(star_sums[aid] / n, 2) if n else 0,
            "sample_review": samples.get(aid, ""),
            "dataset": "play_market_2025",
        }
    return out


def guess_github(package: str) -> str | None:
    if package in KNOWN_GITHUB:
        return KNOWN_GITHUB[package]
    # Heuristic guesses for common OSS patterns
    heuristics = [
        ("antennapod", "AntennaPod/AntennaPod"),
        ("newpipe", "TeamNewPipe/NewPipe"),
        ("anki", "ankidroid/Anki-Android"),
        ("osmand", "osmandapp/OsmAnd"),
        ("termux", "termux/termux-app"),
        ("nextcloud", "nextcloud/android"),
        ("wikipedia", "wikimedia/apps-android-wikipedia"),
        ("wordpress", "wordpress-mobile/WordPress-Android"),
        ("syncthing", "syncthing/syncthing-android"),
        ("k9", "thunderbird/thunderbird-android"),
        ("vlc", "videolan/vlc-android"),
        ("fdroid", "f-droid/fdroidclient"),
        ("organicmaps", "organicmaps/organicmaps"),
        ("keepass", "Kunzisoft/KeePassDX"),
        ("tusky", "tuskyapp/Tusky"),
        ("mastodon", "mastodon/mastodon-android"),
        ("duckduckgo", "duckduckgo/Android"),
        ("torproject", "guardianproject/orbot"),
        ("kdeconnect", "KDE/kdeconnect-android"),
        ("localsend", "localsend/localsend"),
        ("jitsi", "jitsi/jitsi-meet"),
        ("cryptomator", "cryptomator/android"),
        ("briar", "briarproject/briar"),
        ("tasks", "tasks/tasks"),
        ("seadroid", "haiwen/seadroid"),
        ("commons", "commons-app/apps-android-commons"),
        ("element", "element-hq/element-android"),
        ("signal", "signalapp/Signal-Android"),
        ("focus", "mozilla-mobile/focus-android"),
        ("fennec", "mozilla-mobile/firefox-android"),
        ("firefox", "mozilla-mobile/firefox-android"),
        ("uhabits", "iSoron/uhabits"),
        ("ppsspp", "hrydgard/ppsspp"),
        ("opencamera", "almalence/OpenCamera"),
        ("stardroid", "sky-map-team/stardroid"),
        ("muzei", "romannurik/muzei"),
        ("frostwire", "frostwire/frostwire"),
        ("telegram", "Telegram-FOSS-Team/Telegram-FOSS"),
        ("simplemobiletools.gallery", "SimpleMobileTools/Simple-Gallery"),
        ("fbreader", "geometer/FBReaderJ"),
        ("zlibrary.ui.android", "geometer/FBReaderJ"),
        ("kore", "xbmc/Kore"),
        ("smssync", "jberkel/sms-backup-plus"),
        ("zxing", "zxing/zxing"),
        ("talkback", "google/talkback"),
        ("diskusage", "IvanVolosyuk/diskusage"),
        ("authenticator", "google/google-authenticator-android"),
        ("pixeldungeon", "00-Evan/shattered-pixel-dungeon"),
        ("shortyz", "kebernet/shortyz"),
    ]
    low = package.lower()
    for key, repo in heuristics:
        if key in low:
            return repo
    return None


def github_stats(repo: str, session: requests.Session) -> dict:
    """Fetch stars, open issues, milestone counts. Falls back to cache on 403."""
    if repo in CACHED_GH_STATS:
        cached = CACHED_GH_STATS[repo]
        # Still try live; on failure return cache
    try:
        r = session.get(f"https://api.github.com/repos/{repo}", timeout=20)
        if r.status_code == 403 and repo in CACHED_GH_STATS:
            c = CACHED_GH_STATS[repo]
            return {
                "repo": repo,
                "ok": True,
                "cached": True,
                "stars": c["stars"],
                "open_issues": c["open_issues"],
                "milestones_total": c["milestones_total"],
                "milestones_open": c["milestones_open"],
                "html_url": f"https://github.com/{repo}",
            }
        if r.status_code != 200:
            if repo in CACHED_GH_STATS:
                c = CACHED_GH_STATS[repo]
                return {
                    "repo": repo,
                    "ok": True,
                    "cached": True,
                    "stars": c["stars"],
                    "open_issues": c["open_issues"],
                    "milestones_total": c["milestones_total"],
                    "milestones_open": c["milestones_open"],
                    "html_url": f"https://github.com/{repo}",
                }
            return {"repo": repo, "ok": False, "error": f"HTTP {r.status_code}"}
        data = r.json()
        m = session.get(
            f"https://api.github.com/repos/{repo}/milestones",
            params={"state": "all", "per_page": 100},
            timeout=20,
        )
        milestones = m.json() if m.status_code == 200 else []
        open_m = sum(1 for x in milestones if x.get("state") == "open")
        return {
            "repo": repo,
            "ok": True,
            "cached": False,
            "stars": data.get("stargazers_count"),
            "open_issues": data.get("open_issues_count"),
            "milestones_total": len(milestones) if isinstance(milestones, list) else 0,
            "milestones_open": open_m,
            "html_url": data.get("html_url"),
        }
    except Exception as e:
        if repo in CACHED_GH_STATS:
            c = CACHED_GH_STATS[repo]
            return {
                "repo": repo,
                "ok": True,
                "cached": True,
                "stars": c["stars"],
                "open_issues": c["open_issues"],
                "milestones_total": c["milestones_total"],
                "milestones_open": c["milestones_open"],
                "html_url": f"https://github.com/{repo}",
            }
        return {"repo": repo, "ok": False, "error": str(e)}


def print_table(rows: list[dict], title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header = f"{'#':>3}  {'reviews':>8}  {'stars':>5}  {'github':<42}  {'pkg / name'}"
    print(header)
    print("-" * 100)
    for i, row in enumerate(rows, 1):
        gh = row.get("github_repo") or "-"
        if row.get("github_ok"):
            gh = (
                f"{gh} (*{row.get('gh_stars')}, "
                f"iss={row.get('gh_open_issues')}, "
                f"ms={row.get('gh_milestones_open')}/{row.get('gh_milestones_total')})"
            )
        elif row.get("github_repo") and row.get("github_error"):
            gh = f"{row['github_repo']} (!)"
        name = row.get("app_name") or row["package_name"]
        print(
            f"{i:>3}  {row['reviews']:>8}  {row['avg_stars']:>5}  "
            f"{gh[:42]:<42}  {name[:46]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="List applicable Silent Stakeholder apps")
    ap.add_argument(
        "--dataset",
        choices=["sealuzh", "playmarket", "both"],
        default="sealuzh",
        help="Which review dataset to scan (default: sealuzh — best for GitHub OSS match)",
    )
    ap.add_argument("--min-reviews", type=int, default=50, help="Minimum reviews to list")
    ap.add_argument("--top", type=int, default=50, help="Max apps to show / check")
    ap.add_argument("--check-github", action="store_true", help="Hit GitHub API for known/guessed repos")
    ap.add_argument("--limit-rows", type=int, default=None, help="Debug: only read N review rows")
    ap.add_argument("--only-github", action="store_true", help="Only keep apps with a known/guessed GitHub repo")
    ap.add_argument(
        "--only-with-roadmap",
        action="store_true",
        help="Keep apps that have GitHub OR a resolved web roadmap",
    )
    ap.add_argument(
        "--resolve-roadmap",
        action="store_true",
        help="Resolve full product context (GitHub and/or web) for listed apps",
    )
    ap.add_argument(
        "--web-fallback",
        action="store_true",
        help="If no GitHub roadmap, search the web for features/plans/interviews",
    )
    ap.add_argument(
        "--force-web",
        action="store_true",
        help="Always run web roadmap search (even when GitHub exists)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apps: dict[str, dict] = {}

    if args.dataset in ("sealuzh", "both"):
        apps.update(load_sealuzh_counts(args.limit_rows))
    if args.dataset in ("playmarket", "both"):
        apps.update(load_playmarket_counts(args.limit_rows))

    # Enrich with GitHub guesses
    for pkg, meta in apps.items():
        repo = guess_github(pkg)
        meta["github_repo"] = repo
        meta["likely_applicable"] = bool(repo)

    ranked = sorted(apps.values(), key=lambda x: x["reviews"], reverse=True)

    # Always dump FULL package inventory (all review counts)
    all_path = OUT_DIR / f"all_packages_{args.dataset}.csv"
    with all_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "package_name",
                "app_name",
                "reviews",
                "avg_stars",
                "dataset",
                "github_repo",
                "likely_applicable",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        for row in ranked:
            w.writerow(row)
    print(f"Full inventory saved: {all_path} ({len(ranked)} packages)")

    ranked = [a for a in ranked if a["reviews"] >= args.min_reviews]

    if args.only_github:
        ranked = [a for a in ranked if a.get("github_repo")]

    ranked = ranked[: args.top]

    # Apply cache even without --check-github for known repos
    for a in ranked:
        repo = a.get("github_repo")
        if repo and repo in CACHED_GH_STATS and not a.get("github_ok"):
            c = CACHED_GH_STATS[repo]
            a["github_ok"] = True
            a["gh_cached"] = True
            a["gh_stars"] = c["stars"]
            a["gh_open_issues"] = c["open_issues"]
            a["gh_milestones_open"] = c["milestones_open"]
            a["gh_milestones_total"] = c["milestones_total"]
            a["gh_url"] = f"https://github.com/{repo}"

    if args.check_github:
        session = requests.Session()
        session.headers["User-Agent"] = "SilentStakeholder-Discovery/1.0"
        # Optional: GITHUB_TOKEN for higher rate limit
        import os

        token = os.environ.get("GITHUB_TOKEN")
        if token:
            session.headers["Authorization"] = f"Bearer {token}"

        print(f"\nChecking GitHub for {len(ranked)} apps…")
        for a in ranked:
            repo = a.get("github_repo")
            if not repo:
                continue
            stats = github_stats(repo, session)
            if stats.get("ok"):
                a["github_ok"] = True
                a["gh_stars"] = stats["stars"]
                a["gh_open_issues"] = stats["open_issues"]
                a["gh_milestones_open"] = stats["milestones_open"]
                a["gh_milestones_total"] = stats["milestones_total"]
                a["gh_url"] = stats["html_url"]
            else:
                a["github_ok"] = False
                a["github_error"] = stats.get("error")
            time.sleep(0.15)  # be nice to API

    # Optional: resolve roadmap for each listed app (GitHub and/or web)
    if args.resolve_roadmap or args.web_fallback or args.force_web:
        from silent_stakeholder.resolve import resolve_product_context

        print(f"\nResolving product roadmaps for {len(ranked)} apps…")
        enriched: list[dict] = []
        for a in ranked:
            name = a.get("app_name") or a["package_name"]
            try:
                ctx = resolve_product_context(
                    product_id=a["package_name"],
                    display_name=name,
                    package_name=a["package_name"],
                    github_repo=a.get("github_repo"),
                    known_github=KNOWN_GITHUB,
                    reviews=a.get("reviews", 0),
                    avg_stars=float(a.get("avg_stars") or 0),
                    dataset=a.get("dataset", ""),
                    prefer_github=bool(a.get("github_repo")),
                    use_web_fallback=args.web_fallback or args.force_web or not a.get("github_repo"),
                    force_web=args.force_web,
                )
                row = {**a, **ctx.flat_row()}
                row["sample_review"] = a.get("sample_review", "")
                enriched.append(row)
                # Persist full context for gap analysis later
                ctx_path = ROOT / "data" / "roadmaps" / f"{a['package_name']}.json"
                ctx_path.parent.mkdir(parents=True, exist_ok=True)
                ctx_path.write_text(
                    json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    f"  {name[:40]:<40} source={ctx.roadmap_source:<7} "
                    f"features={len(ctx.current_features):2} "
                    f"planned={len(ctx.planned_items):2} "
                    f"promised={len(ctx.promised_unshipped):2}"
                )
            except Exception as e:
                a["roadmap_source"] = "none"
                a["notes"] = f"resolve failed: {e}"
                enriched.append(a)
                print(f"  {name[:40]:<40} ERROR {e}")
            time.sleep(0.2)
        ranked = enriched

        if args.only_with_roadmap:
            ranked = [a for a in ranked if a.get("roadmap_source") not in (None, "", "none")]

    # Split views
    with_gh = [a for a in ranked if a.get("github_repo")]
    without_gh = [a for a in ranked if not a.get("github_repo")]
    with_roadmap = [
        a for a in ranked if a.get("roadmap_source") and a.get("roadmap_source") != "none"
    ]

    print_table(
        with_gh,
        f"CANDIDATES WITH GITHUB MAPPING (reviews >= {args.min_reviews}) - {args.dataset}",
    )
    if not args.only_github:
        print_table(
            without_gh[:30],
            "HIGH-REVIEW APPS WITHOUT GITHUB (use --web-fallback to resolve via internet)",
        )
    if with_roadmap:
        print_table(
            with_roadmap[:40],
            f"RESOLVED ROADMAPS (github|web|hybrid) — {len(with_roadmap)} apps",
        )

    # Save artifacts (generic schema)
    csv_path = OUT_DIR / f"candidates_{args.dataset}.csv"
    json_path = OUT_DIR / f"candidates_{args.dataset}.json"
    fields = [
        "product_id",
        "package_name",
        "app_name",
        "reviews",
        "avg_stars",
        "dataset",
        "roadmap_source",
        "github_repo",
        "likely_applicable",
        "current_features",
        "planned_items",
        "promised_unshipped",
        "evidence_count",
        "evidence_urls",
        "notes",
        "github_ok",
        "gh_stars",
        "gh_open_issues",
        "gh_milestones_open",
        "gh_milestones_total",
        "gh_url",
        "sample_review",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in ranked:
            if "product_id" not in row:
                row = {
                    **row,
                    "product_id": row.get("package_name", ""),
                    "roadmap_source": row.get("roadmap_source")
                    or ("github" if row.get("github_repo") else "none"),
                    "likely_applicable": bool(row.get("github_repo")),
                }
            w.writerow(row)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"\nTotal packages in scan: {len(apps)}")
    print(f"Listed (>={args.min_reviews}): {len(ranked)}")
    print(f"With GitHub mapping: {len(with_gh)}")
    print(f"With resolved roadmap: {len(with_roadmap)}")
    print(
        "\nTip: any app works — GitHub backlog if public, else web features/"
        "plans/interviews via --web-fallback. "
        "Single-app: python scripts/resolve_roadmap.py --name \"App\" --force-web"
    )


if __name__ == "__main__":
    main()
