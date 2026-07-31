"""Rebuild candidates_sealuzh.json from all_packages_sealuzh.csv.

Includes closed-source apps (no GitHub) so search is not OSS-only.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISCOVERY = ROOT / "data" / "discovery"

DISPLAY = {
    "de.danoeh.antennapod": "AntennaPod",
    "org.isoron.uhabits": "Loop Habit Tracker",
    "com.ichi2.anki": "AnkiDroid",
    "org.wordpress.android": "WordPress",
    "org.schabi.newpipe": "NewPipe",
    "com.termux": "Termux",
    "org.wikipedia": "Wikipedia",
    "org.thoughtcrime.securesms": "Signal",
    "com.instagram.android": "Instagram",
    "com.facebook.katana": "Facebook",
    "com.whatsapp": "WhatsApp",
    "com.spotify.music": "Spotify",
    "com.netflix.mediaclient": "Netflix",
    "com.twitter.android": "Twitter / X",
    "com.google.android.youtube": "YouTube",
    "com.google.android.apps.maps": "Google Maps",
    "com.google.android.gms": "Google Play services",
    "com.android.chrome": "Chrome",
    "com.ubercab": "Uber",
    "com.snapchat.android": "Snapchat",
    "com.amazon.mShop.android.shopping": "Amazon Shopping",
    "com.tencent.mm": "WeChat",
    "com.viber.voip": "Viber",
    "com.dropbox.android": "Dropbox",
    "com.evernote": "Evernote",
    "com.adobe.reader": "Adobe Acrobat Reader",
    "com.microsoft.office.outlook": "Outlook",
    "com.slack": "Slack",
    "com.discord": "Discord",
    "com.pinterest": "Pinterest",
    "com.linkedin.android": "LinkedIn",
    "com.google.android.apps.photos": "Google Photos",
    "com.google.android.gm": "Gmail",
    "org.telegram.messenger": "Telegram",
    "com.fsck.k9": "K-9 Mail",
    "net.osmand.plus": "OsmAnd",
    "com.duckduckgo.mobile.android": "DuckDuckGo",
    "org.torproject.android": "Orbot",
    "com.nextcloud.client": "Nextcloud",
    "org.videolan.vlc": "VLC",
    "org.mozilla.firefox": "Firefox",
    "org.mozilla.focus": "Firefox Focus",
    "com.google.android.marvin.talkback": "Android Accessibility Suite",
    "com.frostwire.android": "FrostWire",
    "org.ppsspp.ppsspp": "PPSSPP",
    "org.xbmc.kore": "Kore",
    "com.google.android.apps.authenticator2": "Google Authenticator",
    "com.google.zxing.client.android": "Barcode Scanner",
    "net.nurik.roman.muzei": "Muzei",
    "com.simplemobiletools.gallery": "Simple Gallery",
    "com.simplemobiletools.calendar": "Simple Calendar",
}


def display_from_pkg(pkg: str) -> str:
    if pkg in DISPLAY:
        return DISPLAY[pkg]
    tail = pkg.rsplit(".", 1)[-1]
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", tail)
    if parts:
        return " ".join(p[:1].upper() + p[1:] for p in parts)
    return tail[:1].upper() + tail[1:] if tail else pkg


def load_known_github() -> dict[str, str]:
    known: dict[str, str] = {}
    cand_path = DISCOVERY / "candidates_sealuzh.json"
    if cand_path.exists():
        for row in json.loads(cand_path.read_text(encoding="utf-8")):
            pkg = row.get("package_name")
            gh = (row.get("github_repo") or "").strip()
            if pkg and gh:
                known[pkg] = gh

    script = ROOT / "scripts" / "list_applicable_apps.py"
    text = script.read_text(encoding="utf-8")
    for m in re.finditer(
        r'"([a-zA-Z0-9_.]+)":\s*"([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)"',
        text,
    ):
        pkg, repo = m.group(1), m.group(2)
        if "." in pkg:
            known.setdefault(pkg, repo)
    return known


def main() -> None:
    known = load_known_github()
    rows: list[dict] = []
    with (DISCOVERY / "all_packages_sealuzh.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pkg = r["package_name"]
            gh = (r.get("github_repo") or "").strip() or known.get(pkg)
            name = (r.get("app_name") or "").strip() or display_from_pkg(pkg)
            rows.append(
                {
                    "package_name": pkg,
                    "display_name": name,
                    "reviews": int(r["reviews"] or 0),
                    "avg_stars": float(r["avg_stars"]) if r.get("avg_stars") else None,
                    "sample_review": "",
                    "dataset": r.get("dataset") or "sealuzh/app_reviews",
                    "github_repo": gh,
                    "roadmap_source": "github" if gh else "none",
                    "likely_applicable": bool(gh),
                }
            )

    rows.sort(key=lambda x: -x["reviews"])
    out = DISCOVERY / "candidates_sealuzh.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with_gh = sum(1 for r in rows if r["github_repo"])
    print(f"wrote {len(rows)} apps ({with_gh} with github, {len(rows) - with_gh} none)")


if __name__ == "__main__":
    main()
