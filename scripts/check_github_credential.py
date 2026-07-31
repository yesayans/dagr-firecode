"""
Probe whether the cached git credential also works as a GitHub API token.

Prints only outcomes (status, rate limit, counts) — never the secret itself.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request

REPO = "AntennaPod/AntennaPod"


def read_credential() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        print(f"credential helper failed: {type(e).__name__}: {e}")
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return None


def probe(token: str | None) -> None:
    label = "authenticated" if token else "anonymous"
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/issues?state=all&per_page=5")
    req.add_header("User-Agent", "dagr/1.0")
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
            print(f"[{label}] HTTP {r.status}")
            print(f"[{label}] rate limit remaining: {r.headers.get('x-ratelimit-remaining')} / {r.headers.get('x-ratelimit-limit')}")
            print(f"[{label}] issues returned: {len(data)}")
            for it in data[:3]:
                kind = "PR" if "pull_request" in it else "issue"
                print(f"    #{it.get('number')} [{kind}/{it.get('state')}] {(it.get('title') or '')[:70]}")
    except urllib.error.HTTPError as e:
        print(f"[{label}] HTTP {e.code}")
        print(f"[{label}] rate limit remaining: {e.headers.get('x-ratelimit-remaining')} / {e.headers.get('x-ratelimit-limit')}")


def main() -> None:
    probe(None)
    print()
    token = read_credential()
    if not token:
        print("no cached credential found for github.com")
        return
    kind = "classic PAT" if token.startswith("ghp_") else (
        "fine-grained PAT" if token.startswith("github_pat_") else (
            "OAuth token" if token.startswith("gho_") else "unknown format"
        )
    )
    print(f"cached credential found: {kind}, length {len(token)}")
    probe(token)


if __name__ == "__main__":
    main()
