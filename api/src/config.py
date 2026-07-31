"""Runtime configuration via pydantic-settings / .env."""

from __future__ import annotations

import logging
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (Hackathon/) must be importable for silent_stakeholder
_API_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _API_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

logger = logging.getLogger(__name__)

GitHubTokenSource = Literal["env", "git_credential", "none"]


def _read_git_credential_token() -> str | None:
    """
    Ask the local git credential helper for github.com.
    Never writes a second copy of the secret to disk; returns None on any failure.
    """
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode not in (0, None) and not proc.stdout:
        return None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("password="):
            secret = line.split("=", 1)[1].strip()
            return secret or None
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_API_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openai/gpt-4o-mini", alias="OPENROUTER_MODEL")

    # Explicit env token only — never write git-credential secrets into .env
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    github_token_from_git_credential: bool = Field(
        default=True, alias="GITHUB_TOKEN_FROM_GIT_CREDENTIAL"
    )

    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_KEY")

    embedding_backend: str = Field(default="tfidf", alias="EMBEDDING_BACKEND")
    match_threshold: float = Field(default=0.45, alias="MATCH_THRESHOLD")
    # Calibrated on AntennaPod probes (char_wb 3–5, boilerplate strip, no version
    # milestones; scripts/calibrate_retrieval.py): 5/5 correct top-1, min true ≈0.165.
    # Ambiguous negative ≈0.175 and within-query runners ≈0.33 — distributions overlap.
    # This is a RECALL FLOOR, not a precision cut. Ranking carries the precision.
    match_threshold_tfidf: float = Field(default=0.16, alias="MATCH_THRESHOLD_TFIDF")
    match_threshold_minilm: float = Field(default=0.45, alias="MATCH_THRESHOLD_MINILM")
    # tfidf margin disabled: weakest correct margin ≈0.003 (not useful as a gate).
    match_margin_tfidf: float = Field(default=0.0, alias="MATCH_MARGIN_TFIDF")
    match_margin_minilm: float = Field(default=0.05, alias="MATCH_MARGIN_MINILM")

    max_reviews: int = Field(default=2000, alias="MAX_REVIEWS")
    hf_dataset: str = Field(default="sealuzh/app_reviews", alias="HF_DATASET")

    _resolved_github_token: str | None = PrivateAttr(default=None)
    _github_token_source: GitHubTokenSource = PrivateAttr(default="none")
    _github_resolved: bool = PrivateAttr(default=False)

    @property
    def repo_root(self) -> Path:
        return _REPO_ROOT

    @property
    def api_dir(self) -> Path:
        return _API_DIR

    @property
    def data_dir(self) -> Path:
        return _REPO_ROOT / "data"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.strip())

    def resolve_github_credentials(self) -> None:
        """Resolve once at startup; cache in memory only. Never log the secret."""
        if self._github_resolved:
            return
        env_tok = (self.github_token or "").strip()
        if env_tok:
            self._resolved_github_token = env_tok
            self._github_token_source = "env"
            self._github_resolved = True
            return
        if self.github_token_from_git_credential:
            filled = _read_git_credential_token()
            if filled:
                self._resolved_github_token = filled
                self._github_token_source = "git_credential"
                self._github_resolved = True
                return
        self._resolved_github_token = None
        self._github_token_source = "none"
        self._github_resolved = True

    @property
    def effective_github_token(self) -> str | None:
        self.resolve_github_credentials()
        return self._resolved_github_token

    @property
    def github_token_source(self) -> GitHubTokenSource:
        self.resolve_github_credentials()
        return self._github_token_source

    @property
    def github_token_present(self) -> bool:
        return self.github_token_source != "none"

    def active_match_threshold(self, backend: str | None = None) -> float:
        b = (backend or self.embedding_backend or "tfidf").lower().strip()
        if b == "minilm":
            return float(self.match_threshold_minilm)
        return float(self.match_threshold_tfidf)

    def active_match_margin(self, backend: str | None = None) -> float:
        b = (backend or self.embedding_backend or "tfidf").lower().strip()
        if b == "minilm":
            return float(self.match_margin_minilm)
        return float(self.match_margin_tfidf)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.resolve_github_credentials()
    return s


def main() -> None:
    s = get_settings()
    print(
        {
            "llm_enabled": s.llm_enabled,
            "llm_model": s.openrouter_model,
            "github_token_source": s.github_token_source,
            "github_token_present": s.github_token_present,
            "github_token_from_git_credential": s.github_token_from_git_credential,
            "embedding_backend": s.embedding_backend,
            "match_threshold": s.active_match_threshold(),
            "match_margin": s.active_match_margin(),
            "match_threshold_tfidf": s.match_threshold_tfidf,
            "match_margin_tfidf": s.match_margin_tfidf,
            "match_threshold_minilm": s.match_threshold_minilm,
            "max_reviews": s.max_reviews,
            "hf_dataset": s.hf_dataset,
            "supabase_configured": bool(s.supabase_url and s.supabase_service_key),
            "repo_root": str(s.repo_root),
        }
    )


if __name__ == "__main__":
    main()
