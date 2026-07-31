"""Runtime configuration via pydantic-settings / .env."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (Hackathon/) must be importable for silent_stakeholder
_API_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _API_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_API_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openai/gpt-4o-mini", alias="OPENROUTER_MODEL")

    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")

    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_KEY")

    embedding_backend: str = Field(default="tfidf", alias="EMBEDDING_BACKEND")
    match_threshold: float = Field(default=0.45, alias="MATCH_THRESHOLD")
    match_threshold_tfidf: float = Field(default=0.22, alias="MATCH_THRESHOLD_TFIDF")
    match_threshold_minilm: float = Field(default=0.45, alias="MATCH_THRESHOLD_MINILM")

    max_reviews: int = Field(default=2000, alias="MAX_REVIEWS")
    hf_dataset: str = Field(default="sealuzh/app_reviews", alias="HF_DATASET")

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

    @property
    def github_token_present(self) -> bool:
        return bool(self.github_token and self.github_token.strip())

    def active_match_threshold(self, backend: str | None = None) -> float:
        b = (backend or self.embedding_backend or "tfidf").lower().strip()
        if b == "minilm":
            return float(self.match_threshold_minilm)
        return float(self.match_threshold_tfidf)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def main() -> None:
    s = get_settings()
    print(
        {
            "llm_enabled": s.llm_enabled,
            "llm_model": s.openrouter_model,
            "github_token": s.github_token_present,
            "embedding_backend": s.embedding_backend,
            "match_threshold": s.active_match_threshold(),
            "match_threshold_tfidf": s.match_threshold_tfidf,
            "match_threshold_minilm": s.match_threshold_minilm,
            "max_reviews": s.max_reviews,
            "hf_dataset": s.hf_dataset,
            "supabase_configured": bool(s.supabase_url and s.supabase_service_key),
            "repo_root": str(s.repo_root),
        }
    )


if __name__ == "__main__":
    main()
