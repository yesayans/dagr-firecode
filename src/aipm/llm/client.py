"""LLM transport against any OpenAI-compatible endpoint.

The endpoint is a third-party router, so two assumptions are deliberately not
made: that it enforces `response_format=json_schema`, and that its tokeniser
matches OpenAI's. Structured output is handled in `structured.py`; token usage is
read from the response body rather than estimated locally.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from aipm.config import Settings
from aipm.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Usage:
    """Token and cost accounting, aggregated across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0
    n_failures: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.n_calls += other.n_calls
        self.n_failures += other.n_failures
        self.cost_usd += other.cost_usd

    def summary_line(self) -> str:
        return (
            f"{self.n_calls} call(s), {self.prompt_tokens:,} in / "
            f"{self.completion_tokens:,} out, {self.n_failures} failure(s), "
            f"${self.cost_usd:.4f}"
        )


@dataclass
class LlmResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""


class LlmError(RuntimeError):
    """Transport or protocol failure that survived all retries."""


class LlmClient(ABC):
    """The interface every consumer is injected with."""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @property
    def available(self) -> bool:
        return True

    def healthcheck(self) -> tuple[bool, str]:
        """Cheap probe run before a batch.

        Without this, a bad key or wrong base URL is only discovered per cluster,
        after full retry backoff, for every app in the run. Returns
        ``(ok, reason)``.
        """
        return True, "ok"

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LlmResponse: ...


class OpenAICompatibleClient(LlmClient):
    """Chat-completions client with bounded exponential backoff."""

    #: USD per 1M tokens. Only used for reporting; unknown models cost 0 rather
    #: than an invented number.
    PRICING: dict[str, tuple[float, float]] = {
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-pro": (1.25, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
    }

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        max_retries: int = 3,
        timeout_s: int = 60,
        max_output_tokens: int = 2048,
        user_agent: str = "",
    ) -> None:
        if not api_key:
            raise ValueError("LLM_API_KEY is not set")
        from openai import OpenAI

        # Retries are handled here, not by the SDK, so backoff can be logged and
        # a failed call can be counted in `Usage`.
        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
            # Overriding the User-Agent is a functional requirement, not a nicety:
            # some routers front their API with a WAF that rejects the SDK's own
            # `OpenAI/Python...` agent with a 403.
            default_headers={"User-Agent": user_agent} if user_agent else None,
        )
        self._model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens

    @property
    def model(self) -> str:
        return self._model

    def _price(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Routers namespace model ids by provider ("google/gemini-2.5-flash"),
        # so match on the last path segment.
        name = self._model.rsplit("/", 1)[-1]
        key = next((k for k in self.PRICING if name.startswith(k)), None)
        if key is None:
            return 0.0
        in_rate, out_rate = self.PRICING[key]
        return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000

    def healthcheck(self) -> tuple[bool, str]:
        """One minimal completion, no retries, so a bad config surfaces immediately."""
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, "ok"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.max_output_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                backoff = min(2.0**attempt, 30.0)
                log.warning(
                    "LLM call failed (attempt %d/%d): %s: %s; retrying in %.0fs",
                    attempt, self.max_retries, type(exc).__name__, exc, backoff,
                )
                time.sleep(backoff)
                continue

            # Usage comes from the response. Routers vary in what they populate,
            # so every field is read defensively.
            raw_usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(raw_usage, "completion_tokens", 0) or 0)

            choice = response.choices[0] if response.choices else None
            text = (getattr(choice.message, "content", "") or "") if choice else ""
            return LlmResponse(
                text=text,
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    n_calls=1,
                    cost_usd=self._price(prompt_tokens, completion_tokens),
                ),
                model=getattr(response, "model", self._model),
                finish_reason=getattr(choice, "finish_reason", "") or "" if choice else "",
            )

        raise LlmError(
            f"LLM call failed after {self.max_retries} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


class NullLlmClient(LlmClient):
    """Stands in when no endpoint is configured.

    Every call raises. Callers are expected to check `available` and fall back to
    a deterministic path rather than letting a demo die on a missing key.
    """

    def __init__(self, reason: str = "no LLM endpoint configured") -> None:
        self.reason = reason

    @property
    def model(self) -> str:
        return "none"

    @property
    def available(self) -> bool:
        return False

    def healthcheck(self) -> tuple[bool, str]:
        return False, self.reason

    def complete(self, messages: list[dict[str, str]], **_: object) -> LlmResponse:
        raise LlmError(self.reason)


def build_llm_client(settings: Settings) -> LlmClient:
    """Construct the configured client, or a `NullLlmClient` with the reason why."""
    if not settings.llm_api_key:
        return NullLlmClient("LLM_API_KEY is not set")
    if not settings.llm_base_url:
        return NullLlmClient("LLM_BASE_URL is not set")
    try:
        client = OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            timeout_s=settings.llm_timeout_s,
            max_output_tokens=settings.llm_max_output_tokens,
            user_agent=settings.http_user_agent,
        )
    except Exception as exc:
        return NullLlmClient(f"could not build LLM client: {type(exc).__name__}: {exc}")
    log.info("LLM client: %s @ %s", settings.llm_model, settings.llm_base_url)
    return client
