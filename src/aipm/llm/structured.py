"""Structured output without server-side schema enforcement.

The router may or may not honour `response_format`, so the contract is carried in
the prompt and enforced here:

1. ask for a single JSON object,
2. strip markdown fences and any prose the model wrapped it in,
3. parse,
4. validate against a pydantic model,
5. on failure, retry **once** with the exact validation error appended.

The repair retry matters: models reliably fix their own output when told which
field was wrong, and it turns most malformed responses into usable ones.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from aipm.llm.client import LlmClient, LlmError, LlmResponse, Usage
from aipm.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """The model could not be coaxed into valid JSON for the target schema."""


def strip_code_fences(text: str) -> str:
    """Return the JSON payload from a response that may be fenced or prose-wrapped."""
    if not text:
        return ""
    fenced = _FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def extract_json_object(text: str) -> str:
    """Isolate the outermost JSON object, ignoring any surrounding commentary.

    Brace counting rather than a regex, because review quotes routinely contain
    braces and escaped quotes.
    """
    payload = strip_code_fences(text)
    start = payload.find("{")
    if start == -1:
        return payload

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(payload)):
        char = payload[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return payload[start : i + 1]
    return payload[start:]


def parse_json_response(text: str) -> dict:
    """Parse a model response into a dict, raising `StructuredOutputError` if impossible."""
    candidate = extract_json_object(text)
    if not candidate:
        raise StructuredOutputError("empty response")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def schema_hint(model_cls: type[BaseModel]) -> str:
    """A compact field list for the prompt.

    Deliberately not the raw JSON Schema: it is verbose, and models follow a
    short annotated field list more reliably.
    """
    lines = []
    for name, field in model_cls.model_fields.items():
        annotation = getattr(field.annotation, "__name__", str(field.annotation))
        description = field.description or ""
        required = "required" if field.is_required() else "optional"
        lines.append(f'  "{name}": <{annotation}>  // {required}. {description}'.rstrip())
    return "{\n" + "\n".join(lines) + "\n}"


def generate_structured(
    client: LlmClient,
    model_cls: type[T],
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    context: str = "",
) -> tuple[T, Usage]:
    """Call the LLM and return a validated pydantic object plus token usage.

    Raises `StructuredOutputError` if the repair attempt also fails.
    """
    usage = Usage()
    response: LlmResponse = client.complete(
        messages, temperature=temperature, max_tokens=max_tokens
    )
    usage.add(response.usage)

    try:
        return model_cls.model_validate(parse_json_response(response.text)), usage
    except (StructuredOutputError, ValidationError) as exc:
        # Python unbinds the `as` name when the except block exits, so the error
        # has to be copied out before it can be quoted back to the model.
        first_error_text = _short(exc)
        log.warning(
            "structured output invalid%s (%s); retrying once with the error",
            f" for {context}" if context else "",
            first_error_text,
        )

    repair_messages = [
        *messages,
        {"role": "assistant", "content": response.text[:4000]},
        {
            "role": "user",
            "content": (
                "That response could not be parsed into the required schema.\n"
                f"Error:\n{first_error_text}\n\n"
                f"Required shape:\n{schema_hint(model_cls)}\n\n"
                "Reply with the corrected JSON object only. No markdown fences, "
                "no commentary, no trailing text."
            ),
        },
    ]

    try:
        repaired = client.complete(
            repair_messages, temperature=0.0, max_tokens=max_tokens
        )
    except LlmError as exc:
        usage.n_failures += 1
        raise StructuredOutputError(f"repair call failed: {exc}") from exc
    usage.add(repaired.usage)

    try:
        return model_cls.model_validate(parse_json_response(repaired.text)), usage
    except (StructuredOutputError, ValidationError) as second_error:
        usage.n_failures += 1
        raise StructuredOutputError(
            f"invalid structured output after repair"
            f"{f' for {context}' if context else ''}: {_short(second_error)}"
        ) from second_error


def _short(error: Exception, limit: int = 600) -> str:
    text = str(error).replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")
