"""Configuration helpers for the bounded graph-extraction model catalog."""

from __future__ import annotations

import re
from typing import Literal

_MODEL_NAME_RE = re.compile(r"(?:models/)?[A-Za-z0-9][A-Za-z0-9._/:-]{0,127}\Z")
DEFAULT_GROQ_GRAPH_MODELS = (
    "groq/openai/gpt-oss-120b",
    "groq/openai/gpt-oss-20b",
)
DEFAULT_OLLAMA_GRAPH_MODELS = ("ollama/gpt-oss:20b",)


def configured_graph_models(
    default_model: str | None, configured_models: str | None
) -> tuple[str, ...]:
    """Return a stable, de-duplicated allowlist with the default first."""
    candidates = [default_model, *(configured_models or "").split(",")]
    models: list[str] = []
    for candidate in candidates:
        model = (candidate or "").strip()
        if not model or model in models:
            continue
        if not _MODEL_NAME_RE.fullmatch(model):
            raise ValueError(f"Invalid graph model name: {model!r}")
        models.append(model)
    return tuple(models)


def graph_model_provider(model: str) -> Literal["groq", "gemini", "ollama"]:
    """Resolve explicit provider prefixes while preserving old Gemini model names."""
    if model.startswith("groq/"):
        return "groq"
    if model.startswith("ollama/"):
        return "ollama"
    return "gemini"


def provider_model_name(model: str) -> str:
    """Remove the EvidenceGraph provider namespace before calling its API."""
    if model.startswith("groq/"):
        return model.removeprefix("groq/")
    if model.startswith("ollama/"):
        return model.removeprefix("ollama/")
    if model.startswith("gemini/"):
        return model.removeprefix("gemini/")
    return model
