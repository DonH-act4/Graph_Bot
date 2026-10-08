import pytest

from evidencegraph.graph_models import (
    configured_graph_models,
    graph_model_provider,
    provider_model_name,
)


def test_configured_graph_models_keeps_default_first_and_deduplicates():
    assert configured_graph_models(
        "gemini-default", "gemini-fast, gemini-default,models/gemini-pro"
    ) == ("gemini-default", "gemini-fast", "models/gemini-pro")


def test_configured_graph_models_ignores_empty_entries():
    assert configured_graph_models(None, " , gemini-fast, ") == ("gemini-fast",)


def test_configured_graph_models_rejects_unsafe_name():
    with pytest.raises(ValueError, match="Invalid graph model name"):
        configured_graph_models("gemini-fast; rm", None)


def test_provider_prefixes_route_without_changing_provider_model_id():
    assert graph_model_provider("groq/openai/gpt-oss-120b") == "groq"
    assert provider_model_name("groq/openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert graph_model_provider("gemini/gemini-flash-latest") == "gemini"
    assert provider_model_name("gemini/gemini-flash-latest") == "gemini-flash-latest"
    assert graph_model_provider("gemini-flash-latest") == "gemini"
    assert provider_model_name("gemini-flash-latest") == "gemini-flash-latest"
    assert graph_model_provider("ollama/qwen3:14b-q4_K_M") == "ollama"
    assert provider_model_name("ollama/qwen3:14b-q4_K_M") == "qwen3:14b-q4_K_M"


def test_configured_graph_models_accepts_ollama_tags():
    assert configured_graph_models("ollama/qwen3:14b-q4_K_M", None) == (
        "ollama/qwen3:14b-q4_K_M",
    )
