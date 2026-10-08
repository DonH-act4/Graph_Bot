import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import httpx
import pytest

script = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/create_showcase.py"))
ready_graph_matches = cast(
    Callable[[httpx.Client, str, str], bool],
    script["ready_graph_matches"],
)


@pytest.mark.parametrize(
    ("status_code", "state", "version", "expected"),
    [
        (404, "missing", "", False),
        (200, "failed", "", False),
        (200, "ready", "gpt-oss:20b:ontology-v3.2:section-chunks-v2:ollama-v1.7", True),
        (200, "ready", "qwen3:14b:ontology-v3.2", False),
    ],
)
def test_showcase_resumes_only_a_ready_graph_from_the_requested_model(
    status_code: int, state: str, version: str, expected: bool,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/status"):
            return httpx.Response(status_code, json={"state": state})
        return httpx.Response(200, json={"extractor_version": version})

    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(respond)) as client:
        assert ready_graph_matches(client, "a" * 64, "ollama/gpt-oss:20b") is expected


def test_showcase_does_not_hide_proxy_errors_as_an_empty_graph() -> None:
    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(
        lambda request: httpx.Response(502),
    )) as client:
        with pytest.raises(httpx.HTTPStatusError):
            ready_graph_matches(client, "a" * 64, "ollama/gpt-oss:20b")


@pytest.mark.parametrize(("regenerate", "existing", "suffix"), [
    (False, 200, "/graph"), (True, 200, "/graph/rebuild"), (True, 404, "/graph"),
])
def test_showcase_explicit_regeneration_calls_the_rebuild_route(
    regenerate: bool, existing: int, suffix: str,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(existing if request.method == "GET" else 202, json={})

    request_generation = cast(Callable[..., None], script["request_graph_generation"])
    with httpx.Client(base_url="http://local", transport=httpx.MockTransport(respond)) as client:
        request_generation(client, "a" * 64, "ollama/gpt-oss:20b", regenerate=regenerate)
    assert requests[-1].method == "POST"
    assert requests[-1].url.path.endswith(suffix)
    assert b"gpt-oss:20b" in requests[-1].content
