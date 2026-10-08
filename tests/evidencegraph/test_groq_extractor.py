from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import httpx
import pytest

from evidencegraph.extraction import GraphExtractionError, extract_graph
from evidencegraph.graph_contract import GraphAnnotation
from evidencegraph.groq_extractor import (
    GroqGraphExtractor,
    _chunk_document_blocks,
    _merge_chunk_graphs,
)
from evidencegraph.models import BoundingBox, ParsedBlock, ParsedDocument, SourceLocation

PDF_HASH = "b" * 64


def test_capped_overview_keeps_methods_and_findings_not_only_claims() -> None:
    graphs = []
    evidence = [{"source_sha256": PDF_HASH, "block_id": f"blk_{0:024x}"}]
    for section in range(6):
        nodes = [{
            "node_id": "paper", "node_type": "paper", "name": "The paper",
            "document_sha256": PDF_HASH, "evidence": evidence,
        }]
        relations = []
        for kind, count in (("claim", 7), ("process", 1), ("observation", 1)):
            for number in range(count):
                node_id = f"{section}-{kind}-{number}"
                nodes.append({
                    "node_id": node_id, "node_type": kind, "name": node_id,
                    "document_sha256": PDF_HASH, "evidence": evidence,
                })
                relations.append({
                    "relation_id": node_id, "source_node_id": "paper",
                    "target_node_id": node_id,
                    "relation_type": "introduces" if kind == "process" else "reports",
                    "status": "candidate", "evidence": evidence,
                    "rationale": "The section describes this item.",
                })
        graphs.append(GraphAnnotation.model_validate({
            "schema_version": "3", "nodes": nodes, "relations": relations,
        }))

    merged = _merge_chunk_graphs(graphs, document_sha256=PDF_HASH)

    assert len(merged.nodes) == 25
    assert {node.node_type for node in merged.nodes} == {"paper", "process", "observation", "claim"}
    linked = {endpoint for relation in merged.relations for endpoint in (
        relation.source_node_id, relation.target_node_id,
    )}
    assert linked == {node.node_id for node in merged.nodes}
    assert all(ref.block_id == f"blk_{0:024x}" for node in merged.nodes for ref in node.evidence)


class FakeHttpClient:
    def __init__(self, responses: Iterable[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _block(index: int, text: str, *, label: str = "text") -> ParsedBlock:
    return ParsedBlock(
        block_id=f"blk_{index:024x}",
        source_ref=f"#/texts/{index}",
        label=label,
        text=text,
        locations=(
            SourceLocation(
                page_number=index + 1,
                bounding_box=BoundingBox(
                    left=0,
                    top=1,
                    right=2,
                    bottom=3,
                    coordinate_origin="bottom-left",
                ),
                character_start=0,
                character_end=len(text),
            ),
        ),
    )


def _document() -> ParsedDocument:
    return ParsedDocument(
        source_filename="paper.pdf",
        source_sha256=PDF_HASH,
        parser_version="test",
        page_count=3,
        blocks=(
            _block(0, "A cross-domain paper", label="title"),
            _block(1, "The paper introduces Method A. " + "m" * 620),
            _block(2, "Method A supports the main claim. " + "r" * 620),
        ),
    )


def _graph_payload(*, second: bool = False) -> str:
    paper = {
        "node_id": "paper-from-model",
        "node_type": "paper",
        "name": "A cross-domain paper",
        "domain_type": None,
        "document_sha256": PDF_HASH,
        "evidence": [{"source_sha256": PDF_HASH, "block_id": f"blk_{0:024x}"}],
        "value": None,
        "unit": None,
        "uncertainty": None,
        "conditions": None,
    }
    method = {
        "node_id": "method-a-alternate" if second else "method-a",
        "node_type": "process",
        "name": "Method A",
        "domain_type": "method",
        "document_sha256": PDF_HASH,
        "evidence": [
            {
                "source_sha256": PDF_HASH,
                "block_id": f"blk_{2 if second else 1:024x}",
            }
        ],
        "value": None,
        "unit": None,
        "uncertainty": None,
        "conditions": None,
    }
    nodes = [paper, method]
    relations = []
    if second:
        claim = {
            "node_id": "claim",
            "node_type": "claim",
            "name": "Main claim",
            "domain_type": None,
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": f"blk_{2:024x}"}],
            "value": None,
            "unit": None,
            "uncertainty": None,
            "conditions": None,
        }
        nodes.append(claim)
        relations.extend(
            [
                {
                    "relation_id": "reports",
                    "source_node_id": "paper-from-model",
                    "target_node_id": "claim",
                    "relation_type": "reports",
                    "domain_relation": None,
                    "status": "candidate",
                    "evidence": [
                        {"source_sha256": PDF_HASH, "block_id": f"blk_{2:024x}"}
                    ],
                    "rationale": "The paper reports the claim.",
                },
                {
                    "relation_id": "supports",
                    "source_node_id": "method-a-alternate",
                    "target_node_id": "claim",
                    "relation_type": "supports",
                    "domain_relation": None,
                    "status": "candidate",
                    "evidence": [
                        {"source_sha256": PDF_HASH, "block_id": f"blk_{2:024x}"}
                    ],
                    "rationale": "Method A supports the claim.",
                },
            ]
        )
    else:
        relations.append(
            {
                "relation_id": "introduces",
                "source_node_id": "paper-from-model",
                "target_node_id": "method-a",
                "relation_type": "introduces",
                "domain_relation": None,
                "status": "candidate",
                "evidence": [
                    {"source_sha256": PDF_HASH, "block_id": f"blk_{1:024x}"}
                ],
                "rationale": "The paper introduces Method A.",
            }
        )
    return json.dumps({"schema_version": "3", "nodes": nodes, "relations": relations})


def _response(content: str, *, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    payload = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    return httpx.Response(
        status,
        json=payload,
        headers=headers,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


def test_chunks_and_merges_graph_with_strict_schema_and_progress() -> None:
    client = FakeHttpClient(
        [
            _response(_graph_payload(), headers={"x-ratelimit-remaining-tokens": "8000"}),
            _response(
                _graph_payload(second=True),
                headers={"x-ratelimit-remaining-tokens": "8000"},
            ),
        ]
    )
    extractor = GroqGraphExtractor(
        model="groq/openai/gpt-oss-120b",
        client=client,
        max_chunk_chars=1_000,
        context_chars=200,
    )
    progress: list[tuple[str, int, str | None]] = []

    artifact = extract_graph(
        _document(),
        extractor,
        progress_callback=lambda stage, percent, detail: progress.append(
            (stage, percent, detail)
        ),
    )

    assert len(client.requests) == 2
    assert all(
        request["json"]["model"] == "openai/gpt-oss-120b"
        for request in client.requests
    )
    response_format = client.requests[0]["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    strict_schema = response_format["json_schema"]["schema"]
    assert strict_schema["additionalProperties"] is False
    assert set(strict_schema["required"]) == {"schema_version", "nodes", "relations"}
    assert "pattern" not in json.dumps(strict_schema)
    assert "minLength" not in json.dumps(strict_schema)
    assert "m" * 200 in client.requests[0]["json"]["messages"][1]["content"]
    assert "r" * 200 not in client.requests[0]["json"]["messages"][1]["content"]
    assert "r" * 200 in client.requests[1]["json"]["messages"][1]["content"]

    assert artifact.extractor_name == "groq"
    assert "section-chunks-v2" in artifact.extractor_version
    assert len(artifact.graph.nodes) == 3
    assert len(artifact.graph.relations) == 3
    method = next(node for node in artifact.graph.nodes if node.name == "Method A")
    assert len(method.evidence) == 2
    assert artifact.usage is not None and artifact.usage.total_tokens == 300
    assert [item[0] for item in progress].count("chunk_extraction") == 2
    assert "graph_merge" in [item[0] for item in progress]
    assert progress[-1][:2] == ("saving_graph", 95)


def test_waits_for_429_retry_after_and_reports_it() -> None:
    rate_limited = httpx.Response(
        429,
        json={"error": {"message": "sensitive provider detail"}},
        headers={"retry-after": "2"},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    client = FakeHttpClient([rate_limited, _response(_graph_payload())])
    sleeps: list[float] = []
    progress: list[tuple[str, int, str | None]] = []
    extractor = GroqGraphExtractor(
        model="openai/gpt-oss-120b",
        client=client,
        sleeper=sleeps.append,
        max_chunk_chars=5_000,
    )

    extractor.extract_with_progress(
        _document().model_copy(update={"blocks": _document().blocks[:2]}),
        lambda stage, percent, detail: progress.append((stage, percent, detail)),
    )

    assert sleeps == [2.0]
    wait = next(item for item in progress if item[0] == "rate_limit_wait")
    assert "retrying in 2s" in (wait[2] or "")
    assert len(client.requests) == 2


def test_provider_error_is_actionable_and_does_not_leak_body() -> None:
    unauthorized = httpx.Response(
        401,
        json={"error": {"message": "secret provider detail"}},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    extractor = GroqGraphExtractor(
        model="openai/gpt-oss-120b",
        client=FakeHttpClient([unauthorized]),
        max_chunk_chars=5_000,
    )

    with pytest.raises(GraphExtractionError, match="GROQ_API_KEY") as raised:
        extractor.extract(_document().model_copy(update={"blocks": _document().blocks[:2]}))

    assert "secret provider detail" not in str(raised.value)


def test_chunking_preserves_complete_blocks_and_repeats_small_context() -> None:
    document = _document()

    chunks = _chunk_document_blocks(
        document.blocks,
        max_chunk_chars=1_000,
        context_chars=200,
    )

    assert len(chunks) == 2
    assert chunks[0] == document.blocks[:2]
    assert chunks[1] == (document.blocks[0], document.blocks[2])
    assert {block.block_id for chunk in chunks for block in chunk} == {
        block.block_id for block in document.blocks
    }
