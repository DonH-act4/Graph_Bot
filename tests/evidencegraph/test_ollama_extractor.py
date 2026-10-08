"""Local Ollama graph requests still pass the shared provenance boundary."""

import json
from typing import Any

import httpx
import pytest

from evidencegraph.extraction import GraphExtractionError, extract_graph
from evidencegraph.graph_contract import NodeType, RelationType, relation_endpoints_are_valid
from evidencegraph.models import BoundingBox, ParsedBlock, ParsedDocument, SourceLocation
from evidencegraph.ollama_extractor import (
    OllamaGraphExtractor,
    _overview_blocks,
    _research_blocks,
    _safe_chunk_issue,
    _validate_ollama_chunk_graph,
)

PDF_HASH = "a" * 64
BLOCK_ID = "blk_" + "1" * 24


class FakeClient:
    def __init__(self, response: httpx.Response | Exception | list[httpx.Response]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.requests.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def document() -> ParsedDocument:
    text = "The paper reports that Method A improves accuracy."
    return ParsedDocument(
        source_filename="paper.pdf",
        source_sha256=PDF_HASH,
        parser_version="test",
        page_count=1,
        blocks=(
            ParsedBlock(
                block_id=BLOCK_ID,
                source_ref="#/texts/0",
                label="text",
                text=text,
                locations=(
                    SourceLocation(
                        page_number=1,
                        bounding_box=BoundingBox(
                            left=0, top=1, right=2, bottom=3,
                            coordinate_origin="bottom-left",
                        ),
                        character_start=0,
                        character_end=len(text),
                    ),
                ),
            ),
        ),
    )


def graph_json() -> str:
    return json.dumps({
        "schema_version": "3",
        "nodes": [
            {
                "node_id": "paper",
                "node_type": "paper",
                "name": "paper.pdf",
                "document_sha256": PDF_HASH,
                "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            },
            {
                "node_id": "method",
                "node_type": "process",
                "name": "Method A",
                "document_sha256": PDF_HASH,
                "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            },
        ],
        "relations": [{
            "relation_id": "introduces",
            "source_node_id": "paper",
            "target_node_id": "method",
            "relation_type": "introduces",
            "status": "candidate",
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            "rationale": "The paper describes Method A.",
        }],
    })


def invalid_endpoint_graph_json() -> str:
    graph = json.loads(graph_json())
    graph["nodes"][1]["node_type"] = "concept"
    graph["nodes"].append({
        "node_id": "claim",
        "node_type": "claim",
        "name": "Method A improves accuracy",
        "document_sha256": PDF_HASH,
        "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
    })
    graph["relations"] = [{
        "relation_id": "invalid_support",
        "source_node_id": "method",
        "target_node_id": "claim",
        "relation_type": "supports",
        "status": "candidate",
        "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        "rationale": "A concept alone is not evidence.",
    }]
    return json.dumps(graph)


def invalid_evidence_graph_json() -> str:
    graph = json.loads(graph_json())
    graph["relations"][0]["evidence"][0]["block_id"] = "blk_bad"
    return json.dumps(graph)


def response(content: str, *, reason: str = "stop", status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "message": {"content": content},
            "done_reason": reason,
            "prompt_eval_count": 100,
            "eval_count": 50,
        },
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )


def test_ollama_request_uses_schema_and_graph_is_evidence_checked() -> None:
    client = FakeClient(response(graph_json()))
    artifact = extract_graph(
        document(),
        OllamaGraphExtractor(
            model="ollama/qwen3:14b-q4_K_M",
            base_url="http://localhost:11434/",
            client=client,
        ),
    )

    url, kwargs = client.requests[0]
    payload = kwargs["json"]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen3:14b-q4_K_M"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_predict"] == 8_192
    assert "Never use supports from a concept to a claim" in payload["messages"][0]["content"]
    assert artifact.extractor_version.endswith("ollama-v2.3-overview")
    assert payload["format"]["additionalProperties"] is False
    assert BLOCK_ID in payload["messages"][1]["content"]
    assert artifact.extractor_name == "ollama"
    assert len(artifact.graph.relations) == 1
    assert artifact.usage is not None and artifact.usage.total_tokens == 150


def test_gpt_oss_graph_request_enables_reasoning() -> None:
    client = FakeClient(response(graph_json()))
    extractor = OllamaGraphExtractor(
        model="ollama/gpt-oss:20b", base_url="http://localhost:11434", client=client,
    )

    extract_graph(document(), extractor)

    assert client.requests[0][1]["json"]["think"] == "low"
    assert client.requests[0][1]["json"]["model"] == "gpt-oss:20b"
    assert client.requests[0][1]["json"]["format"] == "json"
    assert "prioritize at least one process node" in client.requests[0][1]["json"]["messages"][0]["content"]
    guidance = client.requests[0][1]["json"]["messages"][0]["content"]
    for relation, source, target in (
        (RelationType.STUDIES, NodeType.PAPER, NodeType.PROCESS),
        (RelationType.PRODUCES, NodeType.PROCESS, NodeType.OBSERVATION),
        (RelationType.REPORTS, NodeType.PAPER, NodeType.CLAIM),
    ):
        assert f"{relation.value} ({source.value} -> {target.value})" in guidance
        assert relation_endpoints_are_valid(relation, source, target, schema_version="3")
    assert "Never use uses from a paper" in guidance


def test_missing_relation_identifiers_are_deterministic_without_changing_facts() -> None:
    payload = json.loads(graph_json())
    payload["relations"][0].pop("relation_id")
    candidate = json.dumps(payload)

    first = _validate_ollama_chunk_graph(candidate)
    second = _validate_ollama_chunk_graph(candidate)
    relation = first.relations[0]

    assert first == second
    assert relation.relation_id.startswith("relation_")
    assert relation.source_node_id == payload["relations"][0]["source_node_id"]
    assert relation.target_node_id == payload["relations"][0]["target_node_id"]
    assert relation.evidence[0].block_id == BLOCK_ID
    client = FakeClient(response(candidate))
    artifact = extract_graph(document(), OllamaGraphExtractor(
        model="gpt-oss:20b", base_url="http://localhost:11434", client=client,
    ))
    assert len(artifact.graph.relations) == 1
    assert len(client.requests) == 1


@pytest.mark.parametrize("identifier", [[], {}, 17])
def test_invalid_relation_identifier_types_still_fail_validation(identifier: Any) -> None:
    payload = json.loads(graph_json())
    payload["relations"][0]["relation_id"] = identifier
    with pytest.raises(GraphExtractionError, match="invalid structured"):
        _validate_ollama_chunk_graph(json.dumps(payload))


def test_overview_samples_all_main_sections_using_only_original_blocks() -> None:
    base = document().blocks[0]
    entries = [("section_header", "A complete research paper title")]
    for heading in ("Abstract", "3. Methods", "5. Results", "6. Conclusions"):
        entries.append(("section_header", heading))
        entries.extend(("text", f"{heading} passage {number}.") for number in range(10))
    entries += [("section_header", "Appendix A. More details"), ("text", "Appendix details.")]
    blocks = tuple(base.model_copy(update={
        "block_id": f"blk_{index:024x}", "label": label, "text": text,
    }) for index, (label, text) in enumerate(entries, start=1))

    selected = _overview_blocks(blocks)

    assert len(selected) < len(blocks)
    assert all(block in blocks for block in selected)
    assert all(any(heading in block.text for block in selected) for heading in (
        "Abstract", "3. Methods", "5. Results", "6. Conclusions",
    ))
    assert not any("Appendix" in block.text for block in selected)
    assert selected == _overview_blocks(blocks)


def test_unfamiliar_sections_use_a_bounded_spread_of_source_passages() -> None:
    base = document().blocks[0]
    blocks = tuple(base.model_copy(update={
        "block_id": f"blk_{index:024x}", "text": f"Theoretical passage {index}.",
    }) for index in range(30))

    selected = _overview_blocks(blocks)

    assert len(selected) == 10
    assert blocks[0] in selected and blocks[-1] in selected


def test_optional_measurements_do_not_break_non_observation_nodes() -> None:
    graph = json.loads(graph_json())
    graph["nodes"][1].update(value="a value", unit="m", uncertainty="low", conditions="field")
    client = FakeClient(response(json.dumps(graph)))
    artifact = extract_graph(
        document(),
        OllamaGraphExtractor(model="gpt-oss:20b", base_url="http://localhost:11434", client=client),
    )

    method = next(node for node in artifact.graph.nodes if node.name == "Method A")
    assert method.node_type == "process"
    assert (method.value, method.unit, method.uncertainty, method.conditions) == (None,) * 4
    assert method.evidence[0].block_id == BLOCK_ID
    assert len(client.requests) == 1


def test_observation_measurements_are_preserved() -> None:
    graph = json.loads(graph_json())
    graph["nodes"][1].update(node_type="observation", value=6.57, unit="m", conditions="40 scenarios")
    graph["relations"][0]["relation_type"] = "reports"

    validated = _validate_ollama_chunk_graph(json.dumps(graph))

    assert validated.nodes[1].value == 6.57
    assert validated.nodes[1].unit == "m"
    assert validated.nodes[1].conditions == "40 scenarios"


def test_optional_metadata_normalization_never_repairs_unknown_evidence() -> None:
    graph = json.loads(graph_json())
    graph["nodes"][1].update(value=1, conditions="field")
    graph["nodes"][1]["evidence"][0]["block_id"] = "blk_bad"

    with pytest.raises(GraphExtractionError):
        _validate_ollama_chunk_graph(json.dumps(graph))


def test_later_sections_receive_title_without_repeating_abstract() -> None:
    original = document()
    base = original.blocks[0]
    entries = [
        ("title", "A cross-domain research paper title"),
        ("section_header", "Abstract"),
        ("text", "A general summary. " * 40),
        ("section_header", "Methods"),
        ("text", "A detailed experimental method. " * 40),
        ("section_header", "Results"),
        ("text", "The measured findings. " * 40),
    ]
    blocks = tuple(
        base.model_copy(update={"block_id": f"blk_{index:024x}", "label": label, "text": text})
        for index, (label, text) in enumerate(entries, start=1)
    )
    extractor = OllamaGraphExtractor(
        model="gpt-oss:20b", base_url="http://localhost:11434", max_chunk_chars=1_200,
    )

    chunks = extractor.chunk_document(original.model_copy(update={"blocks": blocks}))

    assert len(chunks) > 1
    assert all(blocks[0] in chunk for chunk in chunks)
    assert sum(blocks[2] in chunk for chunk in chunks) == 1


@pytest.mark.parametrize(
    ("fake_response", "message"),
    [
        (response(graph_json(), reason="length"), "token limit"),
        (response("", status=404), "HTTP 404"),
        (httpx.ConnectError("private address"), "Cannot reach Ollama"),
        (httpx.ReadTimeout("slow server"), "timed out"),
    ],
)
def test_ollama_fails_clearly_without_persisting_a_graph(
    fake_response: httpx.Response | Exception, message: str
) -> None:
    extractor = OllamaGraphExtractor(
        model="qwen3:14b-q4_K_M",
        base_url="http://localhost:11434",
        client=FakeClient(fake_response),
    )
    with pytest.raises(GraphExtractionError, match=message):
        extract_graph(document(), extractor)


def test_ollama_rejects_invalid_section_json() -> None:
    extractor = OllamaGraphExtractor(
        model="qwen3:14b-q4_K_M",
        base_url="http://localhost:11434",
        client=FakeClient([response("not json"), response("still not json")]),
    )
    with pytest.raises(GraphExtractionError, match="remained invalid after one repair"):
        extract_graph(document(), extractor)


def test_ollama_repairs_invalid_endpoints_once_and_revalidates() -> None:
    client = FakeClient([response(invalid_endpoint_graph_json()), response(graph_json())])
    extractor = OllamaGraphExtractor(
        model="qwen3:14b-q4_K_M",
        base_url="http://localhost:11434",
        client=client,
    )

    artifact = extract_graph(document(), extractor)

    assert len(client.requests) == 2
    assert "invalid_candidate_json" in client.requests[1][1]["json"]["messages"][1]["content"]
    assert artifact.usage is not None and artifact.usage.total_tokens == 300
    assert len(artifact.graph.relations) == 1


def test_ollama_repair_does_not_accept_still_invalid_endpoints() -> None:
    client = FakeClient([
        response(invalid_endpoint_graph_json()),
        response(invalid_endpoint_graph_json()),
    ])
    extractor = OllamaGraphExtractor(
        model="qwen3:14b-q4_K_M",
        base_url="http://localhost:11434",
        client=client,
    )

    with pytest.raises(GraphExtractionError, match="too many invalid ontology relations"):
        extract_graph(document(), extractor)
    assert len(client.requests) == 2


def test_ollama_repairs_malformed_block_id_with_real_evidence_only() -> None:
    client = FakeClient([response(invalid_evidence_graph_json()), response(graph_json())])
    extractor = OllamaGraphExtractor(
        model="qwen3:14b-q4_K_M",
        base_url="http://localhost:11434",
        client=client,
    )

    artifact = extract_graph(document(), extractor)

    repair_prompt = client.requests[1][1]["json"]["messages"][1]["content"]
    assert "relations.0.evidence.0.block_id [string_pattern_mismatch]" in repair_prompt
    assert artifact.usage is not None and artifact.usage.total_tokens == 300
    assert artifact.graph.relations[0].evidence[0].block_id == BLOCK_ID


def test_safe_chunk_issue_never_exposes_invalid_model_value() -> None:
    issue = _safe_chunk_issue(
        invalid_evidence_graph_json(), GraphExtractionError("invalid section")
    )
    assert issue == "relations.0.evidence.0.block_id [string_pattern_mismatch]"
    assert "blk_bad" not in issue


def test_ollama_reports_safe_field_when_evidence_repair_still_fails() -> None:
    client = FakeClient([
        response(invalid_evidence_graph_json()),
        response(invalid_evidence_graph_json()),
    ])
    extractor = OllamaGraphExtractor(
        model="qwen3:14b-q4_K_M",
        base_url="http://localhost:11434",
        client=client,
    )

    with pytest.raises(GraphExtractionError) as raised:
        extract_graph(document(), extractor)

    assert "relations.0.evidence.0.block_id [string_pattern_mismatch]" in str(raised.value)
    assert "blk_bad" not in str(raised.value)
    assert len(client.requests) == 2


def test_ollama_focuses_model_input_without_changing_source_blocks() -> None:
    original = document()
    base = original.blocks[0]
    blocks = tuple(
        base.model_copy(update={"block_id": f"blk_{index:024x}", "label": label, "text": text})
        for index, (label, text) in enumerate(
            (
                ("section_header", "A cross-domain research paper title"),
                ("text", "Author names and contact details"),
                ("section_header", "A B S T R A C T"),
                ("text", "The paper reports its central finding."),
                ("section_header", "References"),
                ("text", "A cited study, not this paper's finding."),
            ),
            start=1,
        )
    )

    focused = _research_blocks(blocks)

    assert [block.text for block in focused] == [
        "A cross-domain research paper title",
        "A B S T R A C T",
        "The paper reports its central finding.",
    ]
    assert len(blocks) == 6
    assert _research_blocks(original.blocks) == original.blocks


def test_administrative_sections_are_skipped_but_research_appendices_survive() -> None:
    base = document().blocks[0]
    entries = (
        ("section_header", "Conclusions"), ("text", "Transfer improves performance."),
        ("section_header", "CRediT authorship contribution statement"),
        ("text", "Author: Writing."),
        ("section_header", "Appendix A. Pressure wave model"),
        ("text", "The appendix derives the method."),
        ("section_header", "Appendix B. Supplementary data"),
        ("text", "Download link."), ("section_header", "Data availability"),
        ("text", "Data available on request."),
        ("section_header", "References"), ("text", "A cited study."),
    )
    blocks = tuple(base.model_copy(update={"label":label,"text":text}) for label,text in entries)

    selected = _research_blocks(blocks)

    assert [block.text for block in selected] == [
        "Conclusions", "Transfer improves performance.",
        "Appendix A. Pressure wave model", "The appendix derives the method.",
    ]
    assert len(blocks) == 12


def test_funding_as_research_topic_is_not_mistaken_for_funding_statement() -> None:
    base = document().blocks[0]
    blocks = (
        base.model_copy(update={"label":"section_header","text":"Funding"}),
        base.model_copy(update={"text":"We study how funding affects social outcomes."}),
    )

    assert _research_blocks(blocks) == blocks
