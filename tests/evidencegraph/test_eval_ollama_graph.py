"""The local model comparison keeps every outcome separate from graph storage."""

import json
from pathlib import Path

import httpx

from evidencegraph.extraction import GraphExtractionError, GraphExtractionResponse
from evidencegraph.models import BoundingBox, ParsedBlock, ParsedDocument, SourceLocation
from evidencegraph.ollama_eval import ThinkingClient, evaluate_case

SOURCE_HASH = "a" * 64
BLOCK_ID = "blk_" + "1" * 24


def _document() -> ParsedDocument:
    text = "This paper introduces Method A."
    return ParsedDocument(
        source_filename="paper.pdf",
        source_sha256=SOURCE_HASH,
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


def _graph_json(*, block_id: str = BLOCK_ID) -> str:
    evidence = [{"source_sha256": SOURCE_HASH, "block_id": block_id}]
    return json.dumps({
        "schema_version": "3",
        "nodes": [
            {
                "node_id": "paper",
                "node_type": "paper",
                "name": "paper.pdf",
                "document_sha256": SOURCE_HASH,
                "evidence": evidence,
            },
            {
                "node_id": "method",
                "node_type": "process",
                "name": "Method A",
                "document_sha256": SOURCE_HASH,
                "evidence": evidence,
            },
        ],
        "relations": [
            {
                "relation_id": "introduces",
                "source_node_id": "paper",
                "target_node_id": "method",
                "relation_type": "introduces",
                "status": "candidate",
                "evidence": evidence,
                "rationale": "The paper introduces Method A.",
            }
        ],
    })


class FakeExtractor:
    model = "test-model"
    version = "test-version"

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.repairs = 0

    def _request_graph(
        self,
        _document: ParsedDocument,
        *,
        index: int,
        total: int,
        repair_candidate: str | None = None,
        validation_error: str | None = None,
    ) -> GraphExtractionResponse:
        assert index == 1 and total == 1
        if repair_candidate is not None:
            assert validation_error is not None
            self.repairs += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return GraphExtractionResponse(raw_graph_json=response)


def _evaluate(tmp_path: Path, extractor: FakeExtractor) -> dict:
    document = _document()
    return evaluate_case(
        document,
        document.blocks,
        extractor,
        index=1,
        total=1,
        repeat=1,
        parsed_sha256="b" * 64,
        think="low",
        case_dir=tmp_path / "case",
    )


def test_thinking_client_overrides_only_reasoning_and_timeout() -> None:
    payloads: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        thinking = ThinkingClient(client, think="low", timeout=12)
        thinking.post(
            "http://localhost:11434/api/chat",
            json={"model": "gpt-oss:20b", "think": False, "stream": False},
            timeout=240,
        )

    assert payloads == [{"model": "gpt-oss:20b", "think": "low", "stream": False}]


def test_success_saves_reviewable_candidate_without_store_write(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, FakeExtractor(_graph_json()))

    assert result["strict_first_valid"] is True
    assert result["final_valid"] is True
    assert result["repair_requested"] is False
    assert result["relation_types"] == {"introduces": 1}
    assert (tmp_path / "case" / "first.json").is_file()
    assert (tmp_path / "case" / "validated_graph.json").is_file()
    review = json.loads((tmp_path / "case" / "review.json").read_text())
    assert review[0]["evidence"][0]["excerpt"] == "This paper introduces Method A."


def test_invalid_output_records_one_repair_and_revalidates(tmp_path: Path) -> None:
    extractor = FakeExtractor(_graph_json(block_id="blk_bad"), _graph_json())
    result = _evaluate(tmp_path, extractor)

    assert result["strict_first_valid"] is False
    assert result["repair_requested"] is True
    assert result["final_valid"] is True
    assert extractor.repairs == 1
    assert (tmp_path / "case" / "repair.json").is_file()


def test_invalid_repair_records_safe_field_location(tmp_path: Path) -> None:
    extractor = FakeExtractor(
        _graph_json(block_id="blk_bad"),
        _graph_json(block_id="blk_bad"),
    )
    result = _evaluate(tmp_path, extractor)

    assert result["final_valid"] is False
    assert result["repair_requested"] is True
    assert any(
        "block_id" in location for location in result["repair_validation_locations"]
    )


def test_unknown_but_well_formed_evidence_is_failure(tmp_path: Path) -> None:
    result = _evaluate(tmp_path, FakeExtractor(_graph_json(block_id="blk_" + "f" * 24)))

    assert result["strict_first_valid"] is True
    assert result["final_valid"] is False
    assert result["error_category"] == "unknown_evidence"
    assert not (tmp_path / "case" / "validated_graph.json").exists()
    assert (tmp_path / "case" / "result.json").is_file()


def test_timeout_is_recorded_without_stopping_the_evaluation(tmp_path: Path) -> None:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    result = _evaluate(tmp_path, FakeExtractor(httpx.ReadTimeout("timed out", request=request)))

    assert result["final_valid"] is False
    assert result["error_category"] == "transport_timeout"
    assert not (tmp_path / "case" / "validated_graph.json").exists()
    assert (tmp_path / "case" / "result.json").is_file()


def test_adapter_wrapped_timeout_is_recorded_as_timeout(tmp_path: Path) -> None:
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    timeout = httpx.ReadTimeout("timed out", request=request)
    wrapped = GraphExtractionError("Ollama graph request timed out")
    wrapped.__cause__ = timeout

    result = _evaluate(tmp_path, FakeExtractor(wrapped))

    assert result["final_valid"] is False
    assert result["error_category"] == "transport_timeout"
