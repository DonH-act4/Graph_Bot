from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from evidencegraph.extraction import (
    GraphArtifact,
    GraphExtractionError,
    GraphExtractionResponse,
    GraphUsage,
    extract_graph,
)
from evidencegraph.graph_contract import GraphAnnotation
from evidencegraph.models import ParsedBlock, ParsedDocument, SourceLocation
from evidencegraph.papers import PaperStore

PDF_HASH = "a" * 64
OTHER_HASH = "b" * 64
BLOCK_ID = "blk_" + "1" * 24


@dataclass(frozen=True)
class FakeExtractor:
    output: str = ""
    error: Exception | None = None
    name: str = "deterministic-test-extractor"
    version: str = "1"

    def extract(self, _document: ParsedDocument) -> str:
        if self.error is not None:
            raise self.error
        return self.output


@pytest.fixture
def document() -> ParsedDocument:
    return ParsedDocument(
        schema_version="2",
        source_filename="example.pdf",
        source_sha256=PDF_HASH,
        parser_version="test",
        page_count=1,
        blocks=(
            ParsedBlock(
                block_id=BLOCK_ID,
                source_ref="#/texts/0",
                label="text",
                text="A paper introduces a method.",
                locations=(
                    SourceLocation(
                        page_number=1,
                        bounding_box={
                            "left": 10,
                            "top": 20,
                            "right": 30,
                            "bottom": 40,
                            "coordinate_origin": "BOTTOMLEFT",
                        },
                        character_start=0,
                        character_end=28,
                    ),
                ),
            ),
        ),
    )


def graph_json(
    *,
    status: str = "candidate",
    source_hash: str = PDF_HASH,
    block_id: str = BLOCK_ID,
) -> str:
    return f"""
    {{
      "schema_version": "1",
      "nodes": [
        {{
          "node_id": "paper",
          "node_type": "paper",
          "name": "Example paper",
          "document_sha256": "{PDF_HASH}",
          "evidence": [{{"source_sha256": "{source_hash}", "block_id": "{block_id}"}}]
        }},
        {{
          "node_id": "method",
          "node_type": "method",
          "name": "Example method",
          "document_sha256": "{PDF_HASH}",
          "evidence": [{{"source_sha256": "{PDF_HASH}", "block_id": "{BLOCK_ID}"}}]
        }}
      ],
      "relations": [
        {{
          "relation_id": "introduces",
          "source_node_id": "paper",
          "target_node_id": "method",
          "relation_type": "introduces",
          "status": "{status}",
          "evidence": [{{"source_sha256": "{PDF_HASH}", "block_id": "{BLOCK_ID}"}}],
          "rationale": "The cited block names the method."
        }}
      ]
    }}
    """


def test_accepts_evidence_linked_candidate(document: ParsedDocument) -> None:
    artifact = extract_graph(document, FakeExtractor(output=graph_json()))

    assert artifact.document_sha256 == PDF_HASH
    assert artifact.extractor_name == "deterministic-test-extractor"
    assert artifact.graph.relations[0].status == "candidate"


def test_preserves_provider_usage_in_artifact(document: ParsedDocument) -> None:
    @dataclass(frozen=True)
    class UsageExtractor:
        name: str = "usage-test"
        version: str = "1"

        def extract(self, _document: ParsedDocument) -> GraphExtractionResponse:
            return GraphExtractionResponse(
                raw_graph_json=graph_json(),
                usage=GraphUsage(
                    input_tokens=100,
                    cached_input_tokens=0,
                    output_tokens=20,
                    thinking_tokens=5,
                    tool_tokens=0,
                    total_tokens=125,
                    paid_standard_estimate_usd=0.0001,
                    pricing_basis="test pricing",
                ),
            )

    artifact = extract_graph(document, UsageExtractor())

    assert artifact.usage is not None
    assert artifact.usage.total_tokens == 125
    assert artifact.usage.pricing_basis == "test pricing"


@pytest.mark.parametrize("output", ["not json", '{"nodes": []}'])
def test_rejects_malformed_structured_output(
    document: ParsedDocument, output: str
) -> None:
    with pytest.raises(GraphExtractionError, match="invalid structured output"):
        extract_graph(document, FakeExtractor(output=output))


def test_validation_error_reports_location_without_model_value(
    document: ParsedDocument,
) -> None:
    private_model_value = "do-not-copy-this-model-value"

    with pytest.raises(GraphExtractionError) as caught:
        extract_graph(
            document,
            FakeExtractor(output=f'{{"nodes":"{private_model_value}"}}'),
        )

    assert "nodes" in str(caught.value)
    assert private_model_value not in str(caught.value)


def test_rejects_model_claiming_human_verification(document: ParsedDocument) -> None:
    with pytest.raises(GraphExtractionError, match="human-verified"):
        extract_graph(document, FakeExtractor(output=graph_json(status="direct")))


def test_artifact_itself_rejects_direct_relation() -> None:
    graph = GraphAnnotation.model_validate_json(graph_json(status="direct"))

    with pytest.raises(ValidationError, match="cannot contain direct relations"):
        GraphArtifact(
            document_sha256=PDF_HASH,
            extractor_name="bypass-attempt",
            extractor_version="1",
            graph=graph,
        )


def test_rejects_cross_document_evidence(document: ParsedDocument) -> None:
    with pytest.raises(GraphExtractionError, match="another PDF"):
        extract_graph(document, FakeExtractor(output=graph_json(source_hash=OTHER_HASH)))


def test_rejects_fabricated_evidence_block(document: ParsedDocument) -> None:
    fabricated_block = "blk_" + "2" * 24

    with pytest.raises(GraphExtractionError, match="unknown evidence"):
        extract_graph(document, FakeExtractor(output=graph_json(block_id=fabricated_block)))


def test_reports_timeout_without_persisting_model_details(document: ParsedDocument) -> None:
    with pytest.raises(GraphExtractionError, match="timed out"):
        extract_graph(document, FakeExtractor(error=TimeoutError("provider secret")))


def test_store_round_trips_validated_graph(
    tmp_path, document: ParsedDocument
) -> None:
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_HASH}))
    folder = tmp_path / PDF_HASH
    folder.mkdir()
    (folder / "record.json").write_text(
        '{"document_id":"' + PDF_HASH + '","state":"ready",'
        '"page_count":1,"block_count":1,"error":null}'
    )
    (folder / "parsed.json").write_text(document.model_dump_json())
    artifact = extract_graph(document, FakeExtractor(output=graph_json()))

    versioned = store.save_graph(artifact)

    assert versioned.graph_version == 1
    assert store.get_graph(PDF_HASH, version=1) == versioned
    assert GraphArtifact.model_validate_json(
        (folder / "graphs" / "000001.json").read_bytes()
    ) == versioned


def test_store_reads_legacy_graph_as_version_one(
    tmp_path, document: ParsedDocument
) -> None:
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_HASH}))
    folder = tmp_path / PDF_HASH
    folder.mkdir()
    (folder / "record.json").write_text(
        '{"document_id":"' + PDF_HASH + '","state":"ready",'
        '"page_count":1,"block_count":1,"error":null}'
    )
    (folder / "parsed.json").write_text(document.model_dump_json())
    artifact = extract_graph(document, FakeExtractor(output=graph_json()))
    (folder / "graph.json").write_text(artifact.model_dump_json())

    loaded = store.get_graph(PDF_HASH)

    assert loaded is not None and loaded.graph_version == 1
    assert store.list_graph_versions(PDF_HASH).current_version == 1
