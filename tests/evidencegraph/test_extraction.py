from __future__ import annotations

import json
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


def v2_graph_json(*, valid_relations: int = 3, invalid_relations: int = 1) -> str:
    nodes = [
        {
            "node_id": "paper",
            "node_type": "paper",
            "name": "Example paper",
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        },
        {
            "node_id": "claim",
            "node_type": "claim",
            "name": "Main conclusion",
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        },
        {
            "node_id": "actor",
            "node_type": "actor",
            "name": "Study population",
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        },
    ]
    relations = [
        {
            "relation_id": f"reports-{index}",
            "source_node_id": "paper",
            "target_node_id": "claim",
            "relation_type": "reports",
            "status": "candidate",
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            "rationale": "The cited block states the conclusion.",
        }
        for index in range(valid_relations)
    ]
    relations.extend(
        {
            "relation_id": f"invalid-introduces-{index}",
            "source_node_id": "paper",
            "target_node_id": "actor",
            "relation_type": "introduces",
            "status": "candidate",
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            "rationale": "The cited block names the population.",
        }
        for index in range(invalid_relations)
    )
    return json.dumps({"schema_version": "2", "nodes": nodes, "relations": relations})


def v3_uses_graph_json(*, uses_relations: int = 1) -> str:
    nodes = [
        {
            "node_id": "paper",
            "node_type": "paper",
            "name": "Example paper",
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        },
        {
            "node_id": "claim",
            "node_type": "claim",
            "name": "Main conclusion",
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        },
        {
            "node_id": "process",
            "node_type": "process",
            "name": "Study workflow",
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        },
        {
            "node_id": "artifact",
            "node_type": "artifact",
            "name": "Tool X",
            "document_sha256": PDF_HASH,
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
        },
    ]
    relations = [
        {
            "relation_id": "reports-claim",
            "source_node_id": "paper",
            "target_node_id": "claim",
            "relation_type": "reports",
            "status": "candidate",
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            "rationale": "The paper states the conclusion.",
        },
        {
            "relation_id": "studies-process",
            "source_node_id": "paper",
            "target_node_id": "process",
            "relation_type": "studies",
            "status": "candidate",
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            "rationale": "The paper studies the workflow.",
        },
    ]
    relations.extend(
        {
            "relation_id": f"uses-tool-{index}",
            "source_node_id": "process",
            "target_node_id": "artifact",
            "relation_type": "uses",
            "status": "candidate",
            "evidence": [{"source_sha256": PDF_HASH, "block_id": BLOCK_ID}],
            "rationale": "The workflow uses the tool.",
        }
        for index in range(uses_relations)
    )
    return json.dumps({"schema_version": "3", "nodes": nodes, "relations": relations})


@dataclass
class RepairingExtractor:
    initial: str
    repaired: str
    name: str = "repairing-test-extractor"
    version: str = "1"
    repair_calls: int = 0

    def extract(self, _document: ParsedDocument) -> str:
        return self.initial

    def repair(
        self,
        _document: ParsedDocument,
        _raw_graph_json: str,
        _validation_error: str,
    ) -> str:
        self.repair_calls += 1
        return self.repaired


def test_accepts_evidence_linked_candidate(document: ParsedDocument) -> None:
    progress: list[tuple[str, int]] = []
    artifact = extract_graph(
        document,
        FakeExtractor(output=graph_json()),
        progress_callback=lambda stage, percent: progress.append((stage, percent)),
    )

    assert artifact.document_sha256 == PDF_HASH
    assert artifact.extractor_name == "deterministic-test-extractor"
    assert artifact.graph.relations[0].status == "candidate"
    assert progress == [("ontology_validation", 60), ("saving_graph", 95)]


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


def test_repairs_one_invalid_endpoint_relation_once(document: ParsedDocument) -> None:
    extractor = RepairingExtractor(
        initial=v2_graph_json(valid_relations=3, invalid_relations=1),
        repaired=v2_graph_json(valid_relations=3, invalid_relations=0),
    )

    progress: list[tuple[str, int]] = []
    artifact = extract_graph(
        document,
        extractor,
        progress_callback=lambda stage, percent: progress.append((stage, percent)),
    )

    assert extractor.repair_calls == 1
    assert len(artifact.graph.relations) == 3
    assert artifact.warnings[0].code == "model_output_repaired"
    assert progress == [
        ("ontology_validation", 60),
        ("ontology_repair", 75),
        ("saving_graph", 95),
    ]


def test_salvages_a_partially_repaired_graph_instead_of_reverting_to_original(
    document: ParsedDocument,
) -> None:
    extractor = RepairingExtractor(
        initial=v2_graph_json(valid_relations=1, invalid_relations=3),
        repaired=v2_graph_json(valid_relations=5, invalid_relations=1),
    )

    artifact = extract_graph(document, extractor)

    assert extractor.repair_calls == 1
    assert len(artifact.graph.relations) == 5
    assert artifact.warnings[0].code == "invalid_relations_omitted"
    assert artifact.warnings[0].relation_ids == ("invalid-introduces-0",)


def test_omits_one_unrepairable_relation_from_mostly_valid_graph(
    document: ParsedDocument,
) -> None:
    artifact = extract_graph(
        document,
        FakeExtractor(output=v2_graph_json(valid_relations=3, invalid_relations=1)),
    )

    assert len(artifact.graph.relations) == 3
    assert artifact.warnings[0].code == "invalid_relations_omitted"
    assert artifact.warnings[0].relation_ids == ("invalid-introduces-0",)


def test_omits_uses_relation_when_evidence_only_mentions_entities(
    document: ParsedDocument,
) -> None:
    artifact = extract_graph(
        document,
        FakeExtractor(output=v3_uses_graph_json()),
    )

    assert [relation.relation_id for relation in artifact.graph.relations] == [
        "reports-claim",
        "studies-process",
    ]
    assert artifact.warnings[0].code == "unsupported_relations_omitted"
    assert artifact.warnings[0].relation_ids == ("uses-tool-0",)


@pytest.mark.parametrize(
    "evidence_text",
    [
        "The experiment used Tool X to measure the samples.",
        "研究采用 Tool X 测量样本。",
    ],
)
def test_keeps_uses_relation_when_evidence_explicitly_states_use(
    document: ParsedDocument,
    evidence_text: str,
) -> None:
    supported_document = document.model_copy(
        update={
            "blocks": (
                document.blocks[0].model_copy(update={"text": evidence_text}),
            )
        }
    )

    artifact = extract_graph(
        supported_document,
        FakeExtractor(output=v3_uses_graph_json()),
    )

    assert len(artifact.graph.relations) == 3
    assert artifact.warnings == ()


def test_prunes_nodes_disconnected_from_the_paper_component(
    document: ParsedDocument,
) -> None:
    artifact = extract_graph(
        document,
        FakeExtractor(output=v2_graph_json(valid_relations=1, invalid_relations=0)),
    )

    assert {node.node_id for node in artifact.graph.nodes} == {"paper", "claim"}
    assert artifact.warnings[0].code == "disconnected_nodes_omitted"
    assert "1 node(s)" in artifact.warnings[0].message


def test_prunes_edges_inside_a_disconnected_component(document: ParsedDocument) -> None:
    payload = json.loads(v2_graph_json(valid_relations=1, invalid_relations=0))
    claim = next(node for node in payload["nodes"] if node["node_id"] == "claim")
    payload["nodes"].extend([
        {**claim, "node_id": "isolated_a", "name": "Isolated finding A"},
        {**claim, "node_id": "isolated_b", "name": "Isolated finding B"},
    ])
    payload["relations"].append({
        **payload["relations"][0],
        "relation_id": "isolated_relation",
        "source_node_id": "isolated_a",
        "target_node_id": "isolated_b",
        "relation_type": "contradicts",
    })

    artifact = extract_graph(document, FakeExtractor(output=json.dumps(payload)))

    assert {node.node_id for node in artifact.graph.nodes} == {"paper", "claim"}
    assert len(artifact.graph.relations) == 1
    assert artifact.warnings[0].code == "disconnected_nodes_omitted"
    # A serialized artifact must also survive the final persistence boundary.
    assert GraphArtifact.model_validate_json(artifact.model_dump_json()) == artifact


def test_rejects_graph_with_too_many_uses_relations_without_usage_evidence(
    document: ParsedDocument,
) -> None:
    with pytest.raises(
        GraphExtractionError,
        match=r"too many uses relations without explicit usage evidence \(4 unsupported\)",
    ):
        extract_graph(
            document,
            FakeExtractor(output=v3_uses_graph_json(uses_relations=4)),
        )


def test_rejects_graph_when_invalid_relations_are_not_a_small_minority(
    document: ParsedDocument,
) -> None:
    with pytest.raises(
        GraphExtractionError,
        match=r"too many invalid ontology relations.*1/2 invalid.*introduces:paper->actor x1",
    ):
        extract_graph(
            document,
            FakeExtractor(output=v2_graph_json(valid_relations=1, invalid_relations=1)),
        )


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
