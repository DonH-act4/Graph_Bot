from pathlib import Path

import pytest
from pydantic import ValidationError

from evidencegraph.graph_contract import (
    EvidenceRef,
    GraphAnnotation,
    GraphNode,
    GraphRelation,
    NodeType,
    RelationStatus,
    RelationType,
)
from evidencegraph.ingestion import (
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    SourceLocation,
)

SOURCE_HASH = "a" * 64
TARGET_HASH = "b" * 64
SOURCE_BLOCK = "blk_" + "1" * 24
TARGET_BLOCK = "blk_" + "2" * 24


def _ref(source_hash: str, block_id: str) -> EvidenceRef:
    return EvidenceRef(source_sha256=source_hash, block_id=block_id)


def _document(source_hash: str, block_id: str) -> ParsedDocument:
    return ParsedDocument(
        source_filename="paper.pdf",
        source_sha256=source_hash,
        parser_version="2.126.0",
        page_count=1,
        blocks=(
            ParsedBlock(
                block_id=block_id,
                source_ref="#/texts/0",
                label="text",
                text="A cited passage",
                locations=(
                    SourceLocation(
                        page_number=1,
                        bounding_box=BoundingBox(
                            left=1,
                            top=2,
                            right=3,
                            bottom=0,
                            coordinate_origin="BOTTOMLEFT",
                        ),
                        character_start=0,
                        character_end=15,
                    ),
                ),
            ),
        ),
    )


def _annotation(*, status: RelationStatus = RelationStatus.DIRECT) -> GraphAnnotation:
    return GraphAnnotation(
        nodes=(
            GraphNode(
                node_id="source",
                node_type=NodeType.PAPER,
                name="Source paper",
                document_sha256=SOURCE_HASH,
                evidence=(_ref(SOURCE_HASH, SOURCE_BLOCK),),
            ),
            GraphNode(
                node_id="target",
                node_type=NodeType.PAPER,
                name="Target paper",
                document_sha256=TARGET_HASH,
                evidence=(_ref(TARGET_HASH, TARGET_BLOCK),),
            ),
        ),
        relations=(
            GraphRelation(
                relation_id="builds_on",
                source_node_id="source",
                target_node_id="target",
                relation_type=RelationType.BUILDS_ON,
                status=status,
                evidence=(_ref(SOURCE_HASH, SOURCE_BLOCK), _ref(TARGET_HASH, TARGET_BLOCK)),
                rationale="The source cites the target.",
            ),
        ),
    )


def test_direct_cross_paper_relation_resolves_both_pdf_versions() -> None:
    annotation = _annotation()
    annotation.validate_against(
        (_document(SOURCE_HASH, SOURCE_BLOCK), _document(TARGET_HASH, TARGET_BLOCK))
    )
    assert len(annotation.verified_relations) == 1


def test_candidate_is_not_a_verified_fact() -> None:
    annotation = _annotation(status=RelationStatus.CANDIDATE)
    assert annotation.verified_relations == ()


def test_unconfirmed_can_have_no_evidence_but_is_not_verified() -> None:
    relation = GraphRelation(
        relation_id="question",
        source_node_id="source",
        target_node_id="target",
        relation_type=RelationType.BUILDS_ON,
        status=RelationStatus.UNCONFIRMED,
        evidence=(),
        rationale="The corpus does not establish this relation.",
    )
    assert relation.evidence == ()


def test_direct_relation_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="require evidence"):
        GraphRelation(
            relation_id="unsupported",
            source_node_id="source",
            target_node_id="target",
            relation_type=RelationType.BUILDS_ON,
            status=RelationStatus.DIRECT,
            evidence=(),
            rationale="No citation.",
        )


def test_relation_to_missing_node_is_rejected() -> None:
    valid = _annotation()
    with pytest.raises(ValidationError, match="unknown target node"):
        GraphAnnotation(
            nodes=valid.nodes,
            relations=(valid.relations[0].model_copy(update={"target_node_id": "missing"}),),
        )


def test_rejects_graphs_over_product_size_limits() -> None:
    valid = _annotation()
    repeated_nodes = tuple(
        valid.nodes[0].model_copy(update={"node_id": f"node-{index}"})
        for index in range(26)
    )
    with pytest.raises(ValidationError, match="at most 25 items"):
        GraphAnnotation(nodes=repeated_nodes, relations=())

    repeated_relations = tuple(
        valid.relations[0].model_copy(update={"relation_id": f"relation-{index}"})
        for index in range(41)
    )
    with pytest.raises(ValidationError, match="at most 40 items"):
        GraphAnnotation(nodes=valid.nodes, relations=repeated_relations)


def test_unknown_evidence_block_is_rejected() -> None:
    annotation = _annotation()
    with pytest.raises(ValueError, match="unknown evidence block"):
        annotation.validate_against(
            (_document(SOURCE_HASH, SOURCE_BLOCK), _document(TARGET_HASH, "blk_" + "3" * 24))
        )


def test_direct_cross_paper_relation_needs_evidence_from_both_papers() -> None:
    valid = _annotation()
    relation = valid.relations[0].model_copy(update={"evidence": (_ref(SOURCE_HASH, SOURCE_BLOCK),)})
    annotation = GraphAnnotation(nodes=valid.nodes, relations=(relation,))
    with pytest.raises(ValueError, match="lacks endpoint evidence"):
        annotation.validate_against(
            (_document(SOURCE_HASH, SOURCE_BLOCK), _document(TARGET_HASH, TARGET_BLOCK))
        )


def test_node_cannot_claim_evidence_from_another_pdf() -> None:
    valid = _annotation()
    wrong_node = valid.nodes[0].model_copy(update={"evidence": (_ref(TARGET_HASH, TARGET_BLOCK),)})
    annotation = GraphAnnotation(nodes=(wrong_node, valid.nodes[1]), relations=valid.relations)
    with pytest.raises(ValueError, match="belongs to another PDF"):
        annotation.validate_against(
            (_document(SOURCE_HASH, SOURCE_BLOCK), _document(TARGET_HASH, TARGET_BLOCK))
        )


def test_real_gold_fixture_shape() -> None:
    fixture = Path(__file__).resolve().parents[2] / "data" / "annotations" / "phase0_gold.json"
    annotation = GraphAnnotation.model_validate_json(fixture.read_text(encoding="utf-8"))
    assert len(annotation.verified_relations) >= 2
    assert any(relation.status is RelationStatus.UNCONFIRMED for relation in annotation.relations)
