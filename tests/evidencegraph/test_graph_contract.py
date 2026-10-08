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


def test_ontology_v2_accepts_cross_domain_core_and_domain_types() -> None:
    evidence = (_ref(SOURCE_HASH, SOURCE_BLOCK),)
    annotation = GraphAnnotation(
        schema_version="2",
        nodes=(
            GraphNode(
                node_id="paper",
                node_type=NodeType.PAPER,
                name="Clinical study",
                document_sha256=SOURCE_HASH,
                evidence=evidence,
            ),
            GraphNode(
                node_id="intervention",
                node_type=NodeType.PROCESS,
                domain_type="medical_intervention",
                name="Treatment protocol",
                document_sha256=SOURCE_HASH,
                evidence=evidence,
            ),
            GraphNode(
                node_id="outcome",
                node_type=NodeType.OBSERVATION,
                domain_type="primary_outcome",
                name="Response rate",
                value=63.5,
                unit="percent",
                uncertainty="95% CI 58.1–68.9",
                conditions="after 12 weeks",
                document_sha256=SOURCE_HASH,
                evidence=evidence,
            ),
        ),
        relations=(
            GraphRelation(
                relation_id="paper_reports_outcome",
                source_node_id="paper",
                target_node_id="outcome",
                relation_type=RelationType.REPORTS,
                domain_relation="reports_primary_outcome",
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The paper reports the primary outcome.",
            ),
        ),
    )

    assert annotation.schema_version == "2"
    assert annotation.nodes[1].domain_type == "medical_intervention"
    assert annotation.nodes[2].value == 63.5
    assert annotation.relations[0].domain_relation == "reports_primary_outcome"


def test_ontology_v2_accepts_shared_patterns_across_disciplines() -> None:
    """One core ontology covers engineering, science, social, and medical papers."""
    evidence = (_ref(SOURCE_HASH, SOURCE_BLOCK),)

    def node(node_id: str, node_type: NodeType, name: str) -> GraphNode:
        return GraphNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            document_sha256=SOURCE_HASH,
            evidence=evidence,
        )

    annotation = GraphAnnotation(
        schema_version="2",
        nodes=(
            node("paper", NodeType.PAPER, "Cross-domain paper"),
            node("population", NodeType.ACTOR, "Target population"),
            node("procedure", NodeType.PROCESS, "Research procedure"),
            node("instrument", NodeType.ARTIFACT, "Measurement instrument"),
            node("construct", NodeType.CONCEPT, "Measured construct"),
            node("recommendation", NodeType.CLAIM, "Evidence-backed recommendation"),
            node("experiment", NodeType.PROCESS, "Validation experiment"),
        ),
        relations=(
            GraphRelation(
                relation_id="procedure_uses_instrument",
                source_node_id="procedure",
                target_node_id="instrument",
                relation_type=RelationType.USES,
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The research procedure uses the instrument.",
            ),
            GraphRelation(
                relation_id="procedure_applies_to_population",
                source_node_id="procedure",
                target_node_id="population",
                relation_type=RelationType.APPLIES_TO,
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The procedure applies to the target population.",
            ),
            GraphRelation(
                relation_id="recommendation_applies_to_population",
                source_node_id="recommendation",
                target_node_id="population",
                relation_type=RelationType.APPLIES_TO,
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The recommendation applies to the target population.",
            ),
            GraphRelation(
                relation_id="instrument_measures_construct",
                source_node_id="instrument",
                target_node_id="construct",
                relation_type=RelationType.MEASURES,
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The instrument measures the construct.",
            ),
            GraphRelation(
                relation_id="experiment_supports_recommendation",
                source_node_id="experiment",
                target_node_id="recommendation",
                relation_type=RelationType.SUPPORTS,
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The validation experiment supports the recommendation.",
            ),
            GraphRelation(
                relation_id="recommendation_builds_on_construct",
                source_node_id="recommendation",
                target_node_id="construct",
                relation_type=RelationType.BUILDS_ON,
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The recommendation builds on the established construct.",
            ),
            GraphRelation(
                relation_id="paper_reports_recommendation",
                source_node_id="paper",
                target_node_id="recommendation",
                relation_type=RelationType.REPORTS,
                status=RelationStatus.CANDIDATE,
                evidence=evidence,
                rationale="The paper reports the recommendation.",
            ),
        ),
    )

    assert len(annotation.relations) == 7


def test_ontology_v3_rejects_paper_uses_but_v2_history_remains_readable() -> None:
    evidence = (_ref(SOURCE_HASH, SOURCE_BLOCK),)
    paper = GraphNode(
        node_id="paper",
        node_type=NodeType.PAPER,
        name="Review paper",
        document_sha256=SOURCE_HASH,
        evidence=evidence,
    )
    instrument = GraphNode(
        node_id="instrument",
        node_type=NodeType.ARTIFACT,
        name="Discussed instrument",
        document_sha256=SOURCE_HASH,
        evidence=evidence,
    )
    relation = GraphRelation(
        relation_id="paper_uses_instrument",
        source_node_id="paper",
        target_node_id="instrument",
        relation_type=RelationType.USES,
        status=RelationStatus.CANDIDATE,
        evidence=evidence,
        rationale="Historical model output.",
    )

    historical = GraphAnnotation(
        schema_version="2",
        nodes=(paper, instrument),
        relations=(relation,),
    )
    assert historical.schema_version == "2"

    with pytest.raises(ValidationError, match="invalid ontology v3 endpoints"):
        GraphAnnotation(
            schema_version="3",
            nodes=(paper, instrument),
            relations=(relation,),
        )



def test_ontology_v2_rejects_invalid_relation_endpoints() -> None:
    evidence = (_ref(SOURCE_HASH, SOURCE_BLOCK),)
    paper = GraphNode(
        node_id="paper",
        node_type=NodeType.PAPER,
        name="Paper",
        document_sha256=SOURCE_HASH,
        evidence=evidence,
    )
    process = GraphNode(
        node_id="process",
        node_type=NodeType.PROCESS,
        name="Experiment",
        document_sha256=SOURCE_HASH,
        evidence=evidence,
    )
    with pytest.raises(ValidationError, match="invalid ontology v2 endpoints"):
        GraphAnnotation(
            schema_version="2",
            nodes=(paper, process),
            relations=(
                GraphRelation(
                    relation_id="invalid_report",
                    source_node_id="process",
                    target_node_id="paper",
                    relation_type=RelationType.REPORTS,
                    status=RelationStatus.CANDIDATE,
                    evidence=evidence,
                    rationale="This direction is not valid for reports.",
                ),
            ),
        )


@pytest.mark.parametrize("paper_count", (0, 2))
def test_ontology_v2_requires_exactly_one_paper_node(paper_count: int) -> None:
    evidence = (_ref(SOURCE_HASH, SOURCE_BLOCK),)
    papers = tuple(
        GraphNode(
            node_id=f"paper_{index}",
            node_type=NodeType.PAPER,
            name=f"Paper {index}",
            document_sha256=SOURCE_HASH,
            evidence=evidence,
        )
        for index in range(paper_count)
    )
    concept = GraphNode(
        node_id="concept",
        node_type=NodeType.CONCEPT,
        name="Research topic",
        document_sha256=SOURCE_HASH,
        evidence=evidence,
    )

    with pytest.raises(ValidationError, match="exactly one paper node"):
        GraphAnnotation(
            schema_version="2",
            nodes=(*papers, concept),
            relations=(),
        )


def test_measurement_fields_require_observation_node() -> None:
    with pytest.raises(ValidationError, match="measurement fields require"):
        GraphNode(
            node_id="concept",
            node_type=NodeType.CONCEPT,
            name="Temperature",
            value=37.2,
            unit="celsius",
            document_sha256=SOURCE_HASH,
            evidence=(_ref(SOURCE_HASH, SOURCE_BLOCK),),
        )


def test_ontology_v2_rejects_legacy_types_but_v1_remains_readable() -> None:
    legacy = GraphAnnotation(
        schema_version="1",
        nodes=(
            GraphNode(
                node_id="method",
                node_type=NodeType.METHOD,
                name="Legacy method",
                document_sha256=SOURCE_HASH,
                evidence=(_ref(SOURCE_HASH, SOURCE_BLOCK),),
            ),
        ),
        relations=(),
    )
    assert legacy.nodes[0].node_type is NodeType.METHOD

    with pytest.raises(ValidationError, match="legacy node type"):
        GraphAnnotation(
            schema_version="2",
            nodes=legacy.nodes,
            relations=(),
        )


@pytest.mark.parametrize(
    ("historical_version", "canonical_version"),
    (
        ("v1", "1"),
        ("1.0", "1"),
        ("1.0.0", "1"),
        ("v1.0.0", "1"),
        ("v2", "2"),
        ("2.0", "2"),
        ("2.0.0", "2"),
        ("v2.0.0", "2"),
        ("v3", "3"),
        ("3.0", "3"),
        ("3.0.0", "3"),
        ("v3.0.0", "3"),
    ),
)
def test_historical_schema_alias_is_normalized(
    historical_version: str, canonical_version: str
) -> None:
    annotation = GraphAnnotation.model_validate(
        {
            "schema_version": historical_version,
            "nodes": [
                {
                    "node_id": "paper",
                    "node_type": "paper",
                    "name": "Historical paper",
                    "document_sha256": SOURCE_HASH,
                    "evidence": [
                        {
                            "source_sha256": SOURCE_HASH,
                            "block_id": SOURCE_BLOCK,
                        }
                    ],
                }
            ],
            "relations": [],
        }
    )

    assert annotation.schema_version == canonical_version
    assert annotation.model_dump()["schema_version"] == canonical_version


@pytest.mark.parametrize("unsupported_version", ("1.1", "1.0.1", "4", "latest"))
def test_unknown_schema_versions_remain_rejected(unsupported_version: str) -> None:
    with pytest.raises(ValidationError, match="Input should be '1', '2' or '3'"):
        GraphAnnotation.model_validate(
            {
                "schema_version": unsupported_version,
                "nodes": [
                    {
                        "node_id": "paper",
                        "node_type": "paper",
                        "name": "Unsupported version",
                        "document_sha256": SOURCE_HASH,
                        "evidence": [
                            {
                                "source_sha256": SOURCE_HASH,
                                "block_id": SOURCE_BLOCK,
                            }
                        ],
                    }
                ],
                "relations": [],
            }
        )


def test_real_gold_fixture_shape() -> None:
    fixture = Path(__file__).resolve().parents[2] / "data" / "annotations" / "phase0_gold.json"
    annotation = GraphAnnotation.model_validate_json(fixture.read_text(encoding="utf-8"))
    assert len(annotation.verified_relations) >= 2
    assert any(relation.status is RelationStatus.UNCONFIRMED for relation in annotation.relations)
