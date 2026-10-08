"""Minimal, evidence-linked graph contract for paper claims."""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evidencegraph.models import ParsedDocument

MAX_GRAPH_NODES = 25
MAX_GRAPH_RELATIONS = 40


class NodeType(StrEnum):
    PAPER = "paper"
    ACTOR = "actor"
    CONCEPT = "concept"
    ARTIFACT = "artifact"
    PROCESS = "process"
    OBSERVATION = "observation"
    CLAIM = "claim"
    CONTEXT = "context"

    # Version 1 compatibility. New extraction requests never expose these values.
    METHOD = "method"
    DATASET = "dataset"
    RESULT = "result"


class RelationType(StrEnum):
    INTRODUCES = "introduces"
    USES = "uses"
    STUDIES = "studies"
    APPLIES_TO = "applies_to"
    MEASURES = "measures"
    PRODUCES = "produces"
    REPORTS = "reports"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    COMPARES_WITH = "compares_with"
    BUILDS_ON = "builds_on"
    PART_OF = "part_of"

    # Version 1 compatibility. New extraction requests never expose this value.
    EVALUATED_ON = "evaluated_on"


class RelationStatus(StrEnum):
    DIRECT = "direct"
    CANDIDATE = "candidate"
    UNCONFIRMED = "unconfirmed"


V2_NODE_TYPES = frozenset(
    {
        NodeType.PAPER,
        NodeType.ACTOR,
        NodeType.CONCEPT,
        NodeType.ARTIFACT,
        NodeType.PROCESS,
        NodeType.OBSERVATION,
        NodeType.CLAIM,
        NodeType.CONTEXT,
    }
)

_RESEARCH_OBJECTS = frozenset(
    {
        NodeType.ACTOR,
        NodeType.CONCEPT,
        NodeType.ARTIFACT,
        NodeType.PROCESS,
        NodeType.CONTEXT,
    }
)

V2_RELATION_ENDPOINTS: dict[
    RelationType, tuple[frozenset[NodeType], frozenset[NodeType]]
] = {
    RelationType.INTRODUCES: (
        frozenset({NodeType.PAPER, NodeType.ACTOR}),
        frozenset(
            {NodeType.CONCEPT, NodeType.ARTIFACT, NodeType.PROCESS, NodeType.CLAIM}
        ),
    ),
    RelationType.USES: (
        frozenset(
            {NodeType.PAPER, NodeType.ACTOR, NodeType.ARTIFACT, NodeType.PROCESS}
        ),
        frozenset({NodeType.CONCEPT, NodeType.ARTIFACT, NodeType.PROCESS}),
    ),
    RelationType.STUDIES: (
        frozenset({NodeType.PAPER, NodeType.ACTOR, NodeType.PROCESS}),
        _RESEARCH_OBJECTS,
    ),
    RelationType.APPLIES_TO: (
        frozenset(
            {NodeType.CONCEPT, NodeType.ARTIFACT, NodeType.PROCESS, NodeType.CLAIM}
        ),
        frozenset(
            {
                NodeType.ACTOR,
                NodeType.CONCEPT,
                NodeType.ARTIFACT,
                NodeType.PROCESS,
                NodeType.CONTEXT,
            }
        ),
    ),
    RelationType.MEASURES: (
        frozenset({NodeType.ACTOR, NodeType.ARTIFACT, NodeType.PROCESS}),
        frozenset({NodeType.CONCEPT, NodeType.ARTIFACT, NodeType.CONTEXT}),
    ),
    RelationType.PRODUCES: (
        frozenset({NodeType.ACTOR, NodeType.ARTIFACT, NodeType.PROCESS}),
        frozenset({NodeType.ARTIFACT, NodeType.OBSERVATION, NodeType.CLAIM}),
    ),
    RelationType.REPORTS: (
        frozenset({NodeType.PAPER, NodeType.ACTOR}),
        frozenset({NodeType.OBSERVATION, NodeType.CLAIM}),
    ),
    RelationType.SUPPORTS: (
        frozenset(
            {
                NodeType.ARTIFACT,
                NodeType.PROCESS,
                NodeType.OBSERVATION,
                NodeType.CLAIM,
            }
        ),
        frozenset({NodeType.CLAIM}),
    ),
    RelationType.CONTRADICTS: (
        frozenset({NodeType.OBSERVATION, NodeType.CLAIM}),
        frozenset({NodeType.CLAIM}),
    ),
    RelationType.COMPARES_WITH: (_RESEARCH_OBJECTS, _RESEARCH_OBJECTS),
    RelationType.BUILDS_ON: (
        frozenset(
            {NodeType.PAPER, NodeType.CONCEPT, NodeType.ARTIFACT, NodeType.PROCESS}
            | {NodeType.CLAIM}
        ),
        frozenset(
            {NodeType.PAPER, NodeType.CONCEPT, NodeType.ARTIFACT, NodeType.PROCESS}
            | {NodeType.CLAIM}
        ),
    ),
    RelationType.PART_OF: (V2_NODE_TYPES, V2_NODE_TYPES),
}

V2_RELATION_TYPES = frozenset(V2_RELATION_ENDPOINTS)
V3_NODE_TYPES = V2_NODE_TYPES
V3_RELATION_ENDPOINTS = {
    **V2_RELATION_ENDPOINTS,
    RelationType.USES: (
        frozenset({NodeType.ACTOR, NodeType.ARTIFACT, NodeType.PROCESS}),
        frozenset({NodeType.CONCEPT, NodeType.ARTIFACT, NodeType.PROCESS}),
    ),
}
V3_RELATION_TYPES = frozenset(V3_RELATION_ENDPOINTS)


def relation_endpoints_are_valid(
    relation_type: RelationType,
    source_type: NodeType,
    target_type: NodeType,
    *,
    schema_version: Literal["2", "3"] = "3",
) -> bool:
    """Return whether one versioned ontology relation uses an allowed endpoint pair."""
    endpoints = (
        V2_RELATION_ENDPOINTS if schema_version == "2" else V3_RELATION_ENDPOINTS
    )
    allowed_sources, allowed_targets = endpoints[relation_type]
    return source_type in allowed_sources and target_type in allowed_targets


class EvidenceRef(BaseModel):
    """A block in one exact version of a source PDF."""

    model_config = ConfigDict(frozen=True)

    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    block_id: str = Field(pattern=r"^blk_[0-9a-f]{24}$")


class GraphNode(BaseModel):
    """An entity owned by one paper, with evidence for its name or meaning."""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(min_length=1)
    node_type: NodeType
    name: str = Field(min_length=1)
    domain_type: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
        description="Optional domain-specific subtype such as dataset, drug, or catalyst.",
    )
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    value: float | str | None = None
    unit: str | None = None
    uncertainty: str | None = None
    conditions: str | None = None

    @model_validator(mode="after")
    def keep_measurements_on_observations(self) -> GraphNode:
        measurement = (self.value, self.unit, self.uncertainty, self.conditions)
        if any(item is not None for item in measurement) and self.node_type not in {
            NodeType.OBSERVATION,
            NodeType.RESULT,
        }:
            raise ValueError("measurement fields require an observation node")
        return self


class GraphRelation(BaseModel):
    """A typed claim; status records the level of human verification."""

    model_config = ConfigDict(frozen=True)

    relation_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relation_type: RelationType
    domain_relation: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
        description="Optional domain-specific relation such as trained_on.",
    )
    status: RelationStatus
    evidence: tuple[EvidenceRef, ...]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_evidence_for_supported_status(self) -> GraphRelation:
        if self.status is not RelationStatus.UNCONFIRMED and not self.evidence:
            raise ValueError("direct and candidate relations require evidence references")
        return self


class GraphAnnotation(BaseModel):
    """A reviewable graph whose evidence can be checked against parsed PDFs."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1", "2", "3"] = "1"
    nodes: tuple[GraphNode, ...] = Field(min_length=1, max_length=MAX_GRAPH_NODES)
    relations: tuple[GraphRelation, ...] = Field(max_length=MAX_GRAPH_RELATIONS)

    @field_validator("schema_version", mode="before")
    @classmethod
    def normalize_legacy_schema_version(cls, value: object) -> object:
        """Read known historical spellings without emitting them for new artifacts."""
        if not isinstance(value, str):
            return value
        match = re.fullmatch(r"v?([123])(?:\.0){0,2}", value)
        return match.group(1) if match else value

    @model_validator(mode="after")
    def validate_graph_structure(self) -> GraphAnnotation:
        node_ids = [node.node_id for node in self.nodes]
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node ID")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("duplicate relation ID")
        known_nodes = set(node_ids)
        node_index = {node.node_id: node for node in self.nodes}
        for relation in self.relations:
            if relation.source_node_id not in known_nodes:
                raise ValueError(f"unknown source node: {relation.source_node_id}")
            if relation.target_node_id not in known_nodes:
                raise ValueError(f"unknown target node: {relation.target_node_id}")
        if self.schema_version in {"2", "3"}:
            self._validate_versioned_ontology(node_index)
        return self

    def _validate_versioned_ontology(
        self, node_index: dict[str, GraphNode]
    ) -> None:
        schema_version: Literal["2", "3"] = (
            "2" if self.schema_version == "2" else "3"
        )
        legacy_nodes = {NodeType.METHOD, NodeType.DATASET, NodeType.RESULT}
        if any(node.node_type in legacy_nodes for node in self.nodes):
            raise ValueError("ontology v2 graph contains a legacy node type")
        paper_count = sum(
            node.node_type is NodeType.PAPER for node in self.nodes
        )
        if paper_count != 1:
            raise ValueError("ontology v2 graph requires exactly one paper node")
        if any(
            relation.relation_type is RelationType.EVALUATED_ON
            for relation in self.relations
        ):
            raise ValueError("ontology v2 graph contains a legacy relation type")

        for relation in self.relations:
            source_type = node_index[relation.source_node_id].node_type
            target_type = node_index[relation.target_node_id].node_type
            if not relation_endpoints_are_valid(
                relation.relation_type,
                source_type,
                target_type,
                schema_version=schema_version,
            ):
                raise ValueError(
                    f"invalid ontology v{schema_version} endpoints for "
                    f"{relation.relation_type}: {source_type} -> {target_type}"
                )

    @property
    def verified_relations(self) -> tuple[GraphRelation, ...]:
        """Only direct claims are safe to display as verified facts."""
        return tuple(relation for relation in self.relations if relation.status is RelationStatus.DIRECT)

    def validate_against(self, documents: Iterable[ParsedDocument]) -> None:
        """Reject stale or fabricated evidence IDs before using an annotation."""
        document_list = tuple(documents)
        document_index = {document.source_sha256: document for document in document_list}
        if len(document_index) != len(document_list):
            raise ValueError("duplicate PDF version in parsed documents")
        known_blocks = {
            (document.source_sha256, block.block_id)
            for document in document_list
            for block in document.blocks
        }
        node_index = {node.node_id: node for node in self.nodes}

        def check_ref(ref: EvidenceRef) -> None:
            if (ref.source_sha256, ref.block_id) not in known_blocks:
                raise ValueError(f"unknown evidence block: {ref.source_sha256}/{ref.block_id}")

        for node in self.nodes:
            if node.document_sha256 not in document_index:
                raise ValueError(f"unknown node document: {node.document_sha256}")
            for ref in node.evidence:
                check_ref(ref)
                if ref.source_sha256 != node.document_sha256:
                    raise ValueError(f"node evidence belongs to another PDF: {node.node_id}")

        for relation in self.relations:
            for ref in relation.evidence:
                check_ref(ref)
            if relation.status is not RelationStatus.DIRECT:
                continue
            source = node_index[relation.source_node_id]
            target = node_index[relation.target_node_id]
            cited_documents = {ref.source_sha256 for ref in relation.evidence}
            required_documents = {source.document_sha256, target.document_sha256}
            if not required_documents.issubset(cited_documents):
                raise ValueError(f"direct relation lacks endpoint evidence: {relation.relation_id}")
