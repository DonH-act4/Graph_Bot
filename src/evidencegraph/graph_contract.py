"""Minimal, evidence-linked graph contract for paper claims."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidencegraph.models import ParsedDocument

MAX_GRAPH_NODES = 25
MAX_GRAPH_RELATIONS = 40


class NodeType(StrEnum):
    PAPER = "paper"
    METHOD = "method"
    DATASET = "dataset"
    RESULT = "result"
    CLAIM = "claim"


class RelationType(StrEnum):
    INTRODUCES = "introduces"
    USES = "uses"
    EVALUATED_ON = "evaluated_on"
    REPORTS = "reports"
    BUILDS_ON = "builds_on"


class RelationStatus(StrEnum):
    DIRECT = "direct"
    CANDIDATE = "candidate"
    UNCONFIRMED = "unconfirmed"


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
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)


class GraphRelation(BaseModel):
    """A typed claim; status records the level of human verification."""

    model_config = ConfigDict(frozen=True)

    relation_id: str = Field(min_length=1)
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relation_type: RelationType
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

    schema_version: str = "1"
    nodes: tuple[GraphNode, ...] = Field(min_length=1, max_length=MAX_GRAPH_NODES)
    relations: tuple[GraphRelation, ...] = Field(max_length=MAX_GRAPH_RELATIONS)

    @model_validator(mode="after")
    def validate_graph_structure(self) -> GraphAnnotation:
        node_ids = [node.node_id for node in self.nodes]
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate node ID")
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("duplicate relation ID")
        known_nodes = set(node_ids)
        for relation in self.relations:
            if relation.source_node_id not in known_nodes:
                raise ValueError(f"unknown source node: {relation.source_node_id}")
            if relation.target_node_id not in known_nodes:
                raise ValueError(f"unknown target node: {relation.target_node_id}")
        return self

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
