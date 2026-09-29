"""Provider-neutral boundary for safe, evidence-linked graph extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evidencegraph.graph_contract import GraphAnnotation, RelationStatus
from evidencegraph.models import ParsedDocument


class GraphExtractionError(RuntimeError):
    """A model response could not be accepted as an evidence graph."""


class GraphUsage(BaseModel):
    """Provider-normalized token usage and a non-billing cost reference."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    thinking_tokens: int = Field(ge=0)
    tool_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    paid_standard_estimate_usd: float | None = Field(default=None, ge=0)
    pricing_basis: str | None = None


@dataclass(frozen=True)
class GraphExtractionResponse:
    raw_graph_json: str
    usage: GraphUsage | None = None


class GraphExtractor(Protocol):
    """Small interface implemented later by Gemini or another model provider."""

    name: str
    version: str

    def extract(self, document: ParsedDocument) -> str | GraphExtractionResponse:
        """Return a JSON graph candidate for one parsed document."""
        ...


class GraphArtifact(BaseModel):
    """Reproducible graph output plus the extractor identity that produced it."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    graph_version: int = Field(default=1, ge=1)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extractor_name: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    usage: GraphUsage | None = None
    graph: GraphAnnotation

    @model_validator(mode="after")
    def enforce_automated_extraction_boundary(self) -> GraphArtifact:
        if any(
            relation.status is RelationStatus.DIRECT for relation in self.graph.relations
        ):
            raise ValueError("automated graph cannot contain direct relations")
        if any(
            node.document_sha256 != self.document_sha256 for node in self.graph.nodes
        ):
            raise ValueError("automated graph node belongs to another PDF")
        evidence = (
            ref for node in self.graph.nodes for ref in node.evidence
        )
        relation_evidence = (
            ref for relation in self.graph.relations for ref in relation.evidence
        )
        if any(
            ref.source_sha256 != self.document_sha256
            for ref in (*evidence, *relation_evidence)
        ):
            raise ValueError("automated graph evidence belongs to another PDF")
        return self


def extract_graph(document: ParsedDocument, extractor: GraphExtractor) -> GraphArtifact:
    """Parse and evidence-check untrusted extractor output before persistence."""
    try:
        response = extractor.extract(document)
    except TimeoutError as exc:
        raise GraphExtractionError("Graph extractor timed out") from exc

    try:
        raw_graph = (
            response.raw_graph_json
            if isinstance(response, GraphExtractionResponse)
            else response
        )
        graph = GraphAnnotation.model_validate_json(raw_graph)
    except ValidationError as exc:
        details = _validation_error_summary(exc)
        raise GraphExtractionError(
            f"Graph extractor returned invalid structured output: {details}"
        ) from exc

    if any(relation.status is RelationStatus.DIRECT for relation in graph.relations):
        raise GraphExtractionError(
            "Automated extraction cannot create human-verified direct relations"
        )

    if any(node.document_sha256 != document.source_sha256 for node in graph.nodes):
        raise GraphExtractionError("Extracted graph contains a node from another PDF")

    if any(
        ref.source_sha256 != document.source_sha256
        for node in graph.nodes
        for ref in node.evidence
    ) or any(
        ref.source_sha256 != document.source_sha256
        for relation in graph.relations
        for ref in relation.evidence
    ):
        raise GraphExtractionError("Extracted graph contains evidence from another PDF")

    try:
        graph.validate_against((document,))
    except ValueError as exc:
        raise GraphExtractionError("Extracted graph cites unknown evidence") from exc

    return GraphArtifact(
        document_sha256=document.source_sha256,
        extractor_name=extractor.name,
        extractor_version=extractor.version,
        usage=response.usage if isinstance(response, GraphExtractionResponse) else None,
        graph=graph,
    )


def _validation_error_summary(exc: ValidationError) -> str:
    """Expose bounded schema diagnostics without model-provided values."""
    summaries = []
    for error in exc.errors(include_input=False, include_url=False)[:3]:
        location = ".".join(str(part) for part in error["loc"]) or "document"
        summaries.append(f"{location} [{error['type']}]: {error['msg']}")
    return "; ".join(summaries)
