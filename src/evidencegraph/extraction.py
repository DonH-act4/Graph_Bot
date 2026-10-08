"""Provider-neutral boundary for safe, evidence-linked graph extraction."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from evidencegraph.graph_contract import (
    GraphAnnotation,
    GraphNode,
    GraphRelation,
    RelationStatus,
    relation_endpoints_are_valid,
)
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


class GraphRepairWarning(BaseModel):
    """A bounded, persisted explanation of automatic graph recovery."""

    model_config = ConfigDict(frozen=True)

    code: Literal[
        "model_output_repaired",
        "invalid_relations_omitted",
        "unsupported_relations_omitted",
        "disconnected_nodes_omitted",
    ]
    message: str = Field(min_length=1, max_length=300)
    relation_ids: tuple[str, ...] = Field(default=(), max_length=3)


class GraphExtractor(Protocol):
    """Small interface implemented later by Gemini or another model provider."""

    name: str
    version: str

    def extract(self, document: ParsedDocument) -> str | GraphExtractionResponse:
        """Return a JSON graph candidate for one parsed document."""
        ...


class GraphRepairingExtractor(GraphExtractor, Protocol):
    """Optional extractor capability for one evidence-bounded repair attempt."""

    def repair(
        self,
        document: ParsedDocument,
        raw_graph_json: str,
        validation_error: str,
    ) -> str | GraphExtractionResponse:
        """Return one corrected full graph candidate without rereading the whole PDF."""
        ...


GraphProgressCallback = Callable[..., None]


class GraphProgressExtractor(GraphExtractor, Protocol):
    """Optional capability for providers with multiple observable extraction steps."""

    def extract_with_progress(
        self,
        document: ParsedDocument,
        progress_callback: GraphProgressCallback,
    ) -> str | GraphExtractionResponse:
        """Return a graph candidate while reporting provider-specific progress."""
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
    warnings: tuple[GraphRepairWarning, ...] = ()
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


def extract_graph(
    document: ParsedDocument,
    extractor: GraphExtractor,
    *,
    progress_callback: GraphProgressCallback | None = None,
) -> GraphArtifact:
    """Parse and evidence-check untrusted extractor output before persistence."""
    report_progress = _progress_reporter(progress_callback)
    try:
        extract_with_progress = getattr(extractor, "extract_with_progress", None)
        if report_progress is not None and callable(extract_with_progress):
            candidate_response = extract_with_progress(document, report_progress)
        else:
            candidate_response = extractor.extract(document)
    except TimeoutError as exc:
        raise GraphExtractionError("Graph extractor timed out") from exc
    if not isinstance(candidate_response, (str, GraphExtractionResponse)):
        raise GraphExtractionError("Graph extractor returned an unsupported response type")
    response: str | GraphExtractionResponse = candidate_response

    raw_graph = (
        response.raw_graph_json
        if isinstance(response, GraphExtractionResponse)
        else response
    )
    usage = response.usage if isinstance(response, GraphExtractionResponse) else None
    warnings: list[GraphRepairWarning] = []
    if report_progress is not None:
        report_progress("ontology_validation", 60, None)
    try:
        graph = GraphAnnotation.model_validate_json(raw_graph)
    except ValidationError as exc:
        details = _validation_error_summary(exc)
        if not _is_endpoint_only_error(exc):
            raise GraphExtractionError(
                f"Graph extractor returned invalid structured output: {details}"
            ) from exc
        graph = None
        salvage_json: str = raw_graph
        if report_progress is not None:
            report_progress("ontology_repair", 75, None)
        repair = getattr(extractor, "repair", None)
        if callable(repair):
            try:
                repaired = repair(document, raw_graph, details)
                if isinstance(repaired, GraphExtractionResponse):
                    salvage_json = repaired.raw_graph_json
                    usage = _merge_usage(usage, repaired.usage)
                elif isinstance(repaired, str):
                    salvage_json = repaired
                else:
                    raise GraphExtractionError(
                        "Graph repair returned an unsupported response type"
                    )
                try:
                    graph = GraphAnnotation.model_validate_json(salvage_json)
                except ValidationError as repair_error:
                    if not _is_endpoint_only_error(repair_error):
                        salvage_json = raw_graph
                else:
                    warnings.append(
                        GraphRepairWarning(
                            code="model_output_repaired",
                            message=(
                                "One or more invalid ontology relations were repaired "
                                "and revalidated."
                            ),
                        )
                    )
            except GraphExtractionError:
                salvage_json = raw_graph
        if graph is None:
            graph, omitted_ids = _omit_invalid_endpoint_relations(salvage_json)
            warnings.append(
                GraphRepairWarning(
                    code="invalid_relations_omitted",
                    message=(
                        f"Omitted {len(omitted_ids)} relation(s) whose endpoint types "
                        "could not be repaired safely."
                    ),
                    relation_ids=omitted_ids,
                )
            )

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

    graph, unsupported_use_ids = _omit_unsupported_use_relations(graph, document)
    if unsupported_use_ids:
        warnings.append(
            GraphRepairWarning(
                code="unsupported_relations_omitted",
                message=(
                    f"Omitted {len(unsupported_use_ids)} uses relation(s) because "
                    "their cited source text did not explicitly describe use."
                ),
                relation_ids=unsupported_use_ids,
            )
        )

    graph, disconnected_node_count = _prune_nodes_disconnected_from_paper(graph)
    if disconnected_node_count:
        warnings.append(
            GraphRepairWarning(
                code="disconnected_nodes_omitted",
                message=(
                    f"Omitted {disconnected_node_count} node(s) that were not connected "
                    "to the source paper by any relationship path."
                ),
            )
        )

    if report_progress is not None:
        report_progress("saving_graph", 95, None)

    return GraphArtifact(
        document_sha256=document.source_sha256,
        extractor_name=extractor.name,
        extractor_version=extractor.version,
        usage=usage,
        warnings=tuple(warnings),
        graph=graph,
    )


def _progress_reporter(
    callback: GraphProgressCallback | None,
) -> Callable[[str, int, str | None], None] | None:
    """Adapt historical two-argument callbacks to progress details safely."""
    if callback is None:
        return None
    try:
        parameters = tuple(signature(callback).parameters.values())
    except (TypeError, ValueError):
        accepts_detail = True
    else:
        accepts_detail = any(
            parameter.kind is Parameter.VAR_POSITIONAL for parameter in parameters
        ) or sum(
            parameter.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}
            for parameter in parameters
        ) >= 3

    if accepts_detail:
        return lambda stage, percent, detail: callback(stage, percent, detail)
    return lambda stage, percent, _detail: callback(stage, percent)


_EXPLICIT_USE_ACTION = re.compile(
    r"(?:"
    r"\b(?:use[sd]?|using|employ(?:ed|s|ing)?|utili[sz](?:e|ed|es|ing)|"
    r"leverag(?:e|ed|es|ing)|adopt(?:ed|s|ing)?)\b|"
    r"\b(?:train(?:ed|s|ing)?|implement(?:ed|s|ing)?|conduct(?:ed|s|ing)?)\s+"
    r"(?:on|with|using)\b|"
    r"(?:使用|采用|利用|运用|借助)"
    r")",
    flags=re.IGNORECASE,
)


def _omit_unsupported_use_relations(
    graph: GraphAnnotation,
    document: ParsedDocument,
) -> tuple[GraphAnnotation, tuple[str, ...]]:
    """Omit a few ``uses`` edges whose cited text never states a use action.

    This deliberately checks only the narrow predicate claim. It does not try to
    infer domain semantics or replace model/evaluation based quality measurement.
    """
    block_text = {block.block_id: block.text for block in document.blocks}
    kept: list[GraphRelation] = []
    omitted: list[str] = []
    for relation in graph.relations:
        if relation.relation_type.value != "uses":
            kept.append(relation)
            continue
        cited_text = "\n".join(block_text[ref.block_id] for ref in relation.evidence)
        if _EXPLICIT_USE_ACTION.search(cited_text):
            kept.append(relation)
        else:
            omitted.append(_bounded_relation_id(relation.relation_id))

    if not omitted:
        return graph, ()
    if len(omitted) > 3:
        raise GraphExtractionError(
            "Graph extractor returned too many uses relations without explicit "
            f"usage evidence ({len(omitted)} unsupported)"
        )
    if not kept:
        raise GraphExtractionError(
            "Graph extractor returned only uses relations without explicit usage evidence"
        )
    return graph.model_copy(update={"relations": tuple(kept)}), tuple(omitted)


def _prune_nodes_disconnected_from_paper(
    graph: GraphAnnotation,
) -> tuple[GraphAnnotation, int]:
    """Keep only the evidence graph component anchored at its paper node."""
    paper_ids = {
        node.node_id for node in graph.nodes if node.node_type.value == "paper"
    }
    if len(paper_ids) != 1:
        return graph, 0

    adjacency: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for relation in graph.relations:
        adjacency[relation.source_node_id].add(relation.target_node_id)
        adjacency[relation.target_node_id].add(relation.source_node_id)

    reachable = set(paper_ids)
    pending = list(paper_ids)
    while pending:
        node_id = pending.pop()
        for neighbor_id in adjacency[node_id] - reachable:
            reachable.add(neighbor_id)
            pending.append(neighbor_id)

    disconnected_count = len(graph.nodes) - len(reachable)
    if disconnected_count == 0:
        return graph, 0
    kept_nodes = tuple(node for node in graph.nodes if node.node_id in reachable)
    kept_relations = tuple(
        relation for relation in graph.relations
        if relation.source_node_id in reachable and relation.target_node_id in reachable
    )
    return graph.model_copy(
        update={"nodes": kept_nodes, "relations": kept_relations}
    ), disconnected_count


def _is_endpoint_only_error(exc: ValidationError) -> bool:
    errors = exc.errors(include_input=False, include_url=False)
    return bool(errors) and all(
        "invalid ontology v" in error["msg"] and " endpoints" in error["msg"]
        for error in errors
    )


def _omit_invalid_endpoint_relations(
    raw_graph_json: str,
) -> tuple[GraphAnnotation, tuple[str, ...]]:
    """Salvage a mostly-valid v2 graph by omitting only invalid endpoint pairs."""
    try:
        payload = json.loads(raw_graph_json)
        if not isinstance(payload, dict):
            raise ValueError("salvage requires a versioned ontology")
        raw_version = payload.get("schema_version")
        if raw_version in {"2", "v2", "2.0"}:
            schema_version: Literal["2", "3"] = "2"
        elif raw_version in {"3", "v3", "3.0"}:
            schema_version = "3"
        else:
            raise ValueError("salvage requires ontology v2 or v3")
        raw_nodes = payload.get("nodes")
        raw_relations = payload.get("relations")
        if not isinstance(raw_nodes, list) or not isinstance(raw_relations, list):
            raise ValueError("graph candidate is not structurally salvageable")
        nodes = tuple(GraphNode.model_validate(node) for node in raw_nodes)
        node_index = {node.node_id: node for node in nodes}
        if len(node_index) != len(nodes):
            raise ValueError("duplicate node ID")
        relations = tuple(GraphRelation.model_validate(item) for item in raw_relations)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise GraphExtractionError(
            "Graph extractor returned invalid structured output that could not be repaired"
        ) from exc

    kept: list[GraphRelation] = []
    omitted: list[str] = []
    invalid_pairs: Counter[tuple[str, str, str]] = Counter()
    for relation in relations:
        source = node_index.get(relation.source_node_id)
        target = node_index.get(relation.target_node_id)
        if source is None or target is None:
            raise GraphExtractionError(
                "Graph extractor returned a relation with an unknown endpoint"
            )
        if relation_endpoints_are_valid(
            relation.relation_type,
            source.node_type,
            target.node_type,
            schema_version=schema_version,
        ):
            kept.append(relation)
        else:
            omitted.append(_bounded_relation_id(relation.relation_id))
            invalid_pairs[
                (
                    relation.relation_type.value,
                    source.node_type.value,
                    target.node_type.value,
                )
            ] += 1

    total = len(relations)
    if (
        not omitted
        or len(omitted) > 3
        or total == 0
        or len(omitted) * 4 > total
        or not kept
    ):
        endpoint_summary = ", ".join(
            f"{relation}:{source}->{target} x{count}"
            for (relation, source, target), count in invalid_pairs.most_common(4)
        )
        raise GraphExtractionError(
            "Graph extractor returned too many invalid ontology relations to recover "
            f"safely ({len(omitted)}/{total} invalid; {endpoint_summary})"
        )
    try:
        graph = GraphAnnotation(
            schema_version=schema_version,
            nodes=nodes,
            relations=tuple(kept),
        )
    except ValidationError as exc:
        raise GraphExtractionError(
            "Graph extractor returned invalid structured output that could not be repaired"
        ) from exc
    return graph, tuple(omitted)


def _bounded_relation_id(value: str) -> str:
    rendered = "".join(character for character in value if character.isalnum() or character in "_.:-")
    return rendered[:80] or "relation"


def _merge_usage(left: GraphUsage | None, right: GraphUsage | None) -> GraphUsage | None:
    if left is None:
        return right
    if right is None:
        return left
    left_estimate = left.paid_standard_estimate_usd
    right_estimate = right.paid_standard_estimate_usd
    estimate = (
        left_estimate + right_estimate
        if left_estimate is not None and right_estimate is not None
        else None
    )
    pricing_basis = left.pricing_basis if left.pricing_basis == right.pricing_basis else None
    return GraphUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        thinking_tokens=left.thinking_tokens + right.thinking_tokens,
        tool_tokens=left.tool_tokens + right.tool_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        paid_standard_estimate_usd=estimate,
        pricing_basis=pricing_basis,
    )


def _validation_error_summary(exc: ValidationError) -> str:
    """Expose bounded schema diagnostics without model-provided values."""
    summaries = []
    for error in exc.errors(include_input=False, include_url=False)[:3]:
        location = ".".join(str(part) for part in error["loc"]) or "document"
        summaries.append(f"{location} [{error['type']}]: {error['msg']}")
    return "; ".join(summaries)
