"""Groq free-tier adapter with evidence-block chunking and deterministic graph merge."""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from evidencegraph.extraction import (
    GraphExtractionError,
    GraphExtractionResponse,
    GraphProgressCallback,
    GraphUsage,
    _is_endpoint_only_error,
    _omit_invalid_endpoint_relations,
)
from evidencegraph.gemini_extractor import (
    _SYSTEM_INSTRUCTION,
    ONTOLOGY_PROMPT_VERSION,
    _allowed_relation_endpoint_text,
    _build_prompt,
    _provider_graph_schema,
    _render_block,
)
from evidencegraph.graph_contract import (
    MAX_GRAPH_NODES,
    MAX_GRAPH_RELATIONS,
    EvidenceRef,
    GraphAnnotation,
    GraphNode,
    GraphRelation,
    NodeType,
    RelationStatus,
)
from evidencegraph.models import ParsedBlock, ParsedDocument

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MAX_CHUNK_CHARS = 6_000
DEFAULT_CONTEXT_CHARS = 1_500
DEFAULT_MAX_COMPLETION_TOKENS = 2_048
GROQ_CHUNKING_VERSION = "section-chunks-v2"


class _HttpClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...


class GroqGraphExtractor:
    """Extract a paper graph in rate-limit-aware sections using Groq structured output."""

    name = "groq"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: _HttpClient | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        context_chars: int = DEFAULT_CONTEXT_CHARS,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        max_attempts: int = 3,
    ) -> None:
        normalized_model = model.removeprefix("groq/").strip()
        if not normalized_model:
            raise ValueError("Groq model name is required")
        if client is None and not api_key:
            raise ValueError("Groq API key is required")
        if max_chunk_chars < 1_000:
            raise ValueError("Groq chunk budget must be at least 1,000 characters")
        if context_chars < 0:
            raise ValueError("Groq context budget cannot be negative")
        if max_completion_tokens < 256:
            raise ValueError("Groq output budget must be at least 256 tokens")
        if max_attempts < 1:
            raise ValueError("Groq request attempts must be positive")

        self.model = normalized_model
        self.version = (
            f"{normalized_model}:{ONTOLOGY_PROMPT_VERSION}:{GROQ_CHUNKING_VERSION}"
        )
        self._api_key = api_key or ""
        self._client = client or httpx.Client()
        self._sleeper = sleeper
        self._max_chunk_chars = max_chunk_chars
        self._context_chars = context_chars
        self._max_completion_tokens = max_completion_tokens
        self._max_attempts = max_attempts

    def extract(self, document: ParsedDocument) -> GraphExtractionResponse:
        """Extract without observable progress for protocol compatibility."""
        return self.extract_with_progress(document, lambda _stage, _percent, _detail: None)

    def extract_with_progress(
        self,
        document: ParsedDocument,
        progress_callback: GraphProgressCallback,
    ) -> GraphExtractionResponse:
        chunks = _chunk_document_blocks(
            document.blocks,
            max_chunk_chars=self._max_chunk_chars,
            context_chars=self._context_chars,
        )
        chunk_graphs: list[GraphAnnotation] = []
        usage: GraphUsage | None = None
        total = len(chunks)

        for index, blocks in enumerate(chunks, start=1):
            percent = 25 + round(((index - 1) / max(total, 1)) * 28)
            progress_callback(
                "chunk_extraction",
                percent,
                f"Extracting evidence section {index} of {total} with {self.model}",
            )
            chunk_document = document.model_copy(update={"blocks": blocks})
            response, headers = self._request_graph(
                chunk_document,
                chunk_index=index,
                chunk_count=total,
                progress_callback=progress_callback,
                progress_percent=percent,
            )
            usage = _merge_usage(usage, response.usage)
            chunk_graphs.append(_validate_chunk_graph(response.raw_graph_json))

            if index < total:
                self._wait_for_token_window_if_needed(
                    headers,
                    progress_callback=progress_callback,
                    progress_percent=percent,
                )

        progress_callback(
            "graph_merge",
            56,
            f"Merging {total} evidence sections and de-duplicating entities",
        )
        merged = _merge_chunk_graphs(chunk_graphs, document_sha256=document.source_sha256)
        return GraphExtractionResponse(
            raw_graph_json=merged.model_dump_json(),
            usage=usage,
        )

    def _request_graph(
        self,
        document: ParsedDocument,
        *,
        chunk_index: int,
        chunk_count: int,
        progress_callback: GraphProgressCallback,
        progress_percent: int,
    ) -> tuple[GraphExtractionResponse, httpx.Headers]:
        system_instruction = (
            _SYSTEM_INSTRUCTION
            + "\n\nExact allowed relation endpoints:\n"
            + _allowed_relation_endpoint_text()
            + "\n\nThis is one evidence section from a larger paper. Return exactly one "
            "paper node and no more than six central non-paper nodes or eight "
            "relations for this section. Reuse the paper title as its name. Do not "
            "guess facts that may appear in omitted sections."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": (
                        f"Evidence section {chunk_index} of {chunk_count}.\n"
                        + _build_prompt(document, max_chars=self._max_chunk_chars + self._context_chars)
                    ),
                },
            ],
            "temperature": 0,
            "max_completion_tokens": self._max_completion_tokens,
            "reasoning_effort": "low",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "evidence_graph_section",
                    "strict": True,
                    "schema": _groq_strict_graph_schema(),
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post(
                    GROQ_CHAT_COMPLETIONS_URL,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )
            except httpx.HTTPError as exc:
                raise GraphExtractionError("Groq API transport failed") from exc

            if response.status_code == 429 and attempt < self._max_attempts:
                wait_seconds = _retry_seconds(response.headers, fallback=60.0)
                progress_callback(
                    "rate_limit_wait",
                    progress_percent,
                    f"Groq free-tier token window is full; retrying in {wait_seconds:g}s",
                )
                self._sleeper(wait_seconds)
                continue
            if response.status_code >= 500 and attempt < self._max_attempts:
                wait_seconds = min(2.0**attempt, 8.0)
                progress_callback(
                    "rate_limit_wait",
                    progress_percent,
                    f"Groq is temporarily unavailable; retrying in {wait_seconds:g}s",
                )
                self._sleeper(wait_seconds)
                continue
            if response.status_code != 200:
                raise GraphExtractionError(_groq_api_error_message(response.status_code))

            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise GraphExtractionError("Groq returned an invalid response envelope") from exc
            if not isinstance(content, str) or not content.strip():
                raise GraphExtractionError("Groq returned no JSON content")
            return (
                GraphExtractionResponse(
                    raw_graph_json=content,
                    usage=_read_groq_usage(data),
                ),
                response.headers,
            )

        raise GraphExtractionError("Groq request attempts were exhausted")

    def _wait_for_token_window_if_needed(
        self,
        headers: httpx.Headers,
        *,
        progress_callback: GraphProgressCallback,
        progress_percent: int,
    ) -> None:
        remaining = _header_int(headers, "x-ratelimit-remaining-tokens")
        if remaining is None or remaining >= 4_000:
            return
        wait_seconds = _duration_seconds(headers.get("x-ratelimit-reset-tokens"))
        if wait_seconds is None or wait_seconds <= 0:
            return
        wait_seconds = min(wait_seconds, 65.0)
        progress_callback(
            "rate_limit_wait",
            progress_percent,
            f"Waiting {wait_seconds:g}s for the Groq free-tier token window",
        )
        self._sleeper(wait_seconds)


def _chunk_document_blocks(
    blocks: tuple[ParsedBlock, ...],
    *,
    max_chunk_chars: int,
    context_chars: int,
) -> tuple[tuple[ParsedBlock, ...], ...]:
    """Split on evidence-block and section boundaries, never inside normal blocks."""
    if not blocks:
        raise GraphExtractionError("Parsed paper has no evidence blocks")

    context: list[ParsedBlock] = []
    context_size = 0
    for block in blocks[:8]:
        rendered_size = len(_render_block(block))
        if context and context_size + rendered_size > context_chars:
            break
        context.append(block)
        context_size += rendered_size
        if context_size >= context_chars:
            break

    groups: list[tuple[ParsedBlock, ...]] = []
    current: list[ParsedBlock] = []
    current_size = 0
    heading_labels = {"section_header", "title", "subtitle"}
    for block in blocks:
        rendered_size = len(_render_block(block))
        starts_section = block.label.casefold() in heading_labels
        should_flush = bool(current) and (
            current_size + rendered_size > max_chunk_chars
            or (starts_section and current_size >= max_chunk_chars // 3)
        )
        if should_flush:
            groups.append(tuple(current))
            current = []
            current_size = 0
        current.append(block)
        current_size += rendered_size
    if current:
        groups.append(tuple(current))

    chunks: list[tuple[ParsedBlock, ...]] = []
    for group in groups:
        by_id = {block.block_id: block for block in (*context, *group)}
        ordered = tuple(
            block for block in blocks if block.block_id in by_id
        )
        chunks.append(ordered)
    return tuple(chunks)


def _validate_chunk_graph(
    raw_graph_json: str, *, provider: str = "Groq"
) -> GraphAnnotation:
    try:
        return GraphAnnotation.model_validate_json(raw_graph_json)
    except ValidationError as exc:
        if _is_endpoint_only_error(exc):
            graph, _omitted = _omit_invalid_endpoint_relations(raw_graph_json)
            return graph
        raise GraphExtractionError(
            f"{provider} returned invalid structured graph output for one evidence section"
        ) from exc


def _merge_chunk_graphs(
    graphs: list[GraphAnnotation], *, document_sha256: str
) -> GraphAnnotation:
    """Deterministically union section graphs without asking the model to guess again."""
    if not graphs:
        raise GraphExtractionError("Groq returned no section graphs to merge")

    nodes_by_key: dict[tuple[str, ...], GraphNode] = {}
    id_maps: list[dict[str, str]] = []
    for graph in graphs:
        id_map: dict[str, str] = {}
        for node in graph.nodes:
            key = _node_key(node)
            node_id = "paper" if node.node_type is NodeType.PAPER else _stable_id("node", key)
            id_map[node.node_id] = node_id
            existing = nodes_by_key.get(key)
            if existing is None:
                nodes_by_key[key] = node.model_copy(
                    update={"node_id": node_id, "document_sha256": document_sha256}
                )
            else:
                nodes_by_key[key] = existing.model_copy(
                    update={
                        "name": max((existing.name, node.name), key=len),
                        "evidence": _merge_evidence(existing.evidence, node.evidence),
                        "domain_type": existing.domain_type or node.domain_type,
                    }
                )
        id_maps.append(id_map)

    connectivity: Counter[str] = Counter()
    for graph, id_map in zip(graphs, id_maps, strict=True):
        for relation in graph.relations:
            connectivity[id_map[relation.source_node_id]] += 1
            connectivity[id_map[relation.target_node_id]] += 1
    ranked_nodes = sorted(
        nodes_by_key.values(),
        key=lambda node: (
            -connectivity[node.node_id], -len(node.evidence), _normalize_name(node.name),
        ),
    )
    # A readable overview must retain methods as well as findings. Taking every
    # claim first can exhaust the cap before any process/artifact survives.
    overview_types = (
        NodeType.PROCESS, NodeType.OBSERVATION, NodeType.CLAIM, NodeType.ARTIFACT,
        NodeType.CONCEPT, NodeType.CONTEXT, NodeType.ACTOR,
    )
    buckets = {
        kind: [node for node in ranked_nodes if node.node_type is kind]
        for kind in overview_types
    }
    ordered_nodes = [node for node in ranked_nodes if node.node_type is NodeType.PAPER]
    while len(ordered_nodes) < MAX_GRAPH_NODES and any(buckets.values()):
        for kind in overview_types:
            if buckets[kind] and len(ordered_nodes) < MAX_GRAPH_NODES:
                ordered_nodes.append(buckets[kind].pop(0))
    kept_ids = {node.node_id for node in ordered_nodes}

    relations_by_key: dict[tuple[str, str, str, str], GraphRelation] = {}
    for graph, id_map in zip(graphs, id_maps, strict=True):
        for relation in graph.relations:
            source_id = id_map[relation.source_node_id]
            target_id = id_map[relation.target_node_id]
            if source_id not in kept_ids or target_id not in kept_ids:
                continue
            key = (
                source_id,
                relation.relation_type.value,
                target_id,
                relation.domain_relation or "",
            )
            existing = relations_by_key.get(key)
            if existing is None:
                relations_by_key[key] = relation.model_copy(
                    update={
                        "relation_id": _stable_id("relation", key),
                        "source_node_id": source_id,
                        "target_node_id": target_id,
                    }
                )
            else:
                evidence = _merge_evidence(existing.evidence, relation.evidence)
                relations_by_key[key] = existing.model_copy(
                    update={
                        "evidence": evidence,
                        "status": (
                            RelationStatus.CANDIDATE
                            if evidence
                            else RelationStatus.UNCONFIRMED
                        ),
                    }
                )

    return GraphAnnotation(
        schema_version="3",
        nodes=tuple(ordered_nodes),
        relations=tuple(relations_by_key.values())[:MAX_GRAPH_RELATIONS],
    )


def _node_key(node: GraphNode) -> tuple[str, ...]:
    if node.node_type is NodeType.PAPER:
        return (NodeType.PAPER.value,)
    measurement = ""
    if node.node_type is NodeType.OBSERVATION:
        measurement = "|".join(
            str(value or "")
            for value in (node.value, node.unit, node.uncertainty, node.conditions)
        )
    return (node.node_type.value, _normalize_name(node.name), measurement)


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def _stable_id(prefix: str, parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _merge_evidence(
    left: tuple[EvidenceRef, ...], right: tuple[EvidenceRef, ...]
) -> tuple[EvidenceRef, ...]:
    merged: dict[tuple[str, str], EvidenceRef] = {}
    for ref in (*left, *right):
        merged[(ref.source_sha256, ref.block_id)] = ref
    return tuple(merged.values())


def _groq_strict_graph_schema() -> dict[str, Any]:
    schema = deepcopy(_provider_graph_schema())
    unsupported_constraints = {
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
    }

    def make_strict(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            for keyword in unsupported_constraints:
                value.pop(keyword, None)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["additionalProperties"] = False
                value["required"] = list(properties)
            for nested in value.values():
                make_strict(nested)
        elif isinstance(value, list):
            for nested in value:
                make_strict(nested)

    make_strict(schema)
    return schema


def _read_groq_usage(payload: dict[str, Any]) -> GraphUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = _nonnegative_int(usage.get("prompt_tokens"))
    output_tokens = _nonnegative_int(usage.get("completion_tokens"))
    total_tokens = _nonnegative_int(usage.get("total_tokens"))
    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = (
        _nonnegative_int(prompt_details.get("cached_tokens"))
        if isinstance(prompt_details, dict)
        else 0
    )
    completion_details = usage.get("completion_tokens_details")
    thinking_tokens = (
        _nonnegative_int(completion_details.get("reasoning_tokens"))
        if isinstance(completion_details, dict)
        else 0
    )
    return GraphUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        tool_tokens=0,
        total_tokens=total_tokens,
    )


def _merge_usage(left: GraphUsage | None, right: GraphUsage | None) -> GraphUsage | None:
    if left is None:
        return right
    if right is None:
        return left
    return GraphUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        thinking_tokens=left.thinking_tokens + right.thinking_tokens,
        tool_tokens=left.tool_tokens + right.tool_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _groq_api_error_message(status_code: int) -> str:
    prefix = f"Groq graph request failed (HTTP {status_code})"
    if status_code == 400:
        return f"{prefix}: the model rejected the structured request"
    if status_code == 401:
        return f"{prefix}: GROQ_API_KEY is missing, invalid, or revoked"
    if status_code == 403:
        return f"{prefix}: this Groq project cannot use the selected model"
    if status_code == 404:
        return f"{prefix}: the selected Groq model is unavailable"
    if status_code == 413:
        return f"{prefix}: this evidence section is too large"
    if status_code == 429:
        return f"{prefix}: free-tier token or request quota is exhausted; retry after reset"
    if status_code >= 500:
        return f"{prefix}: Groq is temporarily unavailable after bounded retries"
    return prefix


def _header_int(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _retry_seconds(headers: httpx.Headers, *, fallback: float) -> float:
    value = headers.get("retry-after")
    try:
        return min(max(float(value), 0.0), 65.0) if value is not None else fallback
    except ValueError:
        return fallback


def _duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*(?:(\d+(?:\.\d+)?)m)?(\d+(?:\.\d+)?)s\s*", value)
    if match is None:
        return None
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0
