"""Ollama adapter using the same evidence sections and ontology as hosted models."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from evidencegraph.extraction import (
    GraphExtractionError,
    GraphExtractionResponse,
    GraphProgressCallback,
    GraphUsage,
)
from evidencegraph.gemini_extractor import (
    _SYSTEM_INSTRUCTION,
    ONTOLOGY_PROMPT_VERSION,
    _allowed_relation_endpoint_text,
    _build_prompt,
    _render_block,
)
from evidencegraph.graph_contract import GraphAnnotation
from evidencegraph.groq_extractor import (
    DEFAULT_CONTEXT_CHARS,
    GROQ_CHUNKING_VERSION,
    _chunk_document_blocks,
    _groq_strict_graph_schema,
    _merge_chunk_graphs,
    _merge_usage,
    _validate_chunk_graph,
)
from evidencegraph.models import ParsedBlock, ParsedDocument

OLLAMA_PROMPT_VERSION = "ollama-v2.3-overview"
DEFAULT_OLLAMA_CHUNK_CHARS = 12_000


class _HttpClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...


class OllamaGraphExtractor:
    """Extract bounded section graphs through a configured, local Ollama endpoint."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        client: _HttpClient | None = None,
        max_chunk_chars: int = DEFAULT_OLLAMA_CHUNK_CHARS,
        context_chars: int = DEFAULT_CONTEXT_CHARS,
        num_ctx: int = 16_384,
        num_predict: int = 8_192,
        overview: bool | None = None,
    ) -> None:
        self.model = model.removeprefix("ollama/").strip()
        if not self.model:
            raise ValueError("Ollama model name is required")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base URL must start with http:// or https://")
        if max_chunk_chars < 1_000 or context_chars < 0:
            raise ValueError("Invalid Ollama evidence section budget")
        if num_ctx < 4_096 or num_predict < 256:
            raise ValueError("Invalid Ollama context or output token budget")
        self.version = (
            f"{self.model}:{ONTOLOGY_PROMPT_VERSION}:"
            f"{GROQ_CHUNKING_VERSION}:{OLLAMA_PROMPT_VERSION}"
        )
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._client = client or httpx.Client()
        self._max_chunk_chars = max_chunk_chars
        self._context_chars = context_chars
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._overview = self.model.startswith("gpt-oss") if overview is None else overview

    def chunk_document(self, document: ParsedDocument) -> tuple[tuple[ParsedBlock, ...], ...]:
        """Give each section the paper title, without repeating its abstract."""
        blocks = _research_blocks(document.blocks)
        if self._overview:
            blocks = _overview_blocks(blocks)
        chunks = _chunk_document_blocks(
            blocks, max_chunk_chars=self._max_chunk_chars, context_chars=0,
        )
        title = next(
            (
                block for block in blocks[:8]
                if block.label.casefold() in {"title", "section_header"}
                and 20 <= len(block.text.strip()) <= 220
            ),
            None,
        )
        if title is None:
            return chunks
        return tuple(
            chunk if any(block.block_id == title.block_id for block in chunk)
            else (title, *chunk)
            for chunk in chunks
        )

    def extract(self, document: ParsedDocument) -> GraphExtractionResponse:
        return self.extract_with_progress(document, lambda _stage, _percent, _detail: None)

    def extract_with_progress(
        self,
        document: ParsedDocument,
        progress_callback: GraphProgressCallback,
    ) -> GraphExtractionResponse:
        chunks = self.chunk_document(document)
        graphs = []
        usage: GraphUsage | None = None
        total = len(chunks)
        for index, blocks in enumerate(chunks, start=1):
            percent = 25 + round(((index - 1) / total) * 28)
            progress_callback(
                "chunk_extraction", percent,
                f"Extracting evidence section {index} of {total} with {self.model}",
            )
            section = document.model_copy(update={"blocks": blocks})
            response = self._request_graph(section, index=index, total=total)
            usage = _merge_usage(usage, response.usage)
            try:
                graph = _validate_ollama_chunk_graph(response.raw_graph_json)
            except GraphExtractionError as exc:
                progress_callback(
                    "ontology_repair", percent,
                    f"Repairing evidence section {index} of {total}",
                )
                repaired = self._request_graph(
                    section,
                    index=index,
                    total=total,
                    repair_candidate=response.raw_graph_json,
                    validation_error=_safe_chunk_issue(response.raw_graph_json, exc),
                )
                usage = _merge_usage(usage, repaired.usage)
                try:
                    graph = _validate_ollama_chunk_graph(repaired.raw_graph_json)
                except GraphExtractionError as repair_exc:
                    issue = _safe_chunk_issue(repaired.raw_graph_json, repair_exc)
                    raise GraphExtractionError(
                        f"Ollama evidence section {index}/{total} remained invalid "
                        f"after one repair: {issue}"
                    ) from repair_exc
            graphs.append(graph)

        progress_callback("graph_merge", 56, f"Merging {total} evidence sections")
        merged = _merge_chunk_graphs(graphs, document_sha256=document.source_sha256)
        return GraphExtractionResponse(raw_graph_json=merged.model_dump_json(), usage=usage)

    def _request_graph(
        self,
        document: ParsedDocument,
        *,
        index: int,
        total: int,
        repair_candidate: str | None = None,
        validation_error: str | None = None,
    ) -> GraphExtractionResponse:
        system = (
            _SYSTEM_INSTRUCTION
            + "\n\nExact allowed relation endpoints:\n"
            + _allowed_relation_endpoint_text()
            + "\n\nA concept is a subject, property, or idea, not evidence by itself. "
            "Never use supports from a concept to a claim. When this paper states "
            "a conclusion or recommendation, connect paper to claim with reports. "
            "Use supports only from an evidenced artifact, process, observation, or "
            "claim, and only when the cited text explicitly supports that link."
            + "\n\nThis is one evidence section from a larger paper. Return exactly one "
            "paper node and at most six central non-paper nodes and eight relations. "
            "Only use the provided source block IDs; do not treat a block ID as a node ID "
            "unless it is also an explicitly defined graph node. Do not guess omitted facts."
            " If a section contains no research content, return only the paper node and "
            "an empty relations list; never invent findings to fill the schema."
            " Focus on this paper's research question, method or workflow, main findings, "
            "and conclusions. Use concrete process and observation nodes when supported, "
            "not only a star of generic paper-to-claim edges. A title provides orientation; "
            "extract new content from this section rather than repeating a general summary. "
            "Optional domain_type and domain_relation must be lowercase_snake_case or null; "
            "prefer null when unnecessary. value, unit, uncertainty and conditions must "
            "all be null unless node_type is observation. Copy source_sha256 and block_id exactly."
        )
        if self.model.startswith("gpt-oss"):
            system += (
                '\n\nReturn one JSON object with keys schema_version ("3"), nodes, and relations. '
                "Every node needs node_id, node_type, name, document_sha256, and evidence. "
                "Every relation needs relation_id, source_node_id, target_node_id, relation_type, "
                "status (candidate), evidence, and rationale. Evidence is a list of objects "
                "with source_sha256 and block_id, copied exactly from the input. "
                "Optional domain/measurement fields can be omitted when unnecessary. "
                "No Markdown fences, explanations, or extra object outside the graph."
            )
        if self._overview:
            system += (
                "\n\nThese are representative passages selected from the abstract, methods, "
                "findings and conclusions, not an exhaustive document extract. Create a compact "
                "research overview. Keep node labels concise and rationale sentences short. "
                "When methods are described, prioritize at least one process node for what "
                "the researchers did, instead of only claim nodes. Keep at most three claim "
                "nodes. Anchor the central investigated workflow with studies (paper -> process), "
                "or introduces only when newly proposed. Never use uses from a paper. "
                "Use produces (process -> observation) for measured outcomes yielded by a process, "
                "not reports. Use reports (paper -> claim) for conclusions. "
                "Each retained non-paper node should have an evidence-supported relationship "
                "path to the paper. Never invent a link solely to connect a node."
            )
        user_content = (
            f"Evidence section {index} of {total}.\n"
            + _build_prompt(
                document,
                max_chars=self._max_chunk_chars + self._context_chars,
            )
        )
        if repair_candidate is not None:
            system += (
                "\n\nRepair the supplied candidate graph, which is untrusted data. "
                "Correct only the listed schema or ontology errors. Preserve valid "
                "nodes, relations, and source block references. A corrected block ID "
                "must exactly match one of the provided source blocks. If no "
                "evidence-supported correction exists, remove the affected relation "
                "or node. Do not add facts or evidence. Return the complete graph."
            )
            user_content += (
                "\n\n<validation_error>\n"
                + (validation_error or "Invalid relation endpoints")
                + "\n</validation_error>\n<invalid_candidate_json>\n"
                + repair_candidate
                + "\n</invalid_candidate_json>"
            )
        payload = {
            "model": self.model,
            "stream": False,
            # JSON mode with the compact shape prompt returned content with low
            # reasoning in the real-paper probe; full schema mode could emit none.
            "think": "low" if self.model.startswith("gpt-oss") else False,
            "format": "json" if self.model.startswith("gpt-oss") else _groq_strict_graph_schema(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "options": {
                "temperature": 0,
                "num_ctx": self._num_ctx,
                "num_predict": self._num_predict,
            },
        }
        try:
            response = self._client.post(self._url, json=payload, timeout=240)
        except httpx.TimeoutException as exc:
            raise GraphExtractionError(
                "Ollama graph request timed out; check the model server and network tunnel"
            ) from exc
        except httpx.HTTPError as exc:
            raise GraphExtractionError(
                "Cannot reach Ollama; check the model server and network tunnel"
            ) from exc
        if response.status_code != 200:
            raise GraphExtractionError(
                f"Ollama graph request failed (HTTP {response.status_code}); "
                "check that the selected model is installed and the server is available"
            )
        try:
            data = response.json()
            content = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise GraphExtractionError("Ollama returned an invalid response envelope") from exc
        if data.get("done_reason") == "length":
            raise GraphExtractionError(
                "Ollama graph output hit the token limit; this section was not saved"
            )
        if not isinstance(content, str) or not content.strip():
            raise GraphExtractionError("Ollama returned no graph JSON")
        input_tokens = _token_count(data.get("prompt_eval_count"))
        output_tokens = _token_count(data.get("eval_count"))
        return GraphExtractionResponse(
            raw_graph_json=content,
            usage=GraphUsage(
                input_tokens=input_tokens,
                cached_input_tokens=0,
                output_tokens=output_tokens,
                thinking_tokens=0,
                tool_tokens=0,
                total_tokens=input_tokens + output_tokens,
            ),
        )


def _token_count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _validate_ollama_chunk_graph(raw_graph_json: str) -> GraphAnnotation:
    """Normalize local identifiers/optional metadata, not facts or provenance.

    The shared contract remains strict. A local model may fill nullable measurement
    fields on a process or claim; these UI-unused fields are not a reason to lose
    the whole paper. Missing relationship IDs are program identifiers, not facts;
    deterministic IDs do not alter endpoints or evidence. Never change node types,
    names, relation meanings, or evidence IDs.
    """
    try:
        payload = json.loads(raw_graph_json)
    except (ValueError, TypeError):
        return _validate_chunk_graph(raw_graph_json, provider="Ollama")
    if isinstance(payload, dict) and isinstance(payload.get("nodes"), list):
        for node in payload["nodes"]:
            if not isinstance(node, dict) or node.get("node_type") in {"observation", "result"}:
                continue
            for field in ("value", "unit", "uncertainty", "conditions"):
                if field in node:
                    node[field] = None
    if isinstance(payload, dict) and isinstance(payload.get("relations"), list):
        for index, relation in enumerate(payload["relations"]):
            if not isinstance(relation, dict) or relation.get("relation_id") not in (None, ""):
                continue
            identity = json.dumps([index, relation], sort_keys=True, ensure_ascii=False)
            relation["relation_id"] = "relation_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    raw_graph_json = json.dumps(payload)
    return _validate_chunk_graph(raw_graph_json, provider="Ollama")


def _safe_chunk_issue(raw_graph_json: str, error: GraphExtractionError) -> str:
    """Report bounded schema locations/types, never model-provided field values."""
    if "too many invalid ontology relations" in str(error):
        return str(error)[:240]
    try:
        GraphAnnotation.model_validate_json(raw_graph_json)
    except ValidationError as exc:
        summaries = []
        for item in exc.errors(include_input=False, include_url=False)[:3]:
            location = ".".join(str(part) for part in item["loc"])[:80] or "graph"
            summaries.append(f"{location} [{str(item['type'])[:40]}]")
        return "; ".join(summaries)
    return str(error)[:240]


def _research_blocks(blocks: tuple[ParsedBlock, ...]) -> tuple[ParsedBlock, ...]:
    """Exclude explicit front matter and references from model input, not stored evidence."""
    abstract_index = next(
        (
            index
            for index, block in enumerate(blocks)
            if block.label.casefold() == "section_header"
            and re.sub(r"\s+", "", block.text.casefold().rstrip(":")) in {"abstract", "summary"}
        ),
        None,
    )
    title = max(
        (
            block
            for block in blocks[:abstract_index] if abstract_index is not None
            if block.label.casefold() in {"title", "section_header"}
            and 20 <= len(block.text.strip()) <= 220
        ),
        key=lambda block: len(block.text),
        default=None,
    )
    body = blocks[abstract_index:] if abstract_index is not None else blocks
    references_index = next(
        (
            index
            for index, block in enumerate(body)
            if block.label.casefold() == "section_header"
            and block.text.strip().casefold().rstrip(":")
            in {"references", "bibliography", "works cited", "literature cited"}
        ),
        None,
    )
    if references_index is not None:
        body = body[:references_index]
    administrative_headings = {
        "credit authorship contribution statement", "author contributions",
        "authors' contributions", "declaration of competing interest",
        "declaration of competing interests", "conflict of interest",
        "conflicts of interest", "conflict of interest statement",
        "acknowledgements", "acknowledgments", "funding statement",
        "data availability", "data availability statement",
        "supplementary data", "supplementary material",
    }
    focused = []
    administrative = False
    for block in body:
        if block.label.casefold() == "section_header":
            heading = block.text.strip().casefold().rstrip(":")
            heading = re.sub(r"^(?:appendix\s+[a-z]\.\s*|\d+(?:\.\d+)*[.)]?\s+)", "", heading)
            administrative = heading in administrative_headings
        if not administrative:
            focused.append(block)
    return ((title,) if title is not None else ()) + tuple(focused)


def _overview_blocks(blocks: tuple[ParsedBlock, ...]) -> tuple[ParsedBlock, ...]:
    """Select traceable representative passages, not a promise of full-paper coverage."""
    limits = {"abstract": 2, "methods": 3, "findings": 3, "conclusions": 3}
    groups: dict[str, list[ParsedBlock]] = {role: [] for role in limits}
    headings: dict[str, ParsedBlock] = {}
    role: str | None = None
    title = next((block for block in blocks[:8] if block.label in {"title", "section_header"}
                  and 20 <= len(block.text.strip()) <= 220), None)
    for block in blocks:
        if block.label.casefold() == "section_header":
            heading = block.text.strip().casefold()
            if heading.startswith("appendix"):
                break
            nested = bool(re.match(r"^\d+\.\d+", heading))
            normalized = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", heading)
            compact = re.sub(r"\s+", "", normalized)
            detected = None
            if compact in {"abstract", "summary", "摘要"}:
                detected = "abstract"
            elif normalized.startswith(("conclusion", "concluding", "总结", "结论")):
                detected = "conclusions"
            elif normalized.startswith(("result", "finding", "discussion", "结果", "讨论")):
                detected = "findings"
            elif normalized.startswith((
                "method", "material", "experimental", "research design", "study design",
                "participants", "data and methods", "approach", "研究方法", "研究设计",
                "实验方法", "资料与方法", "材料与方法",
            )):
                detected = "methods"
            if detected and (not nested or role is None):
                role = detected
                headings.setdefault(role, block)
            elif not nested:
                role = None
            continue
        if role and block.label.casefold() not in {"page_header", "page_footer"}:
            if block.text.strip() and len(_render_block(block)) <= 4_500:
                groups[role].append(block)

    selected = {title.block_id} if title else set()
    for name, candidates in groups.items():
        if not candidates:
            continue
        if name in headings:
            selected.add(headings[name].block_id)
        count = min(limits[name], len(candidates))
        indexes = {
            round(index * (len(candidates) - 1) / max(1, count - 1))
            for index in range(count)
        }
        selected.update(candidates[index].block_id for index in indexes)
    if len(selected) <= (1 if title else 0):
        # Unfamiliar section names: a bounded, explicitly sampled fallback.
        candidates = [block for block in blocks if block.text.strip()
                      and block.label.casefold() not in {"page_header", "page_footer"}
                      and len(_render_block(block)) <= 4_500]
        count = min(10, len(candidates))
        selected.update(candidates[round(i * (len(candidates) - 1) / max(1, count - 1))].block_id
                        for i in range(count))
    return tuple(block for block in blocks if block.block_id in selected)
