"""Run repeatable, non-persistent Ollama graph-section evaluations.

Example:
    PYTHONPATH=src uv run python scripts/eval_ollama_graph.py \
      --parsed data/evidencegraph/eval_inputs/<sha>/parsed.json \
      --model gpt-oss:20b --think low --chunks 1 3 14 19 --repeats 2 \
      --output-dir data/evidencegraph/eval_runs/gpt-oss-20b-low

The output contains source-backed graph candidates. Keep it in a private,
Git-ignored directory. Model reasoning text is never recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from evidencegraph.extraction import GraphExtractionError, GraphExtractionResponse
from evidencegraph.graph_contract import GraphAnnotation
from evidencegraph.models import ParsedBlock, ParsedDocument
from evidencegraph.ollama_extractor import (
    OllamaGraphExtractor,
    _safe_chunk_issue,
    _validate_ollama_chunk_graph,
)


class GraphRequester(Protocol):
    model: str
    version: str

    def _request_graph(
        self,
        document: ParsedDocument,
        *,
        index: int,
        total: int,
        repair_candidate: str | None = None,
        validation_error: str | None = None,
    ) -> GraphExtractionResponse: ...


class ThinkingClient:
    """Change only Ollama's reasoning setting; retain the production prompt."""

    def __init__(self, client: httpx.Client, *, think: bool | str, timeout: float) -> None:
        self.client = client
        self.think = think
        self.timeout = timeout

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        payload = dict(kwargs["json"])
        payload["think"] = self.think
        kwargs["json"] = payload
        kwargs["timeout"] = self.timeout
        return self.client.post(url, **kwargs)


def _validation_locations(error: ValidationError) -> list[str]:
    return [
        f"{'.'.join(map(str, item['loc'])) or 'graph'} [{item['type']}]"
        for item in error.errors(include_input=False, include_url=False)[:5]
    ]


def evaluate_case(
    document: ParsedDocument,
    blocks: tuple[ParsedBlock, ...],
    extractor: GraphRequester,
    *,
    index: int,
    total: int,
    repeat: int,
    parsed_sha256: str,
    think: bool | str,
    case_dir: Path,
) -> dict[str, Any]:
    """Save one candidate and its validation result without changing PaperStore."""
    case_dir.mkdir(parents=True, exist_ok=False)
    section = document.model_copy(update={"blocks": blocks})
    result: dict[str, Any] = {
        "document_sha256": document.source_sha256,
        "parsed_sha256": parsed_sha256,
        "parser_version": document.parser_version,
        "extractor_version": extractor.version,
        "model": extractor.model,
        "think": think,
        "chunk_index": index,
        "chunk_count": total,
        "repeat": repeat,
        "block_ids": [block.block_id for block in blocks],
        "strict_first_valid": False,
        "repair_requested": False,
        "final_valid": False,
    }
    start = time.perf_counter()
    try:
        first = extractor._request_graph(section, index=index, total=total)
        (case_dir / "first.json").write_text(first.raw_graph_json, encoding="utf-8")
        result["first_usage"] = first.usage.model_dump() if first.usage else None
        try:
            GraphAnnotation.model_validate_json(first.raw_graph_json)
            result["strict_first_valid"] = True
        except ValidationError as error:
            result["first_validation_locations"] = _validation_locations(error)

        try:
            graph = _validate_ollama_chunk_graph(first.raw_graph_json)
        except GraphExtractionError as error:
            result["repair_requested"] = True
            repaired = extractor._request_graph(
                section,
                index=index,
                total=total,
                repair_candidate=first.raw_graph_json,
                validation_error=_safe_chunk_issue(first.raw_graph_json, error),
            )
            (case_dir / "repair.json").write_text(repaired.raw_graph_json, encoding="utf-8")
            result["repair_usage"] = repaired.usage.model_dump() if repaired.usage else None
            graph = _validate_ollama_chunk_graph(repaired.raw_graph_json)

        graph.validate_against((document,))
        result["final_valid"] = True
        result["node_count"] = len(graph.nodes)
        result["relation_count"] = len(graph.relations)
        result["relation_types"] = dict(
            Counter(relation.relation_type.value for relation in graph.relations)
        )
        (case_dir / "validated_graph.json").write_text(
            graph.model_dump_json(indent=2), encoding="utf-8"
        )
        node_names = {node.node_id: node.name for node in graph.nodes}
        block_text = {block.block_id: block.text for block in document.blocks}
        review = [
            {
                "source": node_names[relation.source_node_id],
                "type": relation.relation_type.value,
                "target": node_names[relation.target_node_id],
                "rationale": relation.rationale,
                "evidence": [
                    {"block_id": ref.block_id, "excerpt": block_text[ref.block_id][:320]}
                    for ref in relation.evidence
                ],
            }
            for relation in graph.relations
        ]
        (case_dir / "review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (GraphExtractionError, ValidationError, ValueError, httpx.RequestError) as error:
        result["error_type"] = type(error).__name__
        cause = error.__cause__
        if isinstance(error, httpx.TimeoutException) or isinstance(cause, httpx.TimeoutException):
            result["error_category"] = "transport_timeout"
        elif isinstance(error, httpx.RequestError) or isinstance(cause, httpx.RequestError):
            result["error_category"] = "transport_error"
        elif isinstance(error, ValueError):
            result["error_category"] = "unknown_evidence"
        else:
            result["error_category"] = "invalid_model_output"
        repair_path = case_dir / "repair.json"
        if repair_path.is_file():
            try:
                GraphAnnotation.model_validate_json(repair_path.read_bytes())
            except ValidationError as repair_error:
                result["repair_validation_locations"] = _validation_locations(repair_error)
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        (case_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--think", choices=("false", "low", "medium", "high"), required=True)
    parser.add_argument("--chunks", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")

    parsed_bytes = args.parsed.read_bytes()
    document = ParsedDocument.model_validate_json(parsed_bytes)
    if len(set(args.chunks)) != len(args.chunks):
        parser.error("--chunks must not contain duplicates")
    parsed_sha256 = hashlib.sha256(parsed_bytes).hexdigest()
    think: bool | str = False if args.think == "false" else args.think
    args.output_dir.mkdir(parents=True, exist_ok=False)
    failed = False
    with httpx.Client() as client:
        extractor = OllamaGraphExtractor(
            model=args.model,
            base_url=args.base_url,
            client=ThinkingClient(client, think=think, timeout=args.timeout),
        )
        chunks = extractor.chunk_document(document)
        if any(index < 1 or index > len(chunks) for index in args.chunks):
            parser.error(f"--chunks must be between 1 and {len(chunks)}")
        for index in args.chunks:
            for repeat in range(1, args.repeats + 1):
                print(f"Testing {extractor.model} chunk {index}/{len(chunks)} repeat {repeat}", flush=True)
                result = evaluate_case(
                    document,
                    chunks[index - 1],
                    extractor,
                    index=index,
                    total=len(chunks),
                    repeat=repeat,
                    parsed_sha256=parsed_sha256,
                    think=think,
                    case_dir=args.output_dir / f"chunk-{index:02d}-repeat-{repeat:02d}",
                )
                failed |= not result["final_valid"]
                print(
                    json.dumps(
                        {key: result.get(key) for key in (
                            "chunk_index", "repeat", "strict_first_valid", "repair_requested",
                            "final_valid", "node_count", "relation_count", "error_category",
                            "elapsed_seconds",
                        )},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
