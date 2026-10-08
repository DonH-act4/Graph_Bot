"""Prepare a real paper/graph/conversation through the normal local API.

The run manifest is resumable. Publishing its conversation as the read-only
showcase is a separate local ConversationStore.publish_showcase call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx

QUESTIONS = (
    "这篇论文解决什么问题？用三点概括核心方法和贡献。",
    "迁移学习在40个场景测试中带来了什么提升？请给出分类准确率和漏点定位误差的前后数值。",
    "Integrated Gradients 在这项研究中起什么作用？它说明模型关注了哪些压力信号特征？",
)


def wait_ready(client: httpx.Client, path: str, *, max_wait: float) -> dict[str, Any]:
    """Wait for a persisted task, emitting only changed progress milestones."""
    deadline = time.monotonic() + max_wait
    previous = None
    while time.monotonic() < deadline:
        response = client.get(path)
        response.raise_for_status()
        record = response.json()
        milestone = (record.get("state"), record.get("stage"), record.get("progress_detail"))
        if milestone != previous:
            print(json.dumps({"path": path, "progress": milestone}, ensure_ascii=False), flush=True)
            previous = milestone
        if record["state"] == "ready":
            return record
        if record["state"] == "failed":
            raise RuntimeError(record.get("error") or "Paper processing failed")
        time.sleep(2)
    raise TimeoutError(f"Task did not finish within {max_wait:g} seconds: {path}")


def ready_graph_matches(client: httpx.Client, document_id: str, model: str) -> bool:
    """Reuse a successful local-model graph; a not-yet-created graph is not an error."""
    status = client.get(f"/papers/{document_id}/graph/status")
    if status.status_code == 404:
        return False
    status.raise_for_status()
    if status.json()["state"] != "ready":
        return False
    current = client.get(f"/papers/{document_id}/graph")
    current.raise_for_status()
    return current.json()["extractor_version"].startswith(model.removeprefix("ollama/") + ":")


def request_graph_generation(
    client: httpx.Client, document_id: str, model: str, *, regenerate: bool,
) -> None:
    """First-build is idempotent; an existing artifact needs the explicit rebuild route."""
    path = f"/papers/{document_id}/graph"
    if regenerate:
        current = client.get(path)
        if current.status_code == 200:
            path += "/rebuild"
        elif current.status_code != 404:
            current.raise_for_status()
    response = client.post(path, json={"model": model})
    response.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="ollama/gpt-oss:20b")
    parser.add_argument("--regenerate", action="store_true", help="Explicitly rebuild a ready graph")
    parser.add_argument("--max-wait", type=float, default=1_200)
    args = parser.parse_args()
    pdf_bytes = args.pdf.read_bytes()
    document_id = hashlib.sha256(pdf_bytes).hexdigest()
    user_id = str(uuid5(NAMESPACE_URL, "evidencegraph-local-showcase"))
    thread_id = str(uuid5(NAMESPACE_URL, "evidencegraph-showcase:" + document_id))
    state: dict[str, Any] = {
        "document_id": document_id,
        "user_id": user_id,
        "thread_id": thread_id,
        "title": args.title,
        "model": args.model,
        "questions": [],
    }
    if args.output.exists():
        state = json.loads(args.output.read_text(encoding="utf-8"))
        if state["document_id"] != document_id:
            parser.error("Output manifest belongs to another PDF")
        if state["model"] != args.model:
            parser.error("Output manifest belongs to another model; choose a new output path")
    for obsolete in ("error_type", "http_status"):
        state.pop(obsolete, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=300) as client:
            configuration = client.get("/papers/configuration")
            configuration.raise_for_status()
            if len(pdf_bytes) > configuration.json()["max_pdf_bytes"]:
                raise ValueError("PDF exceeds the configured upload limit")
            uploaded = client.post(
                "/papers", content=pdf_bytes, headers={"Content-Type": "application/pdf"},
            )
            uploaded.raise_for_status()
            paper = wait_ready(client, f"/papers/{document_id}", max_wait=args.max_wait)
            state["paper"] = paper
            workspace = client.put(
                f"/conversations/{thread_id}",
                json={
                    "user_id": user_id,
                    "title": args.title,
                    "document_id": document_id,
                    "document_name": args.title,
                    "selected_block_ids": [],
                    "draft_message": "",
                },
            )
            workspace.raise_for_status()
            reusable = not args.regenerate and ready_graph_matches(client, document_id, args.model)
            if not reusable:
                request_graph_generation(
                    client, document_id, args.model, regenerate=args.regenerate,
                )
            state["graph_status"] = wait_ready(
                client, f"/papers/{document_id}/graph/status", max_wait=args.max_wait,
            )
            graph = client.get(f"/papers/{document_id}/graph")
            graph.raise_for_status()
            payload = graph.json()
            state["graph_summary"] = {
                "nodes": len(payload["graph"]["nodes"]),
                "relations": len(payload["graph"]["relations"]),
                "extractor_version": payload["extractor_version"],
            }
            print(json.dumps(state["graph_summary"], ensure_ascii=False), flush=True)
            for question in QUESTIONS[len(state["questions"]):]:
                print("Asking: " + question, flush=True)
                response = client.post(
                    "/research-assistant/invoke",
                    json={
                        "message": question,
                        "model": "ollama",
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "document_id": document_id,
                    },
                )
                response.raise_for_status()
                state["questions"].append({"question": question, "answer": response.json()})
                args.output.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                print(response.json()["content"], flush=True)
            state["complete"] = True
    except (httpx.HTTPError, RuntimeError, TimeoutError, ValueError) as error:
        state["complete"] = False
        state["error_type"] = type(error).__name__
        if isinstance(error, httpx.HTTPStatusError):
            state["http_status"] = error.response.status_code
            print(f"API request failed: HTTP {error.response.status_code}", flush=True)
        else:
            print(str(error), flush=True)
    finally:
        state["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": state.get("complete"), "manifest": str(args.output)}), flush=True)
    return 0 if state.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
