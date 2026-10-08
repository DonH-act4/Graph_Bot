"""Tests for the file-backed worker boundary."""

import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from evidencegraph.models import BoundingBox, ParsedBlock, ParsedDocument, SourceLocation
from evidencegraph.paper_worker import PaperWorker
from evidencegraph.papers import GraphStage, GraphState, PaperStage, PaperState, PaperStore

PDF = b"%PDF-1.4\nworker fixture"
PDF_ID = hashlib.sha256(PDF).hexdigest()
BLOCK_ID = "blk_" + "1" * 24


@dataclass(frozen=True)
class FakeExtractor:
    output: str
    name: str = "fake-worker-extractor"
    version: str = "1"

    def extract(self, _document: ParsedDocument) -> str:
        return self.output


def graph_json() -> str:
    return f"""
    {{
      "schema_version": "1",
      "nodes": [{{
        "node_id": "paper",
        "node_type": "paper",
        "name": "Worker paper",
        "document_sha256": "{PDF_ID}",
        "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}]
      }}],
      "relations": []
    }}
    """


def recoverable_graph_json() -> str:
    valid_relations = ",".join(
        f'''{{
          "relation_id": "reports-{index}",
          "source_node_id": "paper",
          "target_node_id": "claim",
          "relation_type": "reports",
          "status": "candidate",
          "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}],
          "rationale": "The paper reports the claim."
        }}'''
        for index in range(3)
    )
    return f'''{{
      "schema_version": "2",
      "nodes": [
        {{
          "node_id": "paper", "node_type": "paper", "name": "Worker paper",
          "document_sha256": "{PDF_ID}",
          "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}]
        }},
        {{
          "node_id": "claim", "node_type": "claim", "name": "Main claim",
          "document_sha256": "{PDF_ID}",
          "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}]
        }},
        {{
          "node_id": "actor", "node_type": "actor", "name": "Population",
          "document_sha256": "{PDF_ID}",
          "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}]
        }}
      ],
      "relations": [
        {valid_relations},
        {{
          "relation_id": "bad-introduces",
          "source_node_id": "paper",
          "target_node_id": "actor",
          "relation_type": "introduces",
          "status": "candidate",
          "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}],
          "rationale": "The paper names the population."
        }}
      ]
    }}'''


def parsed_document(path: Path) -> ParsedDocument:
    return ParsedDocument(
        source_filename=path.name,
        source_sha256=PDF_ID,
        parser_version="test",
        page_count=1,
        blocks=(
            ParsedBlock(
                block_id=BLOCK_ID,
                source_ref="#/texts/0",
                label="text",
                text="Worker evidence",
                locations=(
                    SourceLocation(
                        page_number=1,
                        bounding_box=BoundingBox(
                            left=1, top=2, right=3, bottom=4, coordinate_origin="bottom-left"
                        ),
                        character_start=0,
                        character_end=15,
                    ),
                ),
            ),
        ),
    )


def test_worker_processes_one_queued_document(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    worker = PaperWorker(store)
    assert worker.run_once()
    record = store.get(PDF_ID)
    assert record is not None and record.state == PaperState.READY
    assert record.stage is PaperStage.COMPLETE
    assert record.progress_percent == 100
    assert not worker.run_once()


def test_deletion_stops_isolated_job_before_removing_pdf(tmp_path: Path, monkeypatch):
    if os.name != "posix":
        return
    store = PaperStore(tmp_path, parser=parsed_document)
    store.accept(PDF)
    worker = PaperWorker(store, isolate_jobs=True)
    original_popen = subprocess.Popen

    def sleeping_child(_args, **kwargs):
        return original_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
        )

    monkeypatch.setattr(subprocess, "Popen", sleeping_child)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(worker.run_once)
        lock = store.source_path(PDF_ID).parent / "worker.lock"
        deadline = time.monotonic() + 5
        while not lock.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert lock.exists()
        store.request_deletion(PDF_ID)
        store.finish_deletion(PDF_ID, timeout=5)
        assert task.result(timeout=5)
    assert not store.source_path(PDF_ID).parent.exists()
    assert store.deletion_state(PDF_ID) == "deleted"


def test_worker_marks_stale_lock_as_interrupted(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    lock_path = store.source_path(PDF_ID).parent / "worker.lock"
    lock_path.touch()
    assert PaperWorker(store).run_once()
    record = store.get(PDF_ID)
    assert record is not None and record.state == PaperState.FAILED
    assert "interrupted" in (record.error or "")
    assert not lock_path.exists()


def test_worker_processes_queued_graph_after_pdf_is_ready(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    worker = PaperWorker(store, graph_extractor=FakeExtractor(graph_json()))
    assert worker.run_once()
    graph_record, needed = store.request_graph(PDF_ID)
    assert needed and graph_record.state is GraphState.QUEUED

    assert worker.run_once()

    graph_record = store.get_graph_record(PDF_ID)
    assert graph_record is not None and graph_record.state is GraphState.READY
    assert graph_record.node_count == 1
    assert graph_record.relation_count == 0
    assert graph_record.extractor_name == "fake-worker-extractor"
    assert graph_record.current_version == 1
    assert graph_record.stage is GraphStage.COMPLETE
    assert graph_record.progress_percent == 100
    assert store.get_graph(PDF_ID) is not None


def test_worker_uses_model_persisted_with_graph_job(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    assert PaperWorker(store).run_once()
    queued, needed = store.request_graph(PDF_ID, model="chosen-model")
    assert needed and queued.requested_model == "chosen-model"
    selected: list[str] = []

    def factory(model: str) -> FakeExtractor:
        selected.append(model)
        return FakeExtractor(graph_json())

    assert PaperWorker(store, graph_extractor_factory=factory).run_once()
    assert selected == ["chosen-model"]
    ready = store.get_graph_record(PDF_ID)
    assert ready is not None and ready.requested_model == "chosen-model"


def test_worker_persists_recovered_graph_warnings(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    PaperWorker(store).run_once()
    store.request_graph(PDF_ID)

    assert PaperWorker(
        store, graph_extractor=FakeExtractor(recoverable_graph_json())
    ).run_once()

    ready = store.get_graph_record(PDF_ID)
    graph = store.get_graph(PDF_ID)
    assert ready is not None and ready.state is GraphState.READY
    assert ready.relation_count == 3
    assert ready.warnings[0].code == "invalid_relations_omitted"
    assert ready.warnings[1].code == "disconnected_nodes_omitted"
    assert graph is not None and graph.warnings == ready.warnings
    assert store.list_graph_versions(PDF_ID).versions[0].warning_count == 2


def test_failed_rebuild_preserves_current_graph_then_success_adds_version(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    PaperWorker(store).run_once()
    store.request_graph(PDF_ID)
    assert PaperWorker(store, graph_extractor=FakeExtractor(graph_json())).run_once()
    first = store.get_graph(PDF_ID)
    assert first is not None and first.graph_version == 1

    queued, needed = store.request_graph(PDF_ID, rebuild=True)
    assert needed and queued.current_version == 1
    assert PaperWorker(store, graph_extractor=FakeExtractor("not json")).run_once()
    failed = store.get_graph_record(PDF_ID)
    assert failed is not None and failed.state is GraphState.FAILED
    assert failed.current_version == 1
    assert store.get_graph(PDF_ID) == first

    store.request_graph(PDF_ID, rebuild=True)
    assert PaperWorker(store, graph_extractor=FakeExtractor(graph_json())).run_once()
    second = store.get_graph(PDF_ID)
    versions = store.list_graph_versions(PDF_ID)
    assert second is not None and second.graph_version == 2
    assert [item.version for item in versions.versions] == [1, 2]
    assert store.get_graph(PDF_ID, version=1) == first


def test_bad_graph_fails_without_losing_parsed_document_and_can_retry(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    PaperWorker(store).run_once()
    store.request_graph(PDF_ID)

    bad_worker = PaperWorker(store, graph_extractor=FakeExtractor("not json"))
    assert bad_worker.run_once()
    graph_record = store.get_graph_record(PDF_ID)
    assert graph_record is not None and graph_record.state is GraphState.FAILED
    assert store.get(PDF_ID).state is PaperState.READY  # type: ignore[union-attr]
    assert store.get_parsed_document(PDF_ID).blocks[0].block_id == BLOCK_ID

    retried, needed = store.request_graph(PDF_ID)
    assert needed and retried.state is GraphState.QUEUED
    assert PaperWorker(store, graph_extractor=FakeExtractor(graph_json())).run_once()
    assert store.get_graph_record(PDF_ID).state is GraphState.READY  # type: ignore[union-attr]


def test_graph_job_waits_when_extractor_is_not_configured(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    PaperWorker(store).run_once()
    store.request_graph(PDF_ID)

    assert not PaperWorker(store).run_once()
    assert store.get_graph_record(PDF_ID).state is GraphState.QUEUED  # type: ignore[union-attr]


def test_graph_worker_marks_stale_lock_as_interrupted(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    store.accept(PDF)
    PaperWorker(store).run_once()
    store.request_graph(PDF_ID)
    lock_path = store.source_path(PDF_ID).parent / "graph-worker.lock"
    lock_path.touch()

    worker = PaperWorker(store, graph_extractor=FakeExtractor(graph_json()))
    assert worker.run_once()

    record = store.get_graph_record(PDF_ID)
    assert record is not None and record.state is GraphState.FAILED
    assert "interrupted" in (record.error or "")
    assert not lock_path.exists()
