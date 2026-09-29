"""The phase 1 internal upload slice without Docling model downloads."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evidencegraph.ingestion import (
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    PdfParseError,
    SourceLocation,
)
from evidencegraph.paper_worker import PaperWorker
from evidencegraph.papers import MAX_SAMPLE_BYTES, PaperState, PaperStore
from service.papers import get_paper_store, graph_extraction_is_configured
from service.service import app

PDF = b"%PDF-1.4\ninternal test fixture"
PDF_ID = hashlib.sha256(PDF).hexdigest()
BLOCK_ID = "blk_" + "1" * 24


@dataclass(frozen=True)
class FakeExtractor:
    name: str = "fake-api-extractor"
    version: str = "1"

    def extract(self, _document: ParsedDocument) -> str:
        return f"""
        {{
          "schema_version": "1",
          "nodes": [{{
            "node_id": "paper",
            "node_type": "paper",
            "name": "API paper",
            "document_sha256": "{PDF_ID}",
            "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}]
          }}, {{
            "node_id": "method",
            "node_type": "method",
            "name": "API method",
            "document_sha256": "{PDF_ID}",
            "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}]
          }}],
          "relations": [{{
            "relation_id": "related",
            "source_node_id": "paper",
            "target_node_id": "method",
            "relation_type": "introduces",
            "status": "candidate",
            "evidence": [{{"source_sha256": "{PDF_ID}", "block_id": "{BLOCK_ID}"}}],
            "rationale": "The evidence introduces the method."
          }}]
        }}
        """


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
                text="A source-located fact",
                locations=(
                    SourceLocation(
                        page_number=1,
                        bounding_box=BoundingBox(
                            left=1, top=2, right=3, bottom=4, coordinate_origin="bottom-left"
                        ),
                        character_start=0,
                        character_end=21,
                    ),
                ),
            ),
        ),
    )


@pytest.fixture
def client(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    app.dependency_overrides[get_paper_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.clear()


def test_upload_persists_source_status_and_evidence(client):
    http, store = client
    response = http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    assert response.status_code == 202
    assert response.json() == {
        "document_id": PDF_ID,
        "state": "processing",
        "page_count": None,
        "block_count": None,
        "error": None,
    }
    assert store.source_path(PDF_ID).read_bytes() == PDF
    store.process(PDF_ID)
    assert http.get(f"/papers/{PDF_ID}").json()["state"] == "ready"
    assert http.get(f"/papers/{PDF_ID}").json()["block_count"] == 1
    block = http.get(f"/papers/{PDF_ID}/blocks/{BLOCK_ID}")
    assert block.status_code == 200
    assert block.json()["locations"][0]["page_number"] == 1
    source = http.get(f"/papers/{PDF_ID}/source")
    assert source.status_code == 200
    assert source.content == PDF
    assert http.get(f"/papers/{PDF_ID}/blocks/unknown").status_code == 404


def test_list_blocks_is_bounded_and_source_located(client):
    http, store = client
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)
    response = http.get(f"/papers/{PDF_ID}/blocks", params={"offset": 0, "limit": 1})
    assert response.status_code == 200
    assert response.json() == {
        "document_id": PDF_ID,
        "total": 1,
        "offset": 0,
        "limit": 1,
        "items": [
            {
                "block_id": BLOCK_ID,
                "source_ref": "#/texts/0",
                "label": "text",
                "text": "A source-located fact",
                "locations": [
                    {
                        "page_number": 1,
                        "bounding_box": {
                            "left": 1.0,
                            "top": 2.0,
                            "right": 3.0,
                            "bottom": 4.0,
                            "coordinate_origin": "bottom-left",
                        },
                        "character_start": 0,
                        "character_end": 21,
                    }
                ],
            }
        ],
    }
    empty_page = http.get(f"/papers/{PDF_ID}/blocks", params={"offset": 1, "limit": 1})
    assert empty_page.status_code == 200
    assert empty_page.json()["items"] == []


def test_list_blocks_requires_ready_document(client):
    http, store = client
    store.accept(PDF)
    assert http.get(f"/papers/{PDF_ID}/blocks").status_code == 409
    assert http.get(f"/papers/{'f' * 64}/blocks").status_code == 404
    assert http.get("/papers/not-a-hash/blocks").status_code == 422


@pytest.mark.parametrize(
    ("query", "expected"), [("offset=-1", 422), ("limit=0", 422), ("limit=51", 422)]
)
def test_list_blocks_validates_pagination(client, query, expected):
    http, _store = client
    assert http.get(f"/papers/{PDF_ID}/blocks?{query}").status_code == expected


def test_duplicate_upload_is_idempotent(client):
    http, store = client
    first = http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)
    second = http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    assert first.status_code == second.status_code == 202
    assert second.json()["state"] == "ready"


def test_parser_failure_can_retry(client):
    http, store = client

    def fail(_path: Path) -> ParsedDocument:
        raise PdfParseError("PDF contains no source-located text blocks")

    store.parser = fail
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)
    failed = http.get(f"/papers/{PDF_ID}").json()
    assert failed["state"] == "failed"
    assert "no source-located" in failed["error"]
    store.parser = parsed_document
    retry = http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    assert retry.json()["state"] == "processing"
    store.process(PDF_ID)
    assert http.get(f"/papers/{PDF_ID}").json()["state"] == "ready"


def test_processing_records_survive_api_restart(tmp_path: Path):
    store = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    record, needed = store.accept(PDF)
    assert needed and record.state == PaperState.PROCESSING
    restarted = PaperStore(tmp_path, allowed_hashes=frozenset({PDF_ID}), parser=parsed_document)
    assert restarted.list_processing_records() == (record,)


@pytest.mark.parametrize(
    ("content", "content_type", "expected"),
    [
        (PDF, "text/plain", 415),
        (b"", "application/pdf", 422),
        (b"not a PDF", "application/pdf", 422),
        (b"%PDF-1.4\na different version", "application/pdf", 422),
        (b"%PDF-" + b"x" * MAX_SAMPLE_BYTES, "application/pdf", 413),
    ],
)
def test_upload_rejects_unapproved_inputs(client, content, content_type, expected):
    http, _store = client
    response = http.post("/papers", content=content, headers={"Content-Type": content_type})
    assert response.status_code == expected


def test_document_id_cannot_escape_store(client):
    http, _store = client
    assert http.get("/papers/not-a-hash").status_code == 422


def test_graph_request_requires_provider_configuration(client):
    http, store = client
    app.dependency_overrides[graph_extraction_is_configured] = lambda: False
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)

    response = http.post(f"/papers/{PDF_ID}/graph")

    assert response.status_code == 503
    assert store.get_graph_record(PDF_ID) is None


def test_graph_request_status_and_artifact_flow(client):
    http, store = client
    app.dependency_overrides[graph_extraction_is_configured] = lambda: True
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)

    requested = http.post(f"/papers/{PDF_ID}/graph")
    duplicate = http.post(f"/papers/{PDF_ID}/graph")

    assert requested.status_code == duplicate.status_code == 202
    assert requested.json()["state"] == duplicate.json()["state"] == "queued"
    assert http.get(f"/papers/{PDF_ID}/graph/status").json()["state"] == "queued"
    assert http.get(f"/papers/{PDF_ID}/graph").status_code == 409

    worker = PaperWorker(store, graph_extractor=FakeExtractor())
    assert worker.run_once()
    ready = http.get(f"/papers/{PDF_ID}/graph/status")
    artifact = http.get(f"/papers/{PDF_ID}/graph")

    assert ready.status_code == artifact.status_code == 200
    assert ready.json()["state"] == "ready"
    assert ready.json()["node_count"] == 2
    assert ready.json()["current_version"] == 1
    assert artifact.json()["extractor_name"] == "fake-api-extractor"
    assert artifact.json()["graph_version"] == 1
    assert artifact.json()["graph"]["nodes"][0]["evidence"][0]["block_id"] == BLOCK_ID

    versions = http.get(f"/papers/{PDF_ID}/graph/versions")
    assert versions.status_code == 200
    assert versions.json()["current_version"] == 1
    assert [item["version"] for item in versions.json()["versions"]] == [1]


def test_graph_request_requires_ready_paper(client):
    http, _store = client
    app.dependency_overrides[graph_extraction_is_configured] = lambda: True
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})

    response = http.post(f"/papers/{PDF_ID}/graph")

    assert response.status_code == 409


def test_graph_status_and_artifact_require_request(client):
    http, store = client
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)

    assert http.get(f"/papers/{PDF_ID}/graph/status").status_code == 404
    assert http.get(f"/papers/{PDF_ID}/graph").status_code == 404
    assert http.post(f"/papers/{PDF_ID}/graph/rebuild").status_code == 409
    assert http.post(f"/papers/{'f' * 64}/graph/rebuild").status_code == 404


def test_relation_review_is_persisted_without_mutating_candidate_graph(client):
    http, store = client
    app.dependency_overrides[graph_extraction_is_configured] = lambda: True
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)
    http.post(f"/papers/{PDF_ID}/graph")
    PaperWorker(store, graph_extractor=FakeExtractor()).run_once()

    empty = http.get(f"/papers/{PDF_ID}/graph/reviews")
    accepted = http.put(
        f"/papers/{PDF_ID}/graph/relations/related/review",
        json={"decision": "accepted"},
    )
    graph = http.get(f"/papers/{PDF_ID}/graph")

    assert empty.status_code == accepted.status_code == 200
    assert empty.json()["reviews"] == []
    assert accepted.json()["reviews"][0]["decision"] == "accepted"
    assert graph.json()["graph"]["relations"][0]["status"] == "candidate"
    assert (store.root / PDF_ID / "graphs" / "000001-reviews.json").is_file()


def test_relation_review_rejects_unknown_relation(client):
    http, store = client
    app.dependency_overrides[graph_extraction_is_configured] = lambda: True
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)
    http.post(f"/papers/{PDF_ID}/graph")
    PaperWorker(store, graph_extractor=FakeExtractor()).run_once()

    response = http.put(
        f"/papers/{PDF_ID}/graph/relations/missing/review",
        json={"decision": "rejected"},
    )

    assert response.status_code == 404


def test_rebuild_versions_graph_and_isolates_reviews(client):
    http, store = client
    app.dependency_overrides[graph_extraction_is_configured] = lambda: True
    http.post("/papers", content=PDF, headers={"Content-Type": "application/pdf"})
    store.process(PDF_ID)
    http.post(f"/papers/{PDF_ID}/graph")
    worker = PaperWorker(store, graph_extractor=FakeExtractor())
    assert worker.run_once()
    http.put(
        f"/papers/{PDF_ID}/graph/relations/related/review",
        json={"decision": "accepted"},
    )

    queued = http.post(f"/papers/{PDF_ID}/graph/rebuild")
    still_current = http.get(f"/papers/{PDF_ID}/graph")
    duplicate = http.post(f"/papers/{PDF_ID}/graph/rebuild")

    assert queued.status_code == duplicate.status_code == 202
    assert queued.json()["state"] == "queued"
    assert queued.json()["current_version"] == 1
    assert still_current.status_code == 200
    assert still_current.json()["graph_version"] == 1
    assert worker.run_once()

    versions = http.get(f"/papers/{PDF_ID}/graph/versions").json()
    current_reviews = http.get(f"/papers/{PDF_ID}/graph/reviews").json()
    first_reviews = http.get(
        f"/papers/{PDF_ID}/graph/reviews", params={"version": 1}
    ).json()
    assert versions["current_version"] == 2
    assert [item["version"] for item in versions["versions"]] == [1, 2]
    assert http.get(f"/papers/{PDF_ID}/graph", params={"version": 1}).status_code == 200
    assert current_reviews["graph_version"] == 2
    assert current_reviews["reviews"] == []
    assert first_reviews["graph_version"] == 1
    assert first_reviews["reviews"][0]["decision"] == "accepted"
