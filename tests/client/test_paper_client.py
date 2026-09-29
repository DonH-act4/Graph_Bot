"""HTTP contract for the internal paper upload/status client."""

from unittest.mock import patch

import httpx
import pytest

from client import (
    AgentClient,
    AgentClientError,
    EvidenceBlockPage,
    GraphArtifact,
    GraphReviews,
    GraphStatus,
    GraphVersions,
    PaperStatus,
)

PAPER_ID = "a" * 64


def test_upload_paper_sends_raw_pdf_with_auth():
    client = AgentClient(base_url="http://test", get_info=False)
    client.auth_secret = "test-secret"
    request = httpx.Request("POST", "http://test/papers")
    response = httpx.Response(
        202, json={"document_id": PAPER_ID, "state": "processing"}, request=request
    )
    with patch("httpx.post", return_value=response) as post:
        result = client.upload_paper(b"%PDF-1.4")
    assert result == PaperStatus(document_id=PAPER_ID, state="processing")
    post.assert_called_once_with(
        "http://test/papers",
        content=b"%PDF-1.4",
        headers={"Authorization": "Bearer test-secret", "Content-Type": "application/pdf"},
        timeout=30.0,
    )


def test_get_paper_reads_status_with_auth():
    client = AgentClient(base_url="http://test", get_info=False)
    client.auth_secret = "test-secret"
    request = httpx.Request("GET", f"http://test/papers/{PAPER_ID}")
    response = httpx.Response(
        200,
        json={"document_id": PAPER_ID, "state": "ready", "page_count": 15, "block_count": 154},
        request=request,
    )
    with patch("httpx.get", return_value=response) as get:
        result = client.get_paper(PAPER_ID)
    assert result.page_count == 15
    assert result.block_count == 154
    get.assert_called_once_with(
        f"http://test/papers/{PAPER_ID}",
        headers={"Authorization": "Bearer test-secret"},
        timeout=10.0,
    )


def test_upload_error_shows_backend_reason():
    client = AgentClient(base_url="http://test", get_info=False)
    request = httpx.Request("POST", "http://test/papers")
    response = httpx.Response(422, json={"detail": "Only phase 0 samples"}, request=request)
    with patch("httpx.post", return_value=response):
        with pytest.raises(AgentClientError, match="422.*Only phase 0 samples"):
            client.upload_paper(b"%PDF-1.4")


def test_network_error_hides_transport_details():
    client = AgentClient(base_url="http://test", get_info=False)
    with patch("httpx.get", side_effect=httpx.ConnectError("private transport detail")):
        with pytest.raises(AgentClientError, match="Paper service is unavailable") as exc:
            client.get_paper(PAPER_ID)
    assert "private transport detail" not in str(exc.value)


@pytest.mark.parametrize("body", ["not-json", '{"document_id":"bad","state":"unknown"}'])
def test_invalid_status_response_is_reported(body):
    client = AgentClient(base_url="http://test", get_info=False)
    request = httpx.Request("GET", f"http://test/papers/{PAPER_ID}")
    response = httpx.Response(200, text=body, request=request)
    with patch("httpx.get", return_value=response):
        with pytest.raises(AgentClientError, match="invalid status response"):
            client.get_paper(PAPER_ID)


def test_get_paper_blocks_reads_bounded_page():
    client = AgentClient(base_url="http://test", get_info=False)
    request = httpx.Request("GET", f"http://test/papers/{PAPER_ID}/blocks")
    response = httpx.Response(
        200,
        json={
            "document_id": PAPER_ID,
            "total": 1,
            "offset": 0,
            "limit": 10,
            "items": [
                {
                    "block_id": "blk_test",
                    "source_ref": "#/texts/0",
                    "label": "text",
                    "text": "A fact",
                    "locations": [
                        {
                            "page_number": 2,
                            "bounding_box": {
                                "left": 1,
                                "top": 2,
                                "right": 3,
                                "bottom": 4,
                                "coordinate_origin": "bottom-left",
                            },
                            "character_start": 0,
                            "character_end": 6,
                        }
                    ],
                }
            ],
        },
        request=request,
    )
    with patch("httpx.get", return_value=response) as get:
        page = client.get_paper_blocks(PAPER_ID)
    assert isinstance(page, EvidenceBlockPage)
    assert page.items[0].locations[0].page_number == 2
    get.assert_called_once_with(
        f"http://test/papers/{PAPER_ID}/blocks",
        params={"offset": 0, "limit": 10},
        headers={},
        timeout=10.0,
    )


def test_get_paper_blocks_rejects_invalid_response():
    client = AgentClient(base_url="http://test", get_info=False)
    request = httpx.Request("GET", f"http://test/papers/{PAPER_ID}/blocks")
    response = httpx.Response(200, json={"items": "not-a-list"}, request=request)
    with patch("httpx.get", return_value=response):
        with pytest.raises(AgentClientError, match="invalid evidence blocks"):
            client.get_paper_blocks(PAPER_ID)


def test_get_paper_block_reads_exact_evidence():
    client = AgentClient(base_url="http://test", get_info=False)
    request = httpx.Request("GET", f"http://test/papers/{PAPER_ID}/blocks/blk_test")
    response = httpx.Response(
        200,
        json={
            "block_id": "blk_test",
            "source_ref": "#/texts/0",
            "label": "text",
            "text": "A fact",
            "locations": [
                {
                    "page_number": 2,
                    "bounding_box": {
                        "left": 1,
                        "top": 2,
                        "right": 3,
                        "bottom": 4,
                        "coordinate_origin": "bottom-left",
                    },
                    "character_start": 0,
                    "character_end": 6,
                }
            ],
        },
        request=request,
    )
    with patch("httpx.get", return_value=response) as get:
        block = client.get_paper_block(PAPER_ID, "blk_test")
    assert block.text == "A fact"
    get.assert_called_once_with(
        f"http://test/papers/{PAPER_ID}/blocks/blk_test", headers={}, timeout=10.0
    )


def test_request_and_read_graph_status():
    client = AgentClient(base_url="http://test", get_info=False)
    client.auth_secret = "test-secret"
    request = httpx.Request("POST", f"http://test/papers/{PAPER_ID}/graph")
    response = httpx.Response(
        202, json={"document_id": PAPER_ID, "state": "queued"}, request=request
    )
    with patch("httpx.request", return_value=response) as send:
        result = client.request_paper_graph(PAPER_ID)
    assert result == GraphStatus(document_id=PAPER_ID, state="queued")
    send.assert_called_once_with(
        "POST",
        f"http://test/papers/{PAPER_ID}/graph",
        headers={"Authorization": "Bearer test-secret"},
        timeout=10.0,
    )


def test_get_graph_reads_validated_artifact():
    client = AgentClient(base_url="http://test", get_info=False)
    request = httpx.Request("GET", f"http://test/papers/{PAPER_ID}/graph")
    response = httpx.Response(
        200,
        json={
            "schema_version": "1",
            "document_sha256": PAPER_ID,
            "extractor_name": "google-gemini",
            "extractor_version": "test-model",
            "graph": {
                "schema_version": "1",
                "nodes": [
                    {
                        "node_id": "paper",
                        "node_type": "paper",
                        "name": "Paper",
                        "document_sha256": PAPER_ID,
                        "evidence": [
                            {"source_sha256": PAPER_ID, "block_id": "blk_test"}
                        ],
                    }
                ],
                "relations": [],
            },
        },
        request=request,
    )
    with patch("httpx.get", return_value=response):
        artifact = client.get_paper_graph(PAPER_ID)
    assert isinstance(artifact, GraphArtifact)
    assert artifact.graph.nodes[0].evidence[0].block_id == "blk_test"


def test_get_graph_rejects_invalid_response():
    client = AgentClient(base_url="http://test", get_info=False)
    request = httpx.Request("GET", f"http://test/papers/{PAPER_ID}/graph")
    response = httpx.Response(200, json={"graph": "bad"}, request=request)
    with patch("httpx.get", return_value=response):
        with pytest.raises(AgentClientError, match="invalid graph"):
            client.get_paper_graph(PAPER_ID)


def test_read_and_update_graph_reviews():
    client = AgentClient(base_url="http://test", get_info=False)
    payload = {
        "schema_version": "1",
        "document_sha256": PAPER_ID,
        "reviews": [
            {
                "relation_id": "rel_1",
                "decision": "accepted",
                "reviewed_at": "2026-09-27T10:00:00Z",
            }
        ],
    }
    get_response = httpx.Response(
        200,
        json=payload,
        request=httpx.Request("GET", f"http://test/papers/{PAPER_ID}/graph/reviews"),
    )
    put_response = httpx.Response(
        200,
        json=payload,
        request=httpx.Request(
            "PUT", f"http://test/papers/{PAPER_ID}/graph/relations/rel_1/review"
        ),
    )

    with patch("httpx.request", side_effect=[get_response, put_response]) as request:
        reviews = client.get_paper_graph_reviews(PAPER_ID)
        updated = client.review_paper_graph_relation(PAPER_ID, "rel_1", "accepted")

    assert isinstance(reviews, GraphReviews)
    assert updated.reviews[0].decision == "accepted"
    assert request.call_args_list[1].kwargs["json"] == {"decision": "accepted"}


def test_rebuild_and_list_graph_versions():
    client = AgentClient(base_url="http://test", get_info=False)
    rebuild_response = httpx.Response(
        202,
        json={"document_id": PAPER_ID, "state": "queued", "current_version": 1},
        request=httpx.Request("POST", f"http://test/papers/{PAPER_ID}/graph/rebuild"),
    )
    versions_response = httpx.Response(
        200,
        json={
            "document_id": PAPER_ID,
            "current_version": 1,
            "versions": [
                {
                    "version": 1,
                    "extractor_name": "google-gemini",
                    "extractor_version": "test-model",
                    "node_count": 2,
                    "relation_count": 1,
                }
            ],
        },
        request=httpx.Request("GET", f"http://test/papers/{PAPER_ID}/graph/versions"),
    )

    with patch("httpx.request", return_value=rebuild_response):
        rebuilt = client.rebuild_paper_graph(PAPER_ID)
    with patch("httpx.get", return_value=versions_response):
        versions = client.get_paper_graph_versions(PAPER_ID)

    assert rebuilt.current_version == 1
    assert isinstance(versions, GraphVersions)
    assert versions.versions[0].version == 1
