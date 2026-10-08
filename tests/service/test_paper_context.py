"""Current-paper context is server-resolved for unselected and follow-up turns."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import StateSnapshot

from evidencegraph.access import PaperAccessStore
from evidencegraph.models import BoundingBox, ParsedBlock, ParsedDocument, SourceLocation
from schema import EvidenceContextInput, UserInput
from service import app
from service.papers import get_paper_access_store, get_paper_store
from service.service import _handle_input, _resolve_evidence_context

DOCUMENT_ID = "a" * 64
BLOCK_ID = "blk_" + "1" * 24


class SourceStore:
    def __init__(self) -> None:
        text = "The paper developed a catalyst for removing pollutants from water."
        self.block = ParsedBlock(
            block_id=BLOCK_ID, source_ref="#/texts/0", label="text", text=text,
            locations=(SourceLocation(
                page_number=2,
                bounding_box=BoundingBox(
                    left=0, top=0, right=1, bottom=1, coordinate_origin="bottom-left"
                ),
                character_start=0, character_end=len(text),
            ),),
        )
        self.parsed = ParsedDocument(
            source_filename="paper.pdf", source_sha256=DOCUMENT_ID,
            parser_version="test", page_count=2, blocks=(self.block,),
        )

    def get(self, document_id):
        return SimpleNamespace() if document_id == DOCUMENT_ID else None

    def get_parsed_document(self, document_id):
        assert document_id == DOCUMENT_ID
        return self.parsed

    def get_block(self, document_id, block_id):
        assert document_id == DOCUMENT_ID
        return self.block if block_id == BLOCK_ID else None


def test_current_paper_automatically_resolves_server_source() -> None:
    store = SourceStore()
    context = _resolve_evidence_context(None, store, DOCUMENT_ID, "做了什么？")  # type: ignore[arg-type]
    assert context == {
        "document_id": DOCUMENT_ID,
        "source_mode": "automatic",
        "blocks": [{"block_id": BLOCK_ID, "pages": [2], "text": store.block.text}],
    }


def test_invoke_current_paper_without_a_manual_selection(test_client, mock_agent) -> None:
    app.dependency_overrides[get_paper_store] = SourceStore
    try:
        response = test_client.post("/invoke", json={
            "message": "论文做了什么？", "document_id": DOCUMENT_ID,
        })
    finally:
        app.dependency_overrides.pop(get_paper_store, None)
    assert response.status_code == 200
    context = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0].additional_kwargs["evidence_context"]
    assert context["source_mode"] == "automatic"
    assert context["blocks"][0]["block_id"] == BLOCK_ID


def test_invoke_unparsed_paper_returns_actionable_error(test_client, mock_agent) -> None:
    class UnparsedStore(SourceStore):
        def get_parsed_document(self, document_id):
            raise RuntimeError("Parsed document is available only after parsing completes")

    app.dependency_overrides[get_paper_store] = UnparsedStore
    try:
        response = test_client.post("/invoke", json={
            "message": "论文做了什么？", "document_id": DOCUMENT_ID,
        })
    finally:
        app.dependency_overrides.pop(get_paper_store, None)
    assert response.status_code == 409
    assert "parsing completes" in response.json()["detail"]
    mock_agent.ainvoke.assert_not_awaited()


def test_invoke_cannot_resolve_another_guests_paper(test_client, mock_agent, tmp_path: Path):
    app.dependency_overrides[get_paper_store] = SourceStore
    app.dependency_overrides[get_paper_access_store] = lambda: PaperAccessStore(tmp_path)
    try:
        response = test_client.post(
            "/invoke", json={"message": "Show me the source", "document_id": DOCUMENT_ID}
        )
    finally:
        app.dependency_overrides.pop(get_paper_store, None)
        app.dependency_overrides.pop(get_paper_access_store, None)
    assert response.status_code == 404
    mock_agent.ainvoke.assert_not_awaited()


def test_selected_context_takes_priority_and_wrong_paper_is_rejected() -> None:
    store = SourceStore()
    selection = EvidenceContextInput(document_id=DOCUMENT_ID, block_ids=[BLOCK_ID])
    context = _resolve_evidence_context(selection, store, DOCUMENT_ID, "做了什么？")  # type: ignore[arg-type]
    assert context is not None and context["source_mode"] == "selected"
    with pytest.raises(HTTPException) as error:
        _resolve_evidence_context(selection, store, "b" * 64, "Question")  # type: ignore[arg-type]
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_short_followup_preserves_previous_selection(mock_agent) -> None:
    selected = {
        "document_id": DOCUMENT_ID, "source_mode": "selected",
        "blocks": [{"block_id": BLOCK_ID, "pages": [2], "text": "Selected result."}],
    }
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [HumanMessage(
            content="解释这段", additional_kwargs={"evidence_context": selected}
        )]}, next=(), config={}, metadata=None, created_at=None,
        parent_config=None, tasks=(), interrupts=(),
    )
    kwargs, _ = await _handle_input(
        UserInput(message="为什么？", document_id=DOCUMENT_ID), mock_agent,
        "research-assistant", {"document_id": DOCUMENT_ID, "source_mode": "automatic", "blocks": []},
    )
    message = kwargs["input"]["messages"][0]
    assert message.additional_kwargs["evidence_context"] == selected
    assert message.additional_kwargs["document_id"] == DOCUMENT_ID


@pytest.mark.asyncio
async def test_followup_does_not_reuse_selection_from_another_paper(mock_agent) -> None:
    mock_agent.aget_state.return_value = StateSnapshot(
        values={"messages": [HumanMessage(
            content="Question", additional_kwargs={"evidence_context": {
                "document_id": "b" * 64, "source_mode": "selected", "blocks": [],
            }}
        )]}, next=(), config={}, metadata=None, created_at=None,
        parent_config=None, tasks=(), interrupts=(),
    )
    automatic = {"document_id": DOCUMENT_ID, "source_mode": "automatic", "blocks": []}
    kwargs, _ = await _handle_input(
        UserInput(message="为什么？", document_id=DOCUMENT_ID), mock_agent,
        "research-assistant", automatic,
    )
    assert kwargs["input"]["messages"][0].additional_kwargs["evidence_context"] == automatic
