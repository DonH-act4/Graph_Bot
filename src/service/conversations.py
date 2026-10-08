"""Paper workspace state and a published real conversation for the demo page."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from core import settings
from evidencegraph.access import PaperAccessStore
from evidencegraph.conversations import ConversationStore
from evidencegraph.papers import PaperStore
from schema import ConversationState, ConversationUpdate, Showcase, UserInput
from service.identity import assert_session_owner, get_chat_identity
from service.papers import get_guest_identity, get_paper_access_store, get_paper_store

router = APIRouter()


def get_conversation_store() -> ConversationStore:
    return ConversationStore(settings.EVIDENCEGRAPH_DATA_DIR)


def _assert_editable(store: ConversationStore, thread_id: str) -> None:
    showcase = store.get_showcase()
    if showcase and showcase.thread_id == thread_id:
        raise HTTPException(status_code=403, detail="Sample conversation is read-only")


def _validate_paper_selection(update: ConversationUpdate, papers: PaperStore) -> None:
    if update.document_id is None:
        return
    try:
        if papers.get(update.document_id) is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        for block_id in update.selected_block_ids:
            if papers.get_block(update.document_id, block_id) is None:
                raise HTTPException(status_code=404, detail="Evidence block not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/showcase")
async def showcase(
    store: Annotated[ConversationStore, Depends(get_conversation_store)],
) -> Showcase | None:
    return await run_in_threadpool(store.get_showcase)


@router.get("/conversations/{thread_id}")
async def conversation(
    thread_id: str,
    user_id: Annotated[str, Query(min_length=1, max_length=200)],
    store: Annotated[ConversationStore, Depends(get_conversation_store)],
    chat_identity: Annotated[str | None, Depends(get_chat_identity)],
) -> ConversationState:
    assert_session_owner(user_id, chat_identity)
    state = await run_in_threadpool(store.get, thread_id)
    if state is None or state.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return state


@router.put("/conversations/{thread_id}")
async def save_conversation(
    thread_id: str,
    update: ConversationUpdate,
    store: Annotated[ConversationStore, Depends(get_conversation_store)],
    papers: Annotated[PaperStore, Depends(get_paper_store)],
    guest_id: Annotated[str, Depends(get_guest_identity)],
    chat_identity: Annotated[str | None, Depends(get_chat_identity)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
) -> ConversationState:
    assert_session_owner(update.user_id, chat_identity)
    if not thread_id or len(thread_id) > 200:
        raise HTTPException(status_code=422, detail="Invalid thread ID")
    await run_in_threadpool(_assert_editable, store, thread_id)
    if update.document_id is not None and not access.can_read(guest_id, update.document_id):
        raise HTTPException(status_code=404, detail="Paper not found")
    await run_in_threadpool(_validate_paper_selection, update, papers)
    try:
        return await run_in_threadpool(store.save, thread_id, update)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc


async def record_chat_workspace(
    user_input: UserInput,
    agent_id: str,
    store: ConversationStore,
    papers: PaperStore,
) -> None:
    """Record workspace metadata only for explicitly identified chat threads."""
    if user_input.thread_id is None:
        return
    await run_in_threadpool(_assert_editable, store, user_input.thread_id)
    if user_input.user_id is None:
        return
    document_id = user_input.document_id
    if user_input.evidence_context is not None:
        selected_document = user_input.evidence_context.document_id
        if document_id is not None and document_id != selected_document:
            raise HTTPException(status_code=422, detail="Evidence belongs to a different paper")
        document_id = selected_document
    if document_id is not None:
        await run_in_threadpool(
            _validate_paper_selection,
            ConversationUpdate(user_id=user_input.user_id, document_id=document_id),
            papers,
        )
    try:
        await run_in_threadpool(
            store.record_turn,
            user_input.thread_id,
            user_input.user_id,
            agent_id,
            user_input.message,
            document_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
