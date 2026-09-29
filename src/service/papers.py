"""Authenticated internal paper upload and provenance lookup endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from core import settings
from evidencegraph.extraction import GraphArtifact
from evidencegraph.models import ParsedBlock
from evidencegraph.papers import (
    MAX_SAMPLE_BYTES,
    EvidenceBlockPage,
    GraphRecord,
    GraphReviews,
    GraphState,
    GraphVersions,
    PaperRecord,
    PaperStore,
    RelationReviewRequest,
)

router = APIRouter(prefix="/papers", tags=["papers"])
_store = PaperStore(settings.EVIDENCEGRAPH_DATA_DIR)


def get_paper_store() -> PaperStore:
    return _store


def graph_extraction_is_configured() -> bool:
    return bool(settings.GOOGLE_API_KEY and settings.EVIDENCEGRAPH_GRAPH_MODEL)


@router.post("", response_model=PaperRecord, status_code=status.HTTP_202_ACCEPTED)
async def upload_paper(
    request: Request,
    store: Annotated[PaperStore, Depends(get_paper_store)],
) -> PaperRecord:
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/pdf":
        raise HTTPException(status_code=415, detail="Send a raw application/pdf request body")
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_SAMPLE_BYTES:
            raise HTTPException(status_code=413, detail="PDF exceeds the phase 0 sample size")
        content.extend(chunk)
    if not content:
        raise HTTPException(status_code=422, detail="PDF is empty")
    try:
        record, _needs_processing = await run_in_threadpool(store.accept, bytes(content))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record


@router.get("/{document_id}", response_model=PaperRecord)
def get_paper(
    document_id: str, store: Annotated[PaperStore, Depends(get_paper_store)]
) -> PaperRecord:
    try:
        record = store.get(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return record


@router.get("/{document_id}/blocks/{block_id}", response_model=ParsedBlock)
def get_paper_block(
    document_id: str,
    block_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
) -> ParsedBlock:
    try:
        block = store.get_block(document_id, block_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if block is None:
        raise HTTPException(status_code=404, detail="Evidence block not found")
    return block


@router.get("/{document_id}/blocks", response_model=EvidenceBlockPage)
def list_paper_blocks(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> EvidenceBlockPage:
    try:
        return store.list_blocks(document_id, offset=offset, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Paper not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{document_id}/source")
def get_paper_source(
    document_id: str, store: Annotated[PaperStore, Depends(get_paper_store)]
) -> FileResponse:
    try:
        record = store.get(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return FileResponse(store.source_path(document_id), media_type="application/pdf")


@router.post(
    "/{document_id}/graph", response_model=GraphRecord, status_code=status.HTTP_202_ACCEPTED
)
def request_paper_graph(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    extraction_configured: Annotated[bool, Depends(graph_extraction_is_configured)],
) -> GraphRecord:
    if not extraction_configured:
        raise HTTPException(
            status_code=503,
            detail="Graph extraction requires GOOGLE_API_KEY and EVIDENCEGRAPH_GRAPH_MODEL",
        )
    try:
        record, _needed = store.request_graph(document_id)
        return record
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Paper not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/{document_id}/graph/rebuild",
    response_model=GraphRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
def rebuild_paper_graph(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    extraction_configured: Annotated[bool, Depends(graph_extraction_is_configured)],
) -> GraphRecord:
    if not extraction_configured:
        raise HTTPException(
            status_code=503,
            detail="Graph extraction requires GOOGLE_API_KEY and EVIDENCEGRAPH_GRAPH_MODEL",
        )
    try:
        if store.get(document_id) is None:
            raise FileNotFoundError(document_id)
        if store.get_graph(document_id) is None:
            raise RuntimeError("Build the first graph before requesting a rebuild")
        record, _needed = store.request_graph(document_id, rebuild=True)
        return record
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Paper not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{document_id}/graph/status", response_model=GraphRecord)
def get_paper_graph_status(
    document_id: str, store: Annotated[PaperStore, Depends(get_paper_store)]
) -> GraphRecord:
    try:
        record = store.get_graph_record(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Graph extraction has not been requested")
    return record


@router.get("/{document_id}/graph/versions", response_model=GraphVersions)
def list_paper_graph_versions(
    document_id: str, store: Annotated[PaperStore, Depends(get_paper_store)]
) -> GraphVersions:
    try:
        return store.list_graph_versions(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{document_id}/graph", response_model=GraphArtifact)
def get_paper_graph(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    version: Annotated[int | None, Query(ge=1)] = None,
) -> GraphArtifact:
    try:
        record = store.get_graph_record(document_id)
        artifact = store.get_graph(document_id, version=version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Graph extraction has not been requested")
    if artifact is not None:
        return artifact
    if version is not None:
        raise HTTPException(status_code=404, detail="Graph version not found")
    if record.state is not GraphState.READY:
        raise HTTPException(status_code=409, detail=f"Graph is {record.state}")
    raise HTTPException(status_code=500, detail="Ready graph artifact is missing")


@router.get("/{document_id}/graph/reviews", response_model=GraphReviews)
def get_paper_graph_reviews(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    version: Annotated[int | None, Query(ge=1)] = None,
) -> GraphReviews:
    try:
        return store.get_graph_reviews(document_id, version=version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put(
    "/{document_id}/graph/relations/{relation_id}/review",
    response_model=GraphReviews,
)
def review_paper_graph_relation(
    document_id: str,
    relation_id: str,
    request: RelationReviewRequest,
    store: Annotated[PaperStore, Depends(get_paper_store)],
) -> GraphReviews:
    try:
        return store.review_relation(document_id, relation_id, request.decision)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Graph relation not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
