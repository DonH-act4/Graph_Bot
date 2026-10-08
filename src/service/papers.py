"""Authenticated internal paper upload and provenance lookup endpoints."""

from __future__ import annotations

import re
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from core import settings
from evidencegraph.access import GUEST_COOKIE, GUEST_LIFETIME, PaperAccessStore
from evidencegraph.extraction import GraphArtifact
from evidencegraph.graph_models import (
    DEFAULT_GROQ_GRAPH_MODELS,
    DEFAULT_OLLAMA_GRAPH_MODELS,
    configured_graph_models,
    graph_model_provider,
)
from evidencegraph.models import ParsedBlock
from evidencegraph.papers import (
    EvidenceBlockPage,
    GraphRecord,
    GraphReviews,
    GraphState,
    GraphVersions,
    PaperRecord,
    PaperStore,
    RelationReviewRequest,
)
from evidencegraph.quotas import QuotaStore
from schema.models import GoogleModelName
from service.auth import get_optional_account, get_paper_access_store
from service.quotas import enforce_quota, get_quota_store

router = APIRouter(prefix="/papers", tags=["papers"])
_store = PaperStore(settings.EVIDENCEGRAPH_DATA_DIR)


class PaperConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_pdf_bytes: int
    graph_models: tuple[str, ...]
    default_graph_model: str | None


class GraphRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None


def get_paper_store() -> PaperStore:
    return _store


def get_guest_identity(
    request: Request,
    response: Response,
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
    account_id: Annotated[str | None, Depends(get_optional_account)],
) -> str:
    token = request.cookies.get(GUEST_COOKIE, "")
    guest_id = access.resolve_guest(token)
    if guest_id is None:
        guest_id, token = access.create_guest()
        response.set_cookie(
            GUEST_COOKIE,
            token,
            max_age=int(GUEST_LIFETIME.total_seconds()),
            httponly=True,
            secure=settings.EVIDENCEGRAPH_COOKIE_SECURE,
            samesite="lax",
            path="/",
        )
    if settings.EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT and account_id is not None:
        return account_id
    return guest_id


def _validate_document_id(document_id: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", document_id) is None:
        raise HTTPException(status_code=422, detail="Invalid document ID")


def require_paper_read(
    document_id: str,
    guest_id: Annotated[str, Depends(get_guest_identity)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
) -> None:
    _validate_document_id(document_id)
    if not access.can_read(guest_id, document_id):
        raise HTTPException(status_code=404, detail="Paper not found")


def require_paper_owner(
    document_id: str,
    guest_id: Annotated[str, Depends(get_guest_identity)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
) -> None:
    _validate_document_id(document_id)
    if not access.owns(guest_id, document_id):
        raise HTTPException(status_code=404, detail="Paper not found")
    if access.is_public_tutorial(document_id):
        raise HTTPException(status_code=403, detail="Tutorial paper is read-only")


def get_graph_model_catalog() -> tuple[str, ...]:
    configured = settings.EVIDENCEGRAPH_GRAPH_MODELS
    if configured is None:
        defaults: list[str] = []
        if settings.OLLAMA_BASE_URL:
            defaults.extend(DEFAULT_OLLAMA_GRAPH_MODELS)
        if settings.GROQ_API_KEY:
            defaults.extend(DEFAULT_GROQ_GRAPH_MODELS)
        if settings.GOOGLE_API_KEY:
            defaults.extend(model.value for model in GoogleModelName)
        configured = ",".join(defaults)
    models = configured_graph_models(
        settings.EVIDENCEGRAPH_GRAPH_MODEL,
        configured,
    )
    return tuple(model for model in models if _graph_model_has_credentials(model))


def _graph_model_has_credentials(model: str) -> bool:
    provider = graph_model_provider(model)
    if provider == "groq":
        return bool(settings.GROQ_API_KEY)
    if provider == "ollama":
        return bool(settings.OLLAMA_BASE_URL)
    return bool(settings.GOOGLE_API_KEY)


def graph_extraction_is_configured() -> bool:
    return bool(get_graph_model_catalog())


def _resolve_graph_model(request: GraphRequest | None, models: tuple[str, ...]) -> str:
    if not models:
        raise HTTPException(status_code=503, detail="No graph extraction model is configured")
    requested = request.model if request is not None else None
    model = requested or models[0]
    if model not in models:
        raise HTTPException(
            status_code=422,
            detail=f"Graph model {model!r} is not in the configured allowlist",
        )
    return model


@router.get("/configuration", response_model=PaperConfiguration)
def get_paper_configuration(
    models: Annotated[tuple[str, ...], Depends(get_graph_model_catalog)],
    _guest_id: Annotated[str, Depends(get_guest_identity)],
) -> PaperConfiguration:
    return PaperConfiguration(
        max_pdf_bytes=settings.EVIDENCEGRAPH_MAX_PDF_BYTES,
        graph_models=models,
        default_graph_model=models[0] if models else None,
    )


@router.post("", response_model=PaperRecord, status_code=status.HTTP_202_ACCEPTED)
async def upload_paper(
    request: Request,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    guest_id: Annotated[str, Depends(get_guest_identity)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
    account_id: Annotated[str | None, Depends(get_optional_account)],
    quotas: Annotated[QuotaStore, Depends(get_quota_store)],
) -> PaperRecord:
    await run_in_threadpool(enforce_quota, request, quotas, "upload", account_id)
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/pdf":
        raise HTTPException(status_code=415, detail="Send a raw application/pdf request body")
    max_bytes = settings.EVIDENCEGRAPH_MAX_PDF_BYTES
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds the upload limit of {max_bytes} bytes",
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds the upload limit of {max_bytes} bytes",
            )
        content.extend(chunk)
    if not content:
        raise HTTPException(status_code=422, detail="PDF is empty")
    try:
        record, _needs_processing = await run_in_threadpool(store.accept, bytes(content))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await run_in_threadpool(access.grant, guest_id, record.document_id)
    return record


@router.get("/{document_id}", response_model=PaperRecord)
def get_paper(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    _authorized: Annotated[None, Depends(require_paper_read)],
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
    _authorized: Annotated[None, Depends(require_paper_read)],
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
    _authorized: Annotated[None, Depends(require_paper_read)],
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
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    _authorized: Annotated[None, Depends(require_paper_read)],
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
    http_request: Request,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    _authorized: Annotated[None, Depends(require_paper_owner)],
    account_id: Annotated[str | None, Depends(get_optional_account)],
    quotas: Annotated[QuotaStore, Depends(get_quota_store)],
    extraction_configured: Annotated[bool, Depends(graph_extraction_is_configured)],
    models: Annotated[tuple[str, ...], Depends(get_graph_model_catalog)],
    request: Annotated[GraphRequest | None, Body()] = None,
) -> GraphRecord:
    enforce_quota(http_request, quotas, "graph", account_id)
    if not extraction_configured:
        raise HTTPException(
            status_code=503,
            detail="Graph extraction requires GOOGLE_API_KEY and at least one graph model",
        )
    model = _resolve_graph_model(request, models)
    try:
        record, _needed = store.request_graph(document_id, model=model)
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
    http_request: Request,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    _authorized: Annotated[None, Depends(require_paper_owner)],
    account_id: Annotated[str | None, Depends(get_optional_account)],
    quotas: Annotated[QuotaStore, Depends(get_quota_store)],
    extraction_configured: Annotated[bool, Depends(graph_extraction_is_configured)],
    models: Annotated[tuple[str, ...], Depends(get_graph_model_catalog)],
    request: Annotated[GraphRequest | None, Body()] = None,
) -> GraphRecord:
    enforce_quota(http_request, quotas, "graph", account_id)
    if not extraction_configured:
        raise HTTPException(
            status_code=503,
            detail="Graph extraction requires GOOGLE_API_KEY and at least one graph model",
        )
    model = _resolve_graph_model(request, models)
    try:
        if store.get(document_id) is None:
            raise FileNotFoundError(document_id)
        if store.get_graph(document_id) is None:
            raise RuntimeError("Build the first graph before requesting a rebuild")
        record, _needed = store.request_graph(document_id, model=model, rebuild=True)
        return record
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Paper not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{document_id}/graph/status", response_model=GraphRecord)
def get_paper_graph_status(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    _authorized: Annotated[None, Depends(require_paper_read)],
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
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    _authorized: Annotated[None, Depends(require_paper_read)],
) -> GraphVersions:
    try:
        if store.get(document_id) is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return store.list_graph_versions(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{document_id}/graph", response_model=GraphArtifact)
def get_paper_graph(
    document_id: str,
    store: Annotated[PaperStore, Depends(get_paper_store)],
    _authorized: Annotated[None, Depends(require_paper_read)],
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
    _authorized: Annotated[None, Depends(require_paper_read)],
    version: Annotated[int | None, Query(ge=1)] = None,
) -> GraphReviews:
    try:
        if store.get(document_id) is None:
            raise HTTPException(status_code=404, detail="Paper not found")
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
    _authorized: Annotated[None, Depends(require_paper_owner)],
) -> GraphReviews:
    try:
        if store.get(document_id) is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return store.review_relation(document_id, relation_id, request.decision)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Graph relation not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
