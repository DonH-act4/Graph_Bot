"""Small, local paper-processing store for the phase 1 internal slice."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evidencegraph.extraction import (
    GraphArtifact,
    GraphExtractionError,
    GraphExtractor,
    extract_graph,
)
from evidencegraph.models import ParsedBlock, ParsedDocument, PdfParseError

SAMPLE_SHA256 = frozenset(
    {
        "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
        "5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e",
        "76a3872d244793563a5b000b818cbaf0ca8972ab1145d32be474c05b2a8f3070",
    }
)
# Only the three measured phase 0 fixtures are accepted, not a general upload limit.
MAX_SAMPLE_BYTES = 2_215_244
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


def _parser_not_configured(_path: Path) -> ParsedDocument:
    raise RuntimeError("This process is not configured to parse PDFs")


class PaperState(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class GraphState(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class ReviewDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PaperRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    state: PaperState
    page_count: int | None = None
    block_count: int | None = None
    error: str | None = None


class GraphRecord(BaseModel):
    """Lifecycle of an automatic graph, independent from PDF parsing state."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    state: GraphState
    node_count: int | None = None
    relation_count: int | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    current_version: int | None = None
    error: str | None = None


class GraphVersionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    extractor_name: str
    extractor_version: str
    node_count: int
    relation_count: int


class GraphVersions(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    current_version: int
    versions: tuple[GraphVersionSummary, ...]


class RelationReview(BaseModel):
    """One current human decision without mutating the model-produced graph."""

    model_config = ConfigDict(frozen=True)

    relation_id: str
    decision: ReviewDecision
    reviewed_at: datetime


class GraphReviews(BaseModel):
    """Human review layer for one exact graph/PDF version."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    document_sha256: str
    graph_version: int = 1
    reviews: tuple[RelationReview, ...] = ()


class RelationReviewRequest(BaseModel):
    decision: ReviewDecision


class EvidenceBlockPage(BaseModel):
    """One stable page of source-located evidence blocks."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    total: int
    offset: int
    limit: int
    items: tuple[ParsedBlock, ...]


class PaperStore:
    """Persist one exact-hash PDF and its provenance blocks without a new database."""

    def __init__(
        self,
        root: Path,
        *,
        allowed_hashes: frozenset[str] = SAMPLE_SHA256,
        parser: Callable[[Path], ParsedDocument] = _parser_not_configured,
    ) -> None:
        self.root = root
        self.allowed_hashes = allowed_hashes
        self.parser = parser
        self._lock = Lock()

    def accept(self, content: bytes) -> tuple[PaperRecord, bool]:
        """Save an approved sample; return whether a new processing task is needed."""
        if not content.startswith(b"%PDF-"):
            raise ValueError("Expected a PDF file")
        document_id = hashlib.sha256(content).hexdigest()
        if document_id not in self.allowed_hashes:
            raise ValueError("Only the three phase 0 sample PDF versions are accepted")
        with self._lock:
            existing = self.get(document_id)
            if existing and existing.state != PaperState.FAILED:
                return existing, False
            if any(
                record.document_id != document_id for record in self.list_processing_records()
            ):
                raise RuntimeError("A PDF is already processing; try again later")
            folder = self.root / document_id
            folder.mkdir(parents=True, exist_ok=True)
            self._atomic_write(folder / "source.pdf", content)
            record = PaperRecord(document_id=document_id, state=PaperState.PROCESSING)
            self._save_record(record)
            return record, True

    def process(self, document_id: str) -> None:
        """Run Docling after the HTTP response and persist either result or failure."""
        self._folder(document_id)
        try:
            parsed = self.parser(self.source_path(document_id))
            if parsed.source_sha256 != document_id:
                raise PdfParseError("Parser returned a different PDF version")
            self._atomic_write(
                self._folder(document_id) / "parsed.json",
                parsed.model_dump_json(indent=2).encode(),
            )
            self._save_record(
                PaperRecord(
                    document_id=document_id,
                    state=PaperState.READY,
                    page_count=parsed.page_count,
                    block_count=len(parsed.blocks),
                )
            )
        except PdfParseError as exc:
            self._save_record(
                PaperRecord(document_id=document_id, state=PaperState.FAILED, error=str(exc))
            )
        except Exception:
            self._save_record(
                PaperRecord(
                    document_id=document_id,
                    state=PaperState.FAILED,
                    error="Unexpected PDF processing failure",
                )
            )
            raise

    def get(self, document_id: str) -> PaperRecord | None:
        path = self._folder(document_id) / "record.json"
        if not path.is_file():
            return None
        return PaperRecord.model_validate_json(path.read_bytes())

    def get_block(self, document_id: str, block_id: str) -> ParsedBlock | None:
        try:
            parsed = self.get_parsed_document(document_id)
        except (FileNotFoundError, RuntimeError):
            return None
        return next((block for block in parsed.blocks if block.block_id == block_id), None)

    def list_blocks(self, document_id: str, *, offset: int, limit: int) -> EvidenceBlockPage:
        """Return a bounded slice without exposing the whole parsed artifact."""
        parsed = self.get_parsed_document(document_id)
        return EvidenceBlockPage(
            document_id=document_id,
            total=len(parsed.blocks),
            offset=offset,
            limit=limit,
            items=parsed.blocks[offset : offset + limit],
        )

    def get_parsed_document(self, document_id: str) -> ParsedDocument:
        """Load the complete parsed artifact for an internal domain service."""
        record = self.get(document_id)
        if record is None:
            raise FileNotFoundError(document_id)
        if record.state != PaperState.READY:
            raise RuntimeError("Parsed document is available only after parsing completes")
        return ParsedDocument.model_validate_json(
            (self._folder(document_id) / "parsed.json").read_bytes()
        )

    def save_graph(self, artifact: GraphArtifact) -> GraphArtifact:
        """Persist a new immutable graph version and return its assigned version."""
        document = self.get_parsed_document(artifact.document_sha256)
        artifact.graph.validate_against((document,))
        folder = self._folder(artifact.document_sha256)
        versions_folder = folder / "graphs"
        versions_folder.mkdir(exist_ok=True)
        legacy_path = folder / "graph.json"
        version_paths = tuple(versions_folder.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        if legacy_path.is_file() and not self._graph_version_path(
            artifact.document_sha256, 1
        ).is_file():
            legacy = GraphArtifact.model_validate_json(legacy_path.read_bytes()).model_copy(
                update={"graph_version": 1}
            )
            self._atomic_write(
                self._graph_version_path(artifact.document_sha256, 1),
                legacy.model_dump_json(indent=2).encode(),
            )
        existing_versions = [int(path.stem) for path in version_paths]
        if legacy_path.is_file():
            existing_versions.append(1)
        next_version = max(existing_versions, default=0) + 1
        versioned = artifact.model_copy(update={"graph_version": next_version})
        self._atomic_write(
            self._graph_version_path(artifact.document_sha256, next_version),
            versioned.model_dump_json(indent=2).encode(),
        )
        return versioned

    def get_graph(
        self, document_id: str, *, version: int | None = None
    ) -> GraphArtifact | None:
        if version is None:
            record = self.get_graph_record(document_id)
            version = record.current_version if record is not None else None
        if version is not None:
            path = self._graph_version_path(document_id, version)
            if path.is_file():
                return GraphArtifact.model_validate_json(path.read_bytes())
            if version != 1:
                return None
        path = self._folder(document_id) / "graph.json"
        if not path.is_file():
            return None
        return GraphArtifact.model_validate_json(path.read_bytes()).model_copy(
            update={"graph_version": 1}
        )

    def list_graph_versions(self, document_id: str) -> GraphVersions:
        graph = self.get_graph(document_id)
        if graph is None:
            raise RuntimeError("Graph has no successful version")
        versions_folder = self._folder(document_id) / "graphs"
        versions: dict[int, GraphArtifact] = {}
        if versions_folder.is_dir():
            for path in versions_folder.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"):
                artifact = GraphArtifact.model_validate_json(path.read_bytes())
                versions[artifact.graph_version] = artifact
        legacy = self._folder(document_id) / "graph.json"
        if legacy.is_file() and 1 not in versions:
            versions[1] = GraphArtifact.model_validate_json(legacy.read_bytes()).model_copy(
                update={"graph_version": 1}
            )
        summaries = tuple(
            GraphVersionSummary(
                version=version,
                extractor_name=item.extractor_name,
                extractor_version=item.extractor_version,
                node_count=len(item.graph.nodes),
                relation_count=len(item.graph.relations),
            )
            for version, item in sorted(versions.items())
        )
        return GraphVersions(
            document_id=document_id,
            current_version=graph.graph_version,
            versions=summaries,
        )

    def get_graph_reviews(
        self, document_id: str, *, version: int | None = None
    ) -> GraphReviews:
        """Read human decisions while keeping an unreviewed graph valid."""
        graph = self.get_graph(document_id, version=version)
        if graph is None:
            raise RuntimeError("Graph is not ready for review")
        path = self._graph_review_path(document_id, graph.graph_version)
        if not path.is_file() and graph.graph_version == 1:
            path = self._folder(document_id) / "graph-reviews.json"
        if not path.is_file():
            return GraphReviews(
                document_sha256=document_id, graph_version=graph.graph_version
            )
        reviews = GraphReviews.model_validate_json(path.read_bytes())
        if (
            reviews.document_sha256 != graph.document_sha256
            or reviews.graph_version != graph.graph_version
        ):
            raise RuntimeError("Graph review belongs to another graph version")
        return reviews

    def review_relation(
        self, document_id: str, relation_id: str, decision: ReviewDecision
    ) -> GraphReviews:
        """Atomically set one human decision for a relation in the current graph."""
        with self._lock:
            graph = self.get_graph(document_id)
            if graph is None:
                raise RuntimeError("Graph is not ready for review")
            relation_ids = tuple(
                relation.relation_id for relation in graph.graph.relations
            )
            if relation_id not in relation_ids:
                raise KeyError(relation_id)
            existing = self.get_graph_reviews(document_id)
            review_by_id = {review.relation_id: review for review in existing.reviews}
            review_by_id[relation_id] = RelationReview(
                relation_id=relation_id,
                decision=decision,
                reviewed_at=datetime.now(UTC),
            )
            reviews = GraphReviews(
                document_sha256=document_id,
                graph_version=graph.graph_version,
                reviews=tuple(
                    review_by_id[current_id]
                    for current_id in relation_ids
                    if current_id in review_by_id
                ),
            )
            review_path = self._graph_review_path(document_id, graph.graph_version)
            review_path.parent.mkdir(exist_ok=True)
            self._atomic_write(
                review_path,
                reviews.model_dump_json(indent=2).encode(),
            )
            return reviews

    def request_graph(
        self, document_id: str, *, rebuild: bool = False
    ) -> tuple[GraphRecord, bool]:
        """Queue an automatic graph after parsing; failed requests can be retried."""
        self.get_parsed_document(document_id)
        with self._lock:
            existing = self.get_graph_record(document_id)
            has_current = self.get_graph(document_id) is not None
            if existing is not None and existing.state in {
                GraphState.QUEUED,
                GraphState.PROCESSING,
            }:
                return existing, False
            if has_current and not rebuild:
                return existing or GraphRecord(
                    document_id=document_id,
                    state=GraphState.READY,
                    current_version=1,
                ), False
            record = GraphRecord(
                document_id=document_id,
                state=GraphState.QUEUED,
                current_version=(
                    existing.current_version
                    if existing is not None and existing.current_version is not None
                    else (1 if has_current else None)
                ),
            )
            self._save_graph_record(record)
            return record, True

    def get_graph_record(self, document_id: str) -> GraphRecord | None:
        path = self._folder(document_id) / "graph-record.json"
        if not path.is_file():
            return None
        return GraphRecord.model_validate_json(path.read_bytes())

    def list_queued_graph_records(self) -> tuple[GraphRecord, ...]:
        """Return graph jobs waiting for the single local worker."""
        if not self.root.exists():
            return ()
        records = []
        for path in self.root.glob("*/graph-record.json"):
            if not _HASH_RE.fullmatch(path.parent.name):
                continue
            record = GraphRecord.model_validate_json(path.read_bytes())
            if record.state is GraphState.QUEUED:
                records.append(record)
        return tuple(records)

    def mark_graph_processing(self, document_id: str) -> None:
        existing = self.get_graph_record(document_id)
        self._save_graph_record(
            GraphRecord(
                document_id=document_id,
                state=GraphState.PROCESSING,
                current_version=(
                    existing.current_version
                    if existing is not None and existing.current_version is not None
                    else (1 if self.get_graph(document_id) is not None else None)
                ),
            )
        )

    def process_graph(self, document_id: str, extractor: GraphExtractor) -> None:
        """Extract and persist one graph, preserving parsed evidence on failure."""
        document = self.get_parsed_document(document_id)
        try:
            artifact = extract_graph(document, extractor)
            versioned = self.save_graph(artifact)
            self._save_graph_record(
                GraphRecord(
                    document_id=document_id,
                    state=GraphState.READY,
                    node_count=len(artifact.graph.nodes),
                    relation_count=len(artifact.graph.relations),
                    extractor_name=artifact.extractor_name,
                    extractor_version=artifact.extractor_version,
                    current_version=versioned.graph_version,
                )
            )
        except GraphExtractionError as exc:
            self.mark_graph_failed(document_id, str(exc))
        except Exception:
            self.mark_graph_failed(document_id, "Unexpected graph extraction failure")
            raise

    def mark_graph_failed(self, document_id: str, error: str) -> None:
        existing = self.get_graph_record(document_id)
        self._save_graph_record(
            GraphRecord(
                document_id=document_id,
                state=GraphState.FAILED,
                current_version=(
                    existing.current_version
                    if existing is not None and existing.current_version is not None
                    else (1 if self.get_graph(document_id) is not None else None)
                ),
                error=error,
            )
        )

    def source_path(self, document_id: str) -> Path:
        return self._folder(document_id) / "source.pdf"

    def list_processing_records(self) -> tuple[PaperRecord, ...]:
        """Return queued/running jobs for the single local worker."""
        if not self.root.exists():
            return ()
        records = []
        for path in self.root.glob("*/record.json"):
            if not _HASH_RE.fullmatch(path.parent.name):
                continue
            record = PaperRecord.model_validate_json(path.read_bytes())
            if record.state == PaperState.PROCESSING:
                records.append(record)
        return tuple(records)

    def mark_failed(self, document_id: str, error: str) -> None:
        self._save_record(
            PaperRecord(document_id=document_id, state=PaperState.FAILED, error=error)
        )

    def _folder(self, document_id: str) -> Path:
        if not _HASH_RE.fullmatch(document_id):
            raise ValueError("Invalid document ID")
        return self.root / document_id

    def _save_record(self, record: PaperRecord) -> None:
        self._atomic_write(
            self._folder(record.document_id) / "record.json",
            record.model_dump_json(indent=2).encode(),
        )

    def _save_graph_record(self, record: GraphRecord) -> None:
        self._atomic_write(
            self._folder(record.document_id) / "graph-record.json",
            record.model_dump_json(indent=2).encode(),
        )

    def _graph_version_path(self, document_id: str, version: int) -> Path:
        if version < 1:
            raise ValueError("Graph version must be positive")
        return self._folder(document_id) / "graphs" / f"{version:06d}.json"

    def _graph_review_path(self, document_id: str, version: int) -> Path:
        path = self._graph_version_path(document_id, version)
        return path.with_name(f"{path.stem}-reviews.json")

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
