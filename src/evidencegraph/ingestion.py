"""Parse born-digital PDF papers into provenance-bearing text blocks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.exceptions import ConversionError

from evidencegraph.models import (
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    PdfParseError,
    SourceLocation,
)


class PdfConverter(Protocol):
    """Small boundary that keeps Docling replaceable and tests lightweight."""

    def convert(self, source: Path) -> Any: ...


def build_pdf_converter() -> DocumentConverter:
    """Build the phase 0 parser for born-digital PDFs.

    OCR is intentionally disabled because the MVP currently rejects scanned PDFs. Table structure
    detection remains enabled so the samples exercise more than plain paragraphs.
    """
    pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def parse_pdf(pdf_path: str | Path, *, converter: PdfConverter | None = None) -> ParsedDocument:
    """Parse a PDF into source-located blocks or raise a specific validation error."""
    source = Path(pdf_path)
    _validate_pdf_path(source)
    source_hash = _sha256(source)
    parser_version = version("docling")
    active_converter = converter or build_pdf_converter()
    try:
        result = active_converter.convert(source)
    except ConversionError as exc:
        raise PdfParseError("PDF conversion failed; file may be damaged or encrypted") from exc

    if result.status is not ConversionStatus.SUCCESS:
        raise PdfParseError(f"Docling conversion did not succeed: {result.status}")

    blocks = tuple(
        _iter_blocks(
            result.document,
            source_hash=source_hash,
            parser_version=parser_version,
        )
    )
    if not blocks:
        raise PdfParseError("PDF contains no source-located text blocks")

    page_count = len(result.pages)
    if page_count < 1:
        raise PdfParseError("PDF conversion returned no pages")

    return ParsedDocument(
        source_filename=source.name,
        source_sha256=source_hash,
        parser_version=parser_version,
        page_count=page_count,
        blocks=blocks,
    )


def _validate_pdf_path(source: Path) -> None:
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {source.name}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size == 0:
        raise PdfParseError("PDF file is empty")


def _sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_blocks(
    document: Any, *, source_hash: str, parser_version: str
) -> Iterable[ParsedBlock]:
    occurrences: dict[tuple[tuple[int, ...], str, str], int] = {}
    for item, _level in document.iterate_items():
        label_value = getattr(item, "label", "unknown")
        label = str(getattr(label_value, "value", label_value))
        text = _item_text(item, document, label=label)
        provenance = getattr(item, "prov", None) or ()
        source_ref = str(getattr(item, "self_ref", ""))
        if not text or not provenance or not source_ref:
            continue

        locations = tuple(_source_location(entry) for entry in provenance)
        pages = tuple(sorted({location.page_number for location in locations}))
        content_key = (pages, label, text)
        occurrence = occurrences.get(content_key, 0)
        occurrences[content_key] = occurrence + 1
        identity = "\0".join(
            (source_hash, parser_version, ",".join(map(str, pages)), label, text, str(occurrence))
        )
        block_id = f"blk_{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        yield ParsedBlock(
            block_id=block_id,
            source_ref=source_ref,
            label=label,
            text=text,
            locations=locations,
        )


def _item_text(item: Any, document: Any, *, label: str) -> str:
    if label == "table":
        return str(item.export_to_markdown(doc=document)).strip()
    return str(getattr(item, "text", "")).strip()


def _source_location(provenance: Any) -> SourceLocation:
    bbox = provenance.bbox
    char_start, char_end = provenance.charspan
    origin_value = getattr(bbox.coord_origin, "value", bbox.coord_origin)
    return SourceLocation(
        page_number=provenance.page_no,
        bounding_box=BoundingBox(
            left=bbox.l,
            top=bbox.t,
            right=bbox.r,
            bottom=bbox.b,
            coordinate_origin=str(origin_value),
        ),
        character_start=char_start,
        character_end=char_end,
    )
