"""EvidenceGraph domain with lazy access to heavyweight parsing functions."""

from typing import Any

from evidencegraph.models import (
    BoundingBox,
    ParsedBlock,
    ParsedDocument,
    PdfParseError,
    SourceLocation,
)

__all__ = [
    "BoundingBox",
    "ParsedBlock",
    "ParsedDocument",
    "PdfParseError",
    "SourceLocation",
    "build_pdf_converter",
    "parse_pdf",
]


def __getattr__(name: str) -> Any:
    if name in {"build_pdf_converter", "parse_pdf"}:
        from evidencegraph import ingestion

        return getattr(ingestion, name)
    raise AttributeError(name)
