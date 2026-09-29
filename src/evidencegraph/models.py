"""Lightweight parser-neutral evidence models safe to import in the API process."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PdfParseError(RuntimeError):
    """Raised when a PDF cannot produce usable, source-located text."""


class BoundingBox(BaseModel):
    """A box in PDF page coordinates."""

    model_config = ConfigDict(frozen=True)

    left: float
    top: float
    right: float
    bottom: float
    coordinate_origin: str


class SourceLocation(BaseModel):
    """Where a parsed block came from in the source PDF."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    bounding_box: BoundingBox
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)


class ParsedBlock(BaseModel):
    """One source-located text block from a specific parser run."""

    model_config = ConfigDict(frozen=True)

    block_id: str
    source_ref: str
    label: str
    text: str = Field(min_length=1)
    locations: tuple[SourceLocation, ...] = Field(min_length=1)


class ParsedDocument(BaseModel):
    """Parser-neutral output used by later evidence extraction."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["2"] = "2"
    source_filename: str
    source_sha256: str
    parser_name: str = "docling"
    parser_version: str
    page_count: int = Field(ge=1)
    blocks: tuple[ParsedBlock, ...] = Field(min_length=1)
