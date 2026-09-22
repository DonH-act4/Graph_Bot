from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import ConversionStatus
from docling.exceptions import ConversionError
from pydantic import ValidationError

from evidencegraph.ingestion import ParsedDocument, PdfParseError, parse_pdf


class FakeConverter:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result

    def convert(self, source: Path) -> SimpleNamespace:
        return self.result


class RaisingConverter:
    def convert(self, source: Path) -> None:
        raise ConversionError("Docling could not load the PDF")


def _pdf(tmp_path: Path, content: bytes = b"%PDF-1.5 test") -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)
    return path


def _result(
    *,
    status: ConversionStatus = ConversionStatus.SUCCESS,
    located: bool = True,
    include_table: bool = False,
    source_ref: str = "#/texts/1",
    duplicate_text: bool = False,
) -> SimpleNamespace:
    bbox = SimpleNamespace(l=1.0, t=20.0, r=30.0, b=2.0, coord_origin="BOTTOMLEFT")
    provenance = SimpleNamespace(page_no=2, bbox=bbox, charspan=(0, 12))
    item = SimpleNamespace(
        text="  evidence text  ",
        prov=[provenance] if located else [],
        self_ref=source_ref,
        label=SimpleNamespace(value="text"),
    )
    items = [(item, 1)]
    if duplicate_text:
        items.append((SimpleNamespace(**{**vars(item), "self_ref": "#/texts/2"}), 1))
    if include_table:
        table = SimpleNamespace(
            text="",
            prov=[provenance],
            self_ref="#/tables/0",
            label=SimpleNamespace(value="table"),
            export_to_markdown=lambda *, doc: "| Dataset | Score |\n| --- | --- |\n| GLUE | 80.5 |",
        )
        items.append((table, 1))
    document = SimpleNamespace(iterate_items=lambda: iter(items))
    return SimpleNamespace(status=status, document=document, pages=[object(), object()])


def test_parse_pdf_returns_parser_neutral_provenance(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)

    parsed = parse_pdf(pdf, converter=FakeConverter(_result()))

    assert parsed.source_filename == "paper.pdf"
    assert parsed.page_count == 2
    assert len(parsed.source_sha256) == 64
    assert parsed.parser_name == "docling"
    assert parsed.blocks[0].text == "evidence text"
    assert parsed.blocks[0].source_ref == "#/texts/1"
    assert parsed.blocks[0].label == "text"
    assert parsed.blocks[0].locations[0].page_number == 2
    assert parsed.blocks[0].locations[0].bounding_box.coordinate_origin == "BOTTOMLEFT"


def test_block_id_is_deterministic_for_same_input(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    converter = FakeConverter(_result())

    first = parse_pdf(pdf, converter=converter)
    second = parse_pdf(pdf, converter=converter)

    assert first.blocks[0].block_id == second.blocks[0].block_id


def test_block_id_survives_docling_item_index_shift(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)

    first = parse_pdf(pdf, converter=FakeConverter(_result(source_ref="#/texts/1")))
    shifted = parse_pdf(pdf, converter=FakeConverter(_result(source_ref="#/texts/3")))

    assert first.blocks[0].source_ref != shifted.blocks[0].source_ref
    assert first.blocks[0].block_id == shifted.blocks[0].block_id


def test_repeated_text_on_same_page_gets_distinct_ids(tmp_path: Path) -> None:
    parsed = parse_pdf(_pdf(tmp_path), converter=FakeConverter(_result(duplicate_text=True)))

    assert parsed.blocks[0].text == parsed.blocks[1].text
    assert parsed.blocks[0].block_id != parsed.blocks[1].block_id


def test_old_parsed_document_schema_is_rejected(tmp_path: Path) -> None:
    parsed = parse_pdf(_pdf(tmp_path), converter=FakeConverter(_result()))
    old_document = parsed.model_dump()
    old_document["schema_version"] = "1"

    with pytest.raises(ValidationError, match="schema_version"):
        ParsedDocument.model_validate(old_document)


def test_parse_pdf_preserves_table_as_markdown_block(tmp_path: Path) -> None:
    parsed = parse_pdf(_pdf(tmp_path), converter=FakeConverter(_result(include_table=True)))

    table = parsed.blocks[1]
    assert table.label == "table"
    assert "| Dataset | Score |" in table.text
    assert table.source_ref == "#/tables/0"


def test_parse_pdf_rejects_non_pdf(tmp_path: Path) -> None:
    text_file = tmp_path / "paper.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a .pdf"):
        parse_pdf(text_file, converter=FakeConverter(_result()))


def test_parse_pdf_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "missing.pdf", converter=FakeConverter(_result()))


def test_parse_pdf_rejects_empty_file(tmp_path: Path) -> None:
    with pytest.raises(PdfParseError, match="empty"):
        parse_pdf(_pdf(tmp_path, b""), converter=FakeConverter(_result()))


def test_parse_pdf_rejects_failed_conversion(tmp_path: Path) -> None:
    failed = _result(status=ConversionStatus.FAILURE)

    with pytest.raises(PdfParseError, match="did not succeed"):
        parse_pdf(_pdf(tmp_path), converter=FakeConverter(failed))


def test_docling_conversion_error_has_safe_message_and_original_cause(tmp_path: Path) -> None:
    with pytest.raises(PdfParseError, match="PDF conversion failed") as caught:
        parse_pdf(_pdf(tmp_path), converter=RaisingConverter())

    assert isinstance(caught.value.__cause__, ConversionError)
    assert "Docling could not load" not in str(caught.value)


def test_parse_pdf_rejects_text_without_provenance(tmp_path: Path) -> None:
    unlocated = _result(located=False)

    with pytest.raises(PdfParseError, match="no source-located text"):
        parse_pdf(_pdf(tmp_path), converter=FakeConverter(unlocated))
