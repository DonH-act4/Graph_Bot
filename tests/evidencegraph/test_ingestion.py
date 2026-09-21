from pathlib import Path
from types import SimpleNamespace

import pytest
from docling.datamodel.base_models import ConversionStatus

from evidencegraph.ingestion import PdfParseError, parse_pdf


class FakeConverter:
    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result

    def convert(self, source: Path) -> SimpleNamespace:
        return self.result


def _pdf(tmp_path: Path, content: bytes = b"%PDF-1.5 test") -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)
    return path


def _result(
    *,
    status: ConversionStatus = ConversionStatus.SUCCESS,
    located: bool = True,
    include_table: bool = False,
) -> SimpleNamespace:
    bbox = SimpleNamespace(l=1.0, t=20.0, r=30.0, b=2.0, coord_origin="BOTTOMLEFT")
    provenance = SimpleNamespace(page_no=2, bbox=bbox, charspan=(0, 12))
    item = SimpleNamespace(
        text="  evidence text  ",
        prov=[provenance] if located else [],
        self_ref="#/texts/1",
        label=SimpleNamespace(value="text"),
    )
    items = [(item, 1)]
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


def test_parse_pdf_rejects_text_without_provenance(tmp_path: Path) -> None:
    unlocated = _result(located=False)

    with pytest.raises(PdfParseError, match="no source-located text"):
        parse_pdf(_pdf(tmp_path), converter=FakeConverter(unlocated))
