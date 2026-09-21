import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidencegraph.graph_contract import EvidenceRef
from evidencegraph.ingestion import BoundingBox, ParsedBlock, ParsedDocument, SourceLocation
from evidencegraph.qa_contract import Answerability, QAGoldCase, QAGoldSet

PDF_HASH = "a" * 64
OTHER_HASH = "b" * 64
BLOCK_ID = "blk_" + "1" * 24


def _ref(source_hash: str = PDF_HASH, block_id: str = BLOCK_ID) -> EvidenceRef:
    return EvidenceRef(source_sha256=source_hash, block_id=block_id)


def _case(**changes: object) -> QAGoldCase:
    values: dict[str, object] = {
        "case_id": "fact",
        "question": "Which encoder is used?",
        "selected_documents": (PDF_HASH,),
        "answerability": Answerability.ANSWERABLE,
        "reference_answer": "Transformer encoder",
        "supporting_evidence": (_ref(),),
        "rationale": "The paper states this directly.",
    }
    values.update(changes)
    return QAGoldCase.model_validate(values)


def _document(source_hash: str = PDF_HASH) -> ParsedDocument:
    return ParsedDocument(
        source_filename="paper.pdf",
        source_sha256=source_hash,
        parser_version="2.126.0",
        page_count=1,
        blocks=(
            ParsedBlock(
                block_id=BLOCK_ID,
                source_ref="#/texts/0",
                label="text",
                text="The paper uses a Transformer encoder.",
                locations=(
                    SourceLocation(
                        page_number=1,
                        bounding_box=BoundingBox(
                            left=1,
                            top=2,
                            right=3,
                            bottom=0,
                            coordinate_origin="BOTTOMLEFT",
                        ),
                        character_start=0,
                        character_end=37,
                    ),
                ),
            ),
        ),
    )


def test_answerable_case_resolves_citation_in_selected_pdf() -> None:
    QAGoldSet(cases=(_case(),)).validate_against((_document(),))


def test_insufficient_case_has_no_invented_answer_or_support() -> None:
    case = _case(
        answerability=Answerability.INSUFFICIENT,
        reference_answer=None,
        supporting_evidence=(),
    )
    QAGoldSet(cases=(case,)).validate_against((_document(),))


def test_answerable_case_requires_answer_and_evidence() -> None:
    with pytest.raises(ValidationError, match="reference answer"):
        _case(reference_answer=None)
    with pytest.raises(ValidationError, match="supporting evidence"):
        _case(supporting_evidence=())


def test_insufficient_case_cannot_claim_an_answer() -> None:
    with pytest.raises(ValidationError, match="cannot claim an answer"):
        _case(answerability=Answerability.INSUFFICIENT, supporting_evidence=())


def test_citation_outside_selected_pdf_is_rejected() -> None:
    case = _case(supporting_evidence=(_ref(OTHER_HASH),))
    with pytest.raises(ValueError, match="outside selected PDFs"):
        QAGoldSet(cases=(case,)).validate_against((_document(), _document(OTHER_HASH)))


def test_unknown_citation_block_is_rejected() -> None:
    case = _case(supporting_evidence=(_ref(block_id="blk_" + "2" * 24),))
    with pytest.raises(ValueError, match="unknown evidence block"):
        QAGoldSet(cases=(case,)).validate_against((_document(),))


def test_unknown_selected_pdf_is_rejected() -> None:
    case = _case(selected_documents=(OTHER_HASH,), supporting_evidence=(_ref(OTHER_HASH),))
    with pytest.raises(ValueError, match="unknown selected PDF"):
        QAGoldSet(cases=(case,)).validate_against((_document(),))


def test_duplicate_case_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate QA case ID"):
        QAGoldSet(cases=(_case(), _case()))


def test_real_gold_fixture_shape() -> None:
    fixture = Path(__file__).resolve().parents[2] / "data" / "annotations" / "phase0_qa_gold.json"
    gold_set = QAGoldSet.model_validate_json(fixture.read_text(encoding="utf-8"))
    assert sum(case.answerability is Answerability.ANSWERABLE for case in gold_set.cases) >= 3
    assert sum(case.answerability is Answerability.INSUFFICIENT for case in gold_set.cases) >= 2


def test_cli_prints_chinese_when_windows_stdout_starts_as_cp1252(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    annotations = tmp_path / "qa.json"
    parsed = tmp_path / "paper.json"
    annotations.write_text(
        QAGoldSet(cases=(_case(question="使用了哪个编码器？"),)).model_dump_json(),
        encoding="utf-8",
    )
    parsed.write_text(_document().model_dump_json(), encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "cp1252"
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "validate_qa_annotations.py"),
            "--annotations",
            str(annotations),
            str(parsed),
        ],
        env=environment,
        capture_output=True,
        check=True,
    )
    assert "使用了哪个编码器？" in result.stdout.decode("utf-8")
