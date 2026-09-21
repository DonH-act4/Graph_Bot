"""Validate saved QA cases against exact parsed PDF versions and print their citations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evidencegraph.ingestion import ParsedDocument
from evidencegraph.qa_contract import Answerability, QAGoldSet


def main() -> None:
    # Windows SSH sessions may default to cp1252, which cannot print Chinese questions.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations/phase0_qa_gold.json")
    )
    parser.add_argument(
        "parsed_pdfs",
        nargs="+",
        type=Path,
        help="Parser-neutral JSON files produced by inspect_pdf_provenance.py",
    )
    args = parser.parse_args()

    gold_set = QAGoldSet.model_validate_json(args.annotations.read_text(encoding="utf-8"))
    documents = tuple(
        ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.parsed_pdfs
    )
    gold_set.validate_against(documents)
    document_index = {document.source_sha256: document for document in documents}
    block_index = {
        (document.source_sha256, block.block_id): block
        for document in documents
        for block in document.blocks
    }

    answerable = sum(case.answerability is Answerability.ANSWERABLE for case in gold_set.cases)
    print(f"Validated {len(gold_set.cases)} QA cases: {answerable} answerable, {len(gold_set.cases) - answerable} insufficient")
    for case in gold_set.cases:
        print(f"{case.case_id} [{case.answerability.value}]: {case.question}")
        if case.reference_answer is not None:
            print(f"  Reference answer: {case.reference_answer}")
        for ref in case.supporting_evidence:
            document = document_index[ref.source_sha256]
            block = block_index[(ref.source_sha256, ref.block_id)]
            pages = sorted({location.page_number for location in block.locations})
            print(f"  Evidence: {document.source_filename}, PDF page {pages}, {ref.block_id}")


if __name__ == "__main__":
    main()
