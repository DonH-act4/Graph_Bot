"""Check the manually reviewed graph against reproducible parsed PDF blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

from evidencegraph.graph_contract import GraphAnnotation
from evidencegraph.ingestion import ParsedDocument


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/phase0_gold.json"),
    )
    parser.add_argument(
        "parsed_pdfs",
        nargs="+",
        type=Path,
        help="Parser-neutral JSON files produced by inspect_pdf_provenance.py",
    )
    args = parser.parse_args()

    annotation = GraphAnnotation.model_validate_json(args.annotations.read_text(encoding="utf-8"))
    documents = tuple(
        ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.parsed_pdfs
    )
    annotation.validate_against(documents)
    document_index = {document.source_sha256: document for document in documents}
    block_index = {
        (document.source_sha256, block.block_id): block
        for document in documents
        for block in document.blocks
    }

    print(f"Validated {len(annotation.nodes)} nodes and {len(annotation.relations)} relations")
    for relation in annotation.relations:
        print(
            f"{relation.status.value.upper()}: {relation.source_node_id} "
            f"--{relation.relation_type.value}--> {relation.target_node_id}"
        )
        for ref in relation.evidence:
            document = document_index[ref.source_sha256]
            block = block_index[(ref.source_sha256, ref.block_id)]
            pages = sorted({location.page_number for location in block.locations})
            print(f"  {document.source_filename}, PDF page {pages}, {ref.block_id}")


if __name__ == "__main__":
    main()
