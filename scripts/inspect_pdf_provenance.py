"""Parse PDF samples and write reproducible provenance summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from evidencegraph import ParsedDocument, build_pdf_converter, parse_pdf


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converter = build_pdf_converter()
    for pdf_path in args.pdfs:
        started_at = perf_counter()
        parsed = parse_pdf(pdf_path, converter=converter)
        elapsed_seconds = perf_counter() - started_at
        output_path = args.output_dir / f"{pdf_path.stem}.json"
        output_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        print(json.dumps(_summary(parsed, output_path, elapsed_seconds), indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect page and bounding-box provenance returned by Docling."
    )
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory for full parser-neutral JSON output (default: data/parsed).",
    )
    return parser.parse_args()


def _summary(
    parsed: ParsedDocument, output_path: Path, elapsed_seconds: float
) -> dict[str, object]:
    labels = Counter(block.label for block in parsed.blocks)
    located_pages = sorted(
        {location.page_number for block in parsed.blocks for location in block.locations}
    )
    return {
        "source_filename": parsed.source_filename,
        "source_sha256": parsed.source_sha256,
        "parser": f"{parsed.parser_name} {parsed.parser_version}",
        "page_count": parsed.page_count,
        "located_page_count": len(located_pages),
        "block_count": len(parsed.blocks),
        "labels": dict(sorted(labels.items())),
        "output_path": str(output_path),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


if __name__ == "__main__":
    main()
