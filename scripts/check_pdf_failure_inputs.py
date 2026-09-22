"""Exercise real phase 0 PDF failure inputs without keeping sample binaries."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter

from evidencegraph.ingestion import PdfParseError, build_pdf_converter, parse_pdf


def main() -> None:
    with TemporaryDirectory(prefix="evidencegraph-phase0-") as directory:
        root = Path(directory)
        malformed = root / "malformed.pdf"
        malformed.write_bytes(b"%PDF-1.5\nnot a valid PDF")

        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        blank = root / "blank.pdf"
        writer.write(blank)
        writer.encrypt("phase0-test")
        encrypted = root / "encrypted.pdf"
        writer.write(encrypted)

        scanned = root / "scanned.pdf"
        image = Image.new("RGB", (600, 300), "white")
        ImageDraw.Draw(image).text((30, 130), "Scanned research paper text", fill="black")
        image.save(scanned, "PDF", resolution=100.0)
        scanned_page = PdfReader(scanned).pages[0]
        if scanned_page.extract_text() or not scanned_page.images:
            raise AssertionError("scanned fixture must have an image but no text layer")

        converter = build_pdf_converter()
        cases = (
            ("malformed", malformed, "PDF conversion failed"),
            ("blank", blank, "no source-located text"),
            ("encrypted", encrypted, "PDF conversion failed"),
            ("scanned", scanned, "no source-located text"),
        )
        for name, path, expected in cases:
            try:
                parse_pdf(path, converter=converter)
            except PdfParseError as error:
                if expected not in str(error):
                    raise AssertionError(f"{name}: unexpected error: {error}") from error
                print(f"{name}: {error}")
            else:
                raise AssertionError(f"{name}: unexpectedly parsed")


if __name__ == "__main__":
    main()
