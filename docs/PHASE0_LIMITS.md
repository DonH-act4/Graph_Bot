# Phase 0 product-limit observations (macOS, 2026-09-22)

These are measurements for three fixed, born-digital English AI papers, not upload limits or
latency guarantees. The machine ran macOS 26.6.2 on arm64. Docling 2.126.0, the project's locked
environment, disabled OCR, cached layout weights, and one PDF per process were used. No LLM API
or API key was involved.

## Measurement method and results

For each PDF from `data/samples/README.md`, run from the repository root:

```bash
PYTHONPATH=src /usr/bin/time -l .venv/bin/python scripts/inspect_pdf_provenance.py \
  data/samples/attention-is-all-you-need.pdf
```

Replace the last path with `bert.pdf` or `roberta.pdf` for the other two runs. The parser's
`elapsed_seconds` covers `parse_pdf` only; `real` also includes interpreter/model setup and JSON writing.
`maximum resident set size` is the macOS process's peak resident memory in bytes. These runs
overwrite only reproducible, Git-ignored `data/parsed/*.json` outputs.

| PDF | File bytes | Located pages | Blocks | Parser seconds | Wall seconds | Peak RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | 2,215,244 | 15/15 | 154 | 10.264 | 13.29 | 1,515,929,600 |
| BERT | 775,166 | 16/16 | 254 | 11.284 | 14.40 | 1,489,682,432 |
| RoBERTa | 209,675 | 13/13 | 206 | 12.035 | 15.19 | 1,377,435,648 |

All three output files have unique block IDs, full page coverage, and no page or character-span
range violations. Transformer and RoBERTa yielded two and one fewer blocks respectively than the
prior Windows run, despite matching PDF hashes and Docling version. The cited text stayed intact;
the parser's internal item numbers shifted. The version 2 content-based ID scheme and migrated
annotations now validate against the macOS outputs. A different text split or page assignment
still requires review, as does reordered identical text on one page.

## Current input and failure boundaries

The parser accepts a nonempty path ending in `.pdf` and requires Docling to return source-located
text. OCR is disabled. Unit tests cover unsupported suffixes, missing and empty files, conversion
failure, and no located text. A repeatable real-input check creates temporary malformed, encrypted,
blank, and image-only scanned PDFs, then requires a `PdfParseError` for each:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_pdf_failure_inputs.py
```

Malformed and encrypted files produce a short conversion-failure message; blank and scanned pages
produce "no source-located text". The generated scanned page contains an image and no extractable
text layer. Docling itself still logs the input path locally. There is no EvidenceGraph upload route
or UI, so no HTTP file-size or page-count limit or user-facing error message has been verified.
Model usage is zero for this parsing and annotation workflow; future extraction and QA must record request tokens,
retries, latency, and cost separately if an API model is introduced.

Phase 0 accepts only the three exact sample PDF hashes for internal evaluation. It makes no general
upload-size, page-count, latency, or concurrent-user promise. Before phase 1 exposes an upload route,
measure cold-cache and concurrent behavior and choose explicit size, page, time, and memory caps
from those results. The upload route must turn parser errors into concise user-facing messages and
avoid logging sensitive paths. These are phase 1 release gates, not claims verified here.
