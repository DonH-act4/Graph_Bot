# Phase 0 PDF samples

The PDF binaries are local evaluation inputs and are ignored by Git. Download them from the
authoritative arXiv URLs below rather than redistributing copies from this repository.

| File | Paper | Source | SHA-256 | Pages |
| --- | --- | --- | --- | ---: |
| `attention-is-all-you-need.pdf` | Attention Is All You Need | <https://arxiv.org/pdf/1706.03762> | `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697` | 15 |
| `bert.pdf` | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | <https://arxiv.org/pdf/1810.04805> | `5692a5514787a8c6727b4ff3b726a3385798bc68e12138d1d4af83947e2acf6e` | 16 |
| `roberta.pdf` | RoBERTa: A Robustly Optimized BERT Pretraining Approach | <https://arxiv.org/pdf/1907.11692v1> | `76a3872d244793563a5b000b818cbaf0ca8972ab1145d32be474c05b2a8f3070` | 13 |

The first two PDFs were selected because they are born-digital English AI papers with multi-column
text, tables, figures, and formulas. BERT builds on the Transformer encoder. RoBERTa was added as
the third sample because its paper explicitly studies BERT pretraining, offering another concrete
cross-paper relationship to evaluate later. RoBERTa has been parsed and source-location checked,
and now has a manually reviewed BERT relation and two QA cases in the gold sets.

From PowerShell in the repository root:

```powershell
New-Item -ItemType Directory -Force data\samples | Out-Null
Invoke-WebRequest https://arxiv.org/pdf/1706.03762 `
  -OutFile data\samples\attention-is-all-you-need.pdf
Invoke-WebRequest https://arxiv.org/pdf/1810.04805 `
  -OutFile data\samples\bert.pdf
Invoke-WebRequest https://arxiv.org/pdf/1907.11692v1 `
  -OutFile data\samples\roberta.pdf
Get-FileHash -Algorithm SHA256 data\samples\*.pdf
```

The hashes record the exact inputs used for the baseline. If arXiv later serves a different revision,
treat it as a new document version rather than silently accepting the changed hash.

## Reproduce the provenance baseline

After `uv sync`, run the parser from PowerShell in the repository root:

```powershell
$env:PYTHONPATH = "src"
uv run --no-sync python scripts\inspect_pdf_provenance.py `
  data\samples\attention-is-all-you-need.pdf `
  data\samples\bert.pdf `
  data\samples\roberta.pdf
```

The command writes parser-neutral JSON to `data\parsed`, which is ignored by Git because it is a
reproducible local artifact. Each text or table block contains a stable block ID, its Docling source
reference, and one or more page/bounding-box/character-span locations. Parsed JSON schema version
`2` uses PDF hash, parser version, page, label, text, and same-page occurrence for block IDs. Reparse
old schema version `1` outputs before validating the current gold annotations; their IDs included
Docling's platform-sensitive internal item number.

## Reproduce on macOS

From the repository root, download the same three URLs above into `data/samples`, then compare each
SHA-256 with this table using `shasum -a 256 data/samples/*.pdf`. With the locked environment present:

```bash
PYTHONPATH=src uv run --no-sync python scripts/inspect_pdf_provenance.py \
  data/samples/attention-is-all-you-need.pdf \
  data/samples/bert.pdf \
  data/samples/roberta.pdf
PYTHONPATH=src uv run --no-sync python scripts/validate_graph_annotations.py data/parsed/*.json
PYTHONPATH=src uv run --no-sync python scripts/validate_qa_annotations.py data/parsed/*.json
```

The glob should include only the three intended phase 0 parsed PDFs. See `docs/PHASE0_LIMITS.md`
for measured macOS resource use and the remaining product-limit checks.

The first run can download Docling's layout model from Hugging Face. It does not call Gemini or any
other generative-model API. OCR is deliberately disabled for this born-digital baseline, so scanned
PDFs are outside the current scope.
