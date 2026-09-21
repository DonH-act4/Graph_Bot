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
but has not yet been added to the manual graph/QA gold sets.

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
reference, and one or more page/bounding-box/character-span locations.

The first run can download Docling's layout model from Hugging Face. It does not call Gemini or any
other generative-model API. OCR is deliberately disabled for this born-digital baseline, so scanned
PDFs are outside the current scope.
