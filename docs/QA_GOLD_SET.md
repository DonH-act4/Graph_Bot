# Phase 0: manually reviewed QA and insufficient-evidence examples

`data/annotations/phase0_qa_gold.json` is a **small answer key**, not a chatbot or a model result.
Each case states which exact PDF versions a future assistant is allowed to use, whether an answer is
supported, a reference answer when it is, and the source-located evidence blocks. The cases use
Chinese questions to exercise the intended Chinese-question/English-paper workflow.

## What was annotated

| Case | Expected outcome | Evidence location |
| --- | --- | --- |
| BERT uses which Transformer component? | Transformer encoder | BERT PDF p. 3 |
| Transformer big's WMT 2014 English-German BLEU? | 28.4 BLEU | Transformer PDF p. 8 |
| BERT's pre-training objective for bidirectional context? | Masked language model (MLM) | BERT PDF pp. 1–2 |
| How are the two architectures related? | BERT is based on the Transformer encoder, not necessarily the complete encoder-decoder | BERT PDF p. 3 and Transformer PDF p. 3 |
| BERT's WMT 2014 English-German BLEU, using only the BERT paper? | Insufficient evidence | No supporting answer citation in the selected PDF |
| Did BERT beat Transformer big's 28.4 BLEU on the same task? | Insufficient evidence | Transformer has a score; selected BERT evidence has no comparable same-task score |

`insufficient` means **not established by these selected PDFs**, not “false in the world.” The
comparison case deliberately tests a common failure: taking a documented Transformer result and
an architectural link to BERT as proof of an unsupported performance comparison.

## Contract and validation

`src/evidencegraph/qa_contract.py` defines `QAGoldCase` and `QAGoldSet` with Pydantic. An
`answerable` case must have both a nonblank reference answer and at least one supporting citation.
An `insufficient` case must have neither. Duplicate case IDs, missing PDF versions, invented block
IDs, and citations outside the selected PDFs are rejected. A citation is
`(source_sha256, block_id)`; the parsed block supplies its page and bounding box.

From the Windows repository root in PowerShell, after reproducing the two parsed JSON files as
described in `data/samples/README.md`:

```powershell
$env:PYTHONPATH = "src"
uv run --no-sync python scripts\validate_qa_annotations.py `
  data\parsed\attention-is-all-you-need.json `
  data\parsed\bert.json
uv run --no-sync pytest -q tests\evidencegraph\test_qa_contract.py
```

The script is read-only, prints the answerability decision and resolved PDF pages, and fails on a
stale or fabricated citation. It explicitly uses UTF-8 for Windows command-line output.

## What this does not test yet

- The validator checks **reference integrity**, not whether the cited sentence actually entails
  the answer. The six decisions were made by reading the passages; future model outputs need
  separate semantic review.
- The reference answer is a guide to meaning, not a required exact string. “MLM” and “masked
  language model” can both be correct. Future scoring should separate answerability, factual
  correctness, citation validity, and citation support instead of using string equality alone.
- There is no Gemini call, retrieval, answer generation, API, or UI in this slice. A future assistant
  must be restricted to `selected_documents` in code, not only instructed by a prompt.
- Six questions over two PDFs are an initial regression set, not evidence of general accuracy.
  Add the roadmap's third paper and harder counterexamples before declaring phase 0 accepted.
