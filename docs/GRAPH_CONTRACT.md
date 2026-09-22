# Phase 0: evidence-linked graph contract

This is a small **manually reviewed reference graph**, not automatic extraction or a user-facing
graph. It lets us test later Gemini output against a saved answer key without mistaking a valid
block ID for a semantically correct claim.

## Data flow and rules

1. `inspect_pdf_provenance.py` parses each exact PDF version into `ParsedDocument` JSON.
2. `phase0_gold.json` names graph nodes and relations and cites evidence using
   `(source_sha256, block_id)`. The SHA-256 prevents citing a block from another PDF revision.
3. `GraphAnnotation.validate_against(...)` checks that every node and cited block exists, node
   evidence belongs to its paper, and a `direct` cross-paper relation cites both endpoint papers.
4. `verified_relations` returns only `direct` relations. A `candidate` may later be shown with a
   distinct review style; `unconfirmed` is kept for review/evaluation and is never a verified fact.

The validator does **not** prove semantic entailment. A real block can still be irrelevant or
contradict a proposed edge. Human review or a separate evaluation must decide that.

| Status | Meaning | Evidence rule | Presentation rule |
| --- | --- | --- | --- |
| `direct` | Human reviewer judged the cited text to state the relation | At least one citation; for cross-paper edges, citations from both papers | May be labelled verified |
| `candidate` | Plausible match or inference needing review | At least one citation explaining the suggestion | Must be visually distinct from verified facts |
| `unconfirmed` | Not supported by the available evidence set | May have no citation | Do not show as a verified graph edge |

The current fixture uses 6 nodes and 6 relation records. Five are directly supported examples:

| Relation | Primary supporting PDF page | Why it was labelled direct |
| --- | --- | --- |
| Transformer paper `introduces` Transformer architecture | Transformer p. 1 | Abstract says it proposes the new architecture |
| Transformer paper `evaluated_on` WMT 2014 English-German | Transformer p. 8 | Results paragraph names the task and model result |
| BERT paper `uses` Transformer encoder | BERT p. 3; Transformer p. 3 | BERT describes its encoder as based on Vaswani et al.; the earlier paper describes its encoder |
| BERT paper `builds_on` Transformer paper | BERT p. 3; Transformer p. 1 | BERT explicitly names Vaswani et al. as the architectural source |
| RoBERTa paper `builds_on` BERT paper | RoBERTa p. 1; BERT pp. 1–2 | RoBERTa describes a replication study and an improved recipe for BERT pretraining; this is methodological lineage, not an unconditional performance ranking |

The sixth record, BERT `evaluated_on` WMT 2014 English-German, is deliberately `unconfirmed`:
sharing a Transformer lineage does not prove sharing an evaluation dataset. It is a negative
example for future extraction tests, not a finding about every possible BERT experiment.

## Reproduce on Windows

First download the three exact PDFs using `data/samples/README.md`, then run from the repository
root in PowerShell:

```powershell
uv sync
$env:PYTHONPATH = "src"
uv run --no-sync python scripts\inspect_pdf_provenance.py `
  data\samples\attention-is-all-you-need.pdf `
  data\samples\bert.pdf `
  data\samples\roberta.pdf
uv run --no-sync python scripts\validate_graph_annotations.py `
  data\parsed\attention-is-all-you-need.json `
  data\parsed\bert.json `
  data\parsed\roberta.json
uv run --no-sync pytest -q tests\evidencegraph\test_graph_contract.py
```

The validation command is read-only. It prints every relation and its resolved PDF page. It fails
if the PDF hash or any evidence ID no longer matches the annotated corpus. The parsed JSON files
are reproducible, ignored local artifacts; the small gold annotation JSON is versioned source.
Schema version `2` block IDs omit Docling's internal item number: the same cited text had shifted
by two internal positions between Windows and macOS despite identical PDFs and Docling 2.126.0.
IDs now include PDF hash, parser version, page, label, text, and an occurrence count for repeated
same-page text. If the parser changes the actual text or page assignment, review and reannotate;
the ID scheme does not make those semantic changes invisible.
If identical text appears more than once on the same page, its occurrence number also depends on
reading order; a change in that order needs manual review of the affected citations.

## Current limitations and next step

- IDs in the hand-written fixture are stable **within this dataset**, but production node and
  relation identity/rebuild rules have not yet been implemented.
- There is no graph database, extraction model, API, or webpage in this slice.
- `candidate` behavior is tested by the schema but has no real annotated example yet.
- Eight question/answer and insufficient-evidence cases now live in
  `data/annotations/phase0_qa_gold.json`; see `docs/QA_GOLD_SET.md`.
- The next bounded implementation is the phase 1 PDF upload and processing path. Automatic
  extraction can then be evaluated against these records: a model should receive bounded evidence
  blocks and return schema-checked candidate nodes/relations, with wrong, missing, and unsupported
  edges scored separately. No API key belongs in Git.
