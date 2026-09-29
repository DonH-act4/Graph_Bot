# Phase 1 RoBERTa candidate-graph review

Date: 2026-09-27

This record covers the first real automatic graph extraction. It is evaluation evidence, not a
claim that phase 1 or the candidate relations are accepted.

## Reproducible input and model

- PDF: RoBERTa arXiv v1, SHA-256
  `76a3872d244793563a5b000b818cbaf0ca8972ab1145d32be474c05b2a8f3070`.
- Parsed artifact: schema version 2, 13 pages, 204 blocks.
- Extractor: `google-gemini`, model `gemini-3.5-flash-lite`.
- Rendered prompt: 79,863 characters; all 204 blocks were included under the 120,000-character
  limit.
- Output: 8 nodes, 8 candidate relations, 16 evidence references resolving to 11 unique blocks.

## Attempts

1. The first request started at 11:29:03 +0800 and failed in roughly 50 seconds because the model
   response did not pass the local Pydantic contract. No `graph.json` was saved and the parsed PDF
   remained ready.
2. The retry started at 11:31:20 +0800 and was ready by 11:32:04 +0800, roughly 44 seconds later.
   The validated artifact was written atomically.

The first failure motivated bounded field-level validation diagnostics. Diagnostics exclude model
values and retain only up to three schema locations, error types, and messages.

## Extracted nodes

The graph contains the paper, the RoBERTa method, three pretraining datasets (CC-NEWS,
OPENWEBTEXT, STORIES), and three evaluation datasets or benchmarks (GLUE, SQuAD, RACE). Every node
has one valid source block.

## Relation review

| Candidate relation | Evidence assessment | Review conclusion |
| --- | --- | --- |
| paper `introduces` RoBERTa | The cited abstract block names the improved recipe and RoBERTa. | Strong direct support; remain candidate until a review action exists. |
| paper `introduces` CC-NEWS | The cited contribution block calls CC-NEWS novel and says it is used. | Plausible, but `introduces` versus `uses/collects` is semantically ambiguous. |
| RoBERTa `uses` CC-NEWS | The relation block refers to additional Section 3.2 datasets without naming this endpoint. | Contextually plausible but not self-contained evidence. |
| RoBERTa `uses` OPENWEBTEXT | Same contextual relation block; the node block names OPENWEBTEXT separately. | Needs both blocks, or a more directly naming relation block. |
| RoBERTa `uses` STORIES | Same contextual relation block; the node block names STORIES separately. | Needs both blocks, or a more directly naming relation block. |
| RoBERTa `evaluated_on` GLUE | The cited abstract block directly names GLUE and reports results. | Strong support. |
| RoBERTa `evaluated_on` SQuAD | The cited abstract block directly names SQuAD and reports results. | Strong support. |
| RoBERTa `evaluated_on` RACE | The cited abstract block directly names RACE and reports results. | Strong support. |

All 16 references passed structural closure: exact PDF hash, real block ID, and valid parsed
artifact. Structural closure did not guarantee that one block was semantically sufficient for a
relation. The prompt now requires relation evidence to identify both endpoints and to include an
additional block when a cited block uses contextual phrases such as references to datasets
described elsewhere. This prompt revision has not yet been measured in a rebuilt graph.

## Remaining limitations

- The original v1 and v2 predate usage persistence, so their token and cost metadata cannot be
  recovered. Version 3 persists provider-reported usage and a clearly labelled Paid Standard cost
  reference.
- Versioned rebuild support now preserves immutable graph versions and isolates their reviews.
  The original real artifact is v1. An explicitly authorized Gemini rebuild produced v2 with the
  same eight entities and seven rather than eight candidate relations; both versions remain
  readable.
- Streamlit now renders a typed directed graph and lets a reviewer inspect exact evidence before
  accepting or rejecting a relation. The model artifact remains unchanged and the latest decision
  is stored separately in `graph-reviews.json`.
- Review persistence currently stores only the latest decision per relation. It has no reviewer
  identity, note, or append-only decision history, and it cannot edit a node or relation.
- The graph is evidence-selectable but not a full drag/filter interaction surface or PDF
  highlighter. A narrow viewport may require collapsing the sidebar or using Graphviz fullscreen.
- One paper and two stochastic attempts are not enough to estimate extraction accuracy or
  reliability.

## Version 2 comparison

The v2 rebuild used the revised endpoint-evidence instruction with the same parsed PDF and
`gemini-3.5-flash-lite` model.

- The ambiguous `paper introduces CC-NEWS` candidate was removed.
- `RoBERTa uses CC-NEWS` now cites the contribution block that directly says the work uses the
  novel CCNEWS dataset.
- OPENWEBTEXT and STORIES relations each cite both the block saying the best RoBERTa model is
  trained over all five Section 3.2 datasets and the block naming the specific dataset.
- GLUE, SQuAD, and RACE relations cite a block that directly names RoBERTa and all three
  benchmarks.

This is a positive one-document result, not an accuracy estimate. Token usage and a Paid Standard
cost reference are now persisted for future graph versions. The v1 and v2 API responses predate
that feature, so their usage cannot be recovered after the fact and remains unavailable.

## Version 3 usage and reliability evidence

An explicitly authorized v3 rebuild completed on 2026-09-28 with 9 nodes and 8 candidate
relations. It adds `BOOKCORPUS plus English WIKIPEDIA`; all eight relations contain evidence and
the artifact passed exact-hash/block validation. The immutable v1 and v2 artifacts remain
readable, and `current_version` changed to 3 only after the valid artifact was saved.

Provider-reported usage persisted with v3:

- input: 24,429 tokens;
- output: 4,725 tokens;
- thinking: 0 tokens;
- total: 29,154 tokens;
- Paid Standard reference: $0.0191412 USD using the 2026-09-28 pricing snapshot.

The user confirmed that billing is not linked. Therefore this reference is not an actual charge;
Free Tier usage costs $0 while the project remains within Google's limits.

Several rejected attempts are also acceptance evidence for the safety boundary. Truncated JSON,
an incompatible complex-schema request, and candidate relations without evidence all left v2
readable and did not create a partial v3. The final adapter uses a 16,384-token output budget,
requires every relation to emit an `evidence` field, normalizes provider API errors without
persisting raw provider content, and locally enforces at most 25 nodes and 40 relations.

The local Streamlit page rendered the v3 Graphviz network, evidence inspector, usage totals, and
cost reference. The upload workflow now writes `document_id` to the URL, and a fresh session
validates and restores that exact hash without uploading, parsing, or generating again. A real
browser check opened the RoBERTa URL and directly recovered v3, its three stored versions, and all
204 evidence blocks. Graph-version selection itself is not yet encoded in the URL; restored links
open the current version.
