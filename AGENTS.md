# AGENTS.md

## Project Overview

EvidenceGraph is an auditable document intelligence and research agent.

The application processes complex documents such as research papers, annual
reports, regulatory circulars, and technical reports. It extracts structured
information, preserves source provenance, supports hybrid and graph retrieval,
and generates evidence-backed answers.

The project is being developed incrementally as both a deployable portfolio
project and a learning project.

## Development Approach

- Work in small, testable vertical slices.
- Keep the application runnable after each completed change.
- Avoid large rewrites unless the existing design clearly prevents progress.
- Before making a substantial change, explain:
  - the problem being solved;
  - the relevant concept;
  - the files that will change;
  - the main design alternatives.
- After implementing a change, explain:
  - how the implementation works;
  - how to run and verify it;
  - known limitations and failure cases;
  - likely interview questions related to the change.
- Prefer simple implementations first, then improve them using measured results.
- Do not add a technology solely to make the stack appear more advanced.
- Before running an unfamiliar shell or Docker command, explain its purpose,
  whether it changes state, and the expected result. The user wants to learn
  each step and to understand what code was changed, not just receive outcomes.

## Development Environments

- This repository has separate macOS and Windows clones. The Windows clone is
  at `G:\Logs_ML\Graph_Bot` and is used for remote development and eventual
  deployment.
- SSH access from the Mac to the Windows machine over Tailscale was verified on
  2026-09-17. The SSH host is configured on the Mac; do not store its address,
  keys, passwords, or other connection secrets in this repository.
- A read-only check confirmed that the Windows clone contains `src`, `tests`,
  `docs`, and the other project files. Direct SSH can be used to inspect or edit
  that clone when the user asks; a Codex CLI installation on Windows is not
  required for direct SSH file operations.
- The Mac and Windows clones do not synchronize automatically. Before changing
  code, establish which clone is the target and inspect its branch, Git status,
  and existing changes. Preserve user work in both clones.
- Do not assume that editing the Windows clone authorizes starting services,
  restarting containers, deploying, or pushing changes. Agree on those actions
  separately, especially if a file watcher could restart a running service.

## Current Architecture

The existing repository is based on Agent Service Toolkit.

The intended architecture is:

- FastAPI for the backend API.
- LangGraph for bounded, tool-using workflows.
- Pydantic for configuration and structured data validation.
- Docling and Docling Graph for document parsing, extraction, and provenance.
- PostgreSQL for application metadata.
- Qdrant for vector retrieval.
- Neo4j for relationship-oriented queries where graph retrieval is justified.
- Gemini during early development.
- Ollama or vLLM for local model serving.
- React or Next.js for the eventual production frontend.
- Docker Compose for local and server deployment.

This architecture is incremental. Do not introduce every component at once.

## Package and Dependency Rules

- Use `uv` for Python dependency management.
- Keep `pyproject.toml` and `uv.lock` synchronized.
- Explain why a new production dependency is necessary before adding it.
- Prefer maintained libraries with clear licenses.
- Avoid introducing overlapping libraries that solve the same problem.
- Pin model names and important dependency versions where reproducibility matters.

## Python Standards

- Support the Python versions declared in `pyproject.toml`.
- Use type annotations for public functions and service boundaries.
- Use Pydantic models at API and LLM structured-output boundaries.
- Keep domain logic separate from FastAPI route handlers.
- Prefer explicit interfaces over direct dependencies on one model provider.
- Use asynchronous code only when the operation is genuinely I/O-bound.
- Do not hide errors with broad `except Exception` blocks.
- Include useful error context without exposing secrets.

## Testing and Verification

- Run relevant tests after every behavioral change.
- Run the full test suite before declaring a milestone complete.
- Add tests for new behavior and important failure cases.
- Do not weaken or delete tests merely to make a change pass.
- Test malformed model output, missing evidence, API timeouts, and unsupported files.
- Evaluation results must be reproducible from saved datasets and configuration.

Default verification commands:

```bash
uv run pytest
uv run ruff check .
uv run pyrefly check
```

## Product Roadmap

Follow `docs/PRODUCT_ROADMAP.md` for the phased EvidenceGraph product plan,
scope, dependencies, and acceptance criteria. The initial product focuses on
AI research papers, with PDF reading, traceable answers, an interactive evidence
graph, and cross-paper comparison. Other document domains are future extensions.

- Implement one agreed phase at a time; a roadmap is not authorization to deploy,
  purchase infrastructure, or implement every phase in one turn.
- Keep phase status and acceptance evidence in the roadmap up to date.
- Preserve the distinction between graph browsing (part of the MVP) and using
  graph retrieval to improve answers (requires measured justification).
- Explain concepts and design trade-offs in plain language; the user prefers
  understanding mechanisms to following line-by-line code tutorials.

## Current Handoff (2026-09-22)

- The active development checkout is this macOS clone. The separate Windows
  clone at `G:\Logs_ML\Graph_Bot` does not receive local changes automatically.
  Confirm the target checkout and inspect branch/status before further edits.
- Phase 0 remains **not complete**. Three sample AI papers have been parsed with
  Docling and provenance checked; graph and QA gold annotations exist for the
  first two papers. RoBERTa annotation expansion, product-limit measurements,
  and the PDF-to-graph user workflow remain outstanding. See
  `docs/PRODUCT_ROADMAP.md` for exact acceptance evidence and limitations.
- On macOS, the Stage 0 Compose scaffold was made runnable using
  `compose.stage0.yaml`. The original backend image failed during a
  `transformers 5.16.1` import with `torch 2.14.0`. A uv constraint
  `torch<2.14` in `pyproject.toml` and the regenerated `uv.lock` select
  `torch 2.13.0`. A forced no-cache backend image rebuild was necessary after
  an earlier cached build kept the old image. Last verified: backend container
  healthy, `/health` returned `{"status":"ok"}`, and the Streamlit port 8501
  returned HTTP 200. The user's subsequent stop/start actions are unknown;
  inspect live state before assuming services are running.
- Verification after the dependency fix: `uv lock --check --offline` passed;
  206 tests passed and 4 Docker tests skipped; Ruff passed; Pyrefly reported
  0 errors. This verifies the scaffold, not PDF upload or graph browsing.
- At this handoff, `AGENTS.md`, `docs/PRODUCT_ROADMAP.md`, `pyproject.toml`, and
  `uv.lock` have uncommitted changes in the Mac checkout. The `AGENTS.md`
  environment notes predated the dependency fix and belong to the user; preserve
  them. The dependency fix has not been committed, pushed, or copied to Windows.
- `docker compose -f compose.stage0.yaml stop` stops the two services while
  keeping their containers and named SQLite volume. Never use `down -v` as a
  routine stop command. Do not start/restart services, rebuild images, deploy,
  commit, or push without the user's authorization for that action.

## Phase 0 Follow-up (2026-09-22)

- The three fixed PDF samples have been restored and hash-checked in the Mac
  checkout. RoBERTa now has a manually reviewed BERT relationship and two QA
  cases. The annotation sets validate against all three newly parsed PDFs.
- Parsed-document schema version 2 removes Docling's internal item number from
  evidence block IDs after a Mac/Windows item-number shift broke four old
  Transformer citations. Reparse old version 1 JSON before validation.
- Initial per-file macOS time and peak-memory observations are recorded in
  `docs/PHASE0_LIMITS.md`. Real malformed, encrypted, and blank PDFs now have a
  repeatable failure check. Phase 0 is still not accepted: image-only scanned
  input, cold-cache/concurrent behavior, upload limits, and user-facing errors
  remain unverified. The PDF-to-graph user workflow belongs to phase 1.
- Verification after this follow-up: 210 tests passed, 4 Docker tests skipped;
  Ruff passed and Pyrefly reported 0 errors. No service, container, deployment,
  commit, or push action was taken.

## Phase 0 Closeout (2026-09-22)

- Phase 0 is accepted for the three exact-hash, local research-paper samples;
  see the acceptance table in `docs/PRODUCT_ROADMAP.md`. Earlier incomplete
  status notes above are historical snapshots. This does not include a PDF
  upload UI, automatic graph extraction, or general upload limits.
- A real image-only scanned PDF is now part of the repeatable failure check and
  is rejected without OCR. `docs/STAGE0_RUNBOOK.md` documents the previously
  verified local scaffold start; this closeout only checked Compose syntax and
  did not start or restart containers. Cold-cache, concurrency, upload caps,
  user-facing errors, and path redaction remain phase 1 release gates.
- Existing uncommitted user changes remain in place. No commit, push, image
  rebuild, deployment, or API-key use was performed during this closeout.
