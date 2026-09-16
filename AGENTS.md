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
