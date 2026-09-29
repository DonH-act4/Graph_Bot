# Local Docker runbook

This runbook reproduces the Agent Service Toolkit scaffold plus the phase 1 internal PDF upload
slice on macOS. Compose runs the API, a Docling-only paper worker, and the Streamlit frontend.
The worker is separate because native parsing failures must not terminate the Agent API process.
It provides upload, processing status, paginated evidence browsing, and an optional candidate-graph
task. The graph path has not yet been verified with a real model and is not an interactive graph UI.

## Prerequisites and configuration

- Docker with Compose, the checked-out repository, and `compose.stage0.yaml`.
- A Git-ignored `.env` file. If absent, copy `.env.example` and set `USE_FAKE_MODEL=true` and
  `DATABASE_TYPE=sqlite`; leave `DEFAULT_MODEL` blank. No provider API key is needed for the fake
  model smoke test. Do not overwrite an existing `.env` while reproducing this runbook.
- Real candidate extraction additionally requires both `GOOGLE_API_KEY` and an explicit
  `EVIDENCEGRAPH_GRAPH_MODEL`. Keep the key out of chat, logs, commits, and screenshots. If either
  is absent, PDF parsing still works and graph task creation returns HTTP 503.
- The current `uv.lock` plus the existing `torch<2.14` uv constraint. An image built before this
  constraint may still contain the incompatible `torch 2.14.0` package.
- The service image installs the Debian OpenCV runtime libraries required by Docling's table model.
  The `stage0_models` volume caches Hugging Face weights; `stage0_sqlite` persists both SQLite and
  EvidenceGraph PDF artifacts. Do not remove these volumes during a routine stop.

From the repository root, validate the Compose configuration without starting anything:

```bash
docker compose -f compose.stage0.yaml config --quiet
```

With separate authorization to build and start the local scaffold, run:

```bash
docker compose -f compose.stage0.yaml up -d --build
docker compose -f compose.stage0.yaml ps
curl --fail http://127.0.0.1:8080/health
curl --silent --show-error --output /dev/null --write-out '%{http_code}\n' http://127.0.0.1:8501/
```

The API should be healthy, `paper_worker` should remain running, `/health` should return
`{"status":"ok"}`, and the last command should print `200`. If a previously cached backend image still imports `torch 2.14.0`, a separately
authorized `docker compose -f compose.stage0.yaml build --no-cache agent_service` is needed before
starting again. Do not use `docker compose watch` for this baseline; it can restart services when
files change.

When authorized to stop the services, use
`docker compose -f compose.stage0.yaml stop`. This preserves containers and both named volumes.
Do not use `down -v` as a routine stop command. The Mac and Windows clones do not synchronize;
these instructions and the observed result apply to this Mac checkout only.
