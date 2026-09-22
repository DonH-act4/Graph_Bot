# Stage 0 local runbook

This runbook reproduces the existing Agent Service Toolkit scaffold on macOS. It checks the
backend and Streamlit frontend only; it does not provide EvidenceGraph PDF upload or graph browsing.
The last recorded live check was on 2026-09-22: backend healthy, `/health` returned
`{"status":"ok"}`, and the local Streamlit page returned HTTP 200. The present phase 0 closeout
checked Compose configuration but did not start or restart either service.

## Prerequisites and configuration

- Docker with Compose, the checked-out repository, and `compose.stage0.yaml`.
- A Git-ignored `.env` file. If absent, copy `.env.example` and set `USE_FAKE_MODEL=true` and
  `DATABASE_TYPE=sqlite`; leave `DEFAULT_MODEL` blank. No provider API key is needed for the fake
  model smoke test. Do not overwrite an existing `.env` while reproducing this runbook.
- The current `uv.lock` plus the existing `torch<2.14` uv constraint. An image built before this
  constraint may still contain the incompatible `torch 2.14.0` package.

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

The backend should be healthy, `/health` should return `{"status":"ok"}`, and the last command
should print `200`. If a previously cached backend image still imports `torch 2.14.0`, a separately
authorized `docker compose -f compose.stage0.yaml build --no-cache agent_service` is needed before
starting again. Do not use `docker compose watch` for this baseline; it can restart services when
files change.

When authorized to stop the smoke-test services, use
`docker compose -f compose.stage0.yaml stop`. This preserves containers and the named SQLite volume.
Do not use `down -v` as a routine stop command. The Mac and Windows clones do not synchronize;
these instructions and the observed result apply to this Mac checkout only.
