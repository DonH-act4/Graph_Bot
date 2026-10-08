# Local Docker runbook

This runbook reproduces the Agent Service Toolkit scaffold plus the phase 1 internal PDF upload
and evidence-chat slice on macOS. Compose runs the API, a Docling-only paper worker, the React
workspace served by nginx, and the legacy Streamlit frontend.
The worker is separate because native parsing failures must not terminate the Agent API process.
The React workspace provides upload, processing status, the current graph, interactive node/relation
evidence cards, evidence selection, and evidence-scoped chat. Streamlit remains available as a
temporary compatibility and diagnostics surface.

## Prerequisites and configuration

- Docker with Compose, the checked-out repository, and `compose.stage0.yaml`.
- A Git-ignored `.env` file. If absent, copy `.env.example` and set `USE_FAKE_MODEL=true` and
  `DATABASE_TYPE=sqlite`; leave `DEFAULT_MODEL` blank. No provider API key is needed for the fake
  model smoke test. Do not overwrite an existing `.env` while reproducing this runbook.
- Real graph extraction requires a configured provider: an Ollama server URL for local extraction,
  or credentials for a hosted provider. For the Mac-to-Windows local path, keep Ollama bound to
  Windows loopback, run an SSH local port forward on the Mac, and configure the Git-ignored `.env`:

  ```dotenv
  USE_FAKE_MODEL=false
  DEFAULT_MODEL=ollama
  EVIDENCEGRAPH_GRAPH_MODEL=ollama/gpt-oss:20b
  EVIDENCEGRAPH_GRAPH_MODELS=ollama/gpt-oss:20b
  OLLAMA_MODEL=gpt-oss:20b
  OLLAMA_BASE_URL=http://host.docker.internal:11434
  ```

  The Windows host has a separately installed Ollama CLI and `gpt-oss:20b` model on G:.
  Start its server with `OLLAMA_MODELS` pointing to the model directory and
  `OLLAMA_HOST=127.0.0.1:11434`; then forward Mac port 11434 to that Windows loopback port with
  `ssh -N -L 11434:127.0.0.1:11434 <configured-windows-ssh-host>`. Check
  `http://127.0.0.1:11434/api/tags` on the Mac before starting the Docker worker. Do not put the
  Tailscale address or SSH credentials in this repository. The Windows server and Mac tunnel must
  remain running while generating a graph or answering questions; Docker configuration alone
  does not start them. This local demo enables gpt-oss reasoning, a 16,384-token context,
  and an 8,192-token generation ceiling. Those are ceilings, not guaranteed output sizes.
  Graph extraction uses compact JSON mode with low reasoning and at most one repair
  per section; chat uses normal reasoning. In this installation, full-schema mode with
  low reasoning sometimes returned an empty body despite HTTP 200. Empty or invalid
  output is not treated as a successful graph.
  The model directory and SSH host may differ on another installation. Ollama needs no cloud
  API key, but its quality is not yet accepted for cross-domain papers.
- For the Groq
  free-tier path, sign in at `https://console.groq.com/`, create or select a project, open
  `https://console.groq.com/keys`, and create a key named for the local EvidenceGraph environment.
  Copy it once into the Git-ignored `.env`; never paste it into chat, logs, commits, screenshots, or
  `.env.example`:

  ```dotenv
  GROQ_API_KEY=your_groq_key_here
  EVIDENCEGRAPH_GRAPH_MODEL=groq/openai/gpt-oss-120b
  EVIDENCEGRAPH_GRAPH_MODELS=
  ```

  Leaving the allowlist blank lets EvidenceGraph show the maintained production Groq models for
  the configured key. `groq/openai/gpt-oss-20b` is the faster secondary option. Gemini remains
  optional through `GOOGLE_API_KEY`; historical unprefixed Gemini model names continue to work.
  If no graph provider is configured, PDF parsing still works and graph task creation returns
  HTTP 503. A configured but unreachable Ollama server instead produces a bounded worker error.
- Groq Free currently publishes an 8K token-per-minute and 200K token-per-day limit for these
  models. EvidenceGraph therefore extracts source-located sections serially, records section and
  quota-wait progress, and merges them locally. This avoids a known impossible whole-paper request,
  but it does not make the free quota fast or unlimited; a long paper may take several minutes.
- PDF uploads are limited to 25 MiB at both nginx and FastAPI. The backend value defaults to
  `EVIDENCEGRAPH_MAX_PDF_BYTES=26214400`; keep nginx and API limits aligned if this changes.
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
docker compose -f compose.stage0.yaml up -d --build agent_service paper_worker web_app
docker compose -f compose.stage0.yaml ps
curl --fail http://127.0.0.1:8080/health
curl --silent --show-error --output /dev/null --write-out '%{http_code}\n' http://127.0.0.1:3000/
```

The API should be healthy, `paper_worker` should remain running, `/health` should return
`{"status":"ok"}`, and the React check should print `200`. Open
`http://127.0.0.1:3000/` for the React workspace. The legacy Streamlit fallback at port 8501
is not started by the three-service command above; it is not needed for the portfolio demo.
The nginx frontend is intentionally bound to loopback and is not production-ready authentication.
Changing `.env` or backend code requires a separately authorized worker/API rebuild and restart
before the model catalog changes in the running site. The frontend also requires a rebuild for
new model labels and UI cleanup.
If a previously cached backend image still imports `torch 2.14.0`, a separately
authorized `docker compose -f compose.stage0.yaml build --no-cache agent_service` is needed before
starting again. Do not use `docker compose watch` for this baseline; it can restart services when
files change.

When authorized to stop the services, use
`docker compose -f compose.stage0.yaml stop`. This preserves containers and both named volumes.
Do not use `down -v` as a routine stop command. The Mac and Windows clones do not synchronize;
these instructions and the observed result apply to this Mac checkout only.

For the research showcase, real sample conversation, and persistence design, see
`docs/DEMO_RUNBOOK.md`. The current local catalog intentionally exposes only gpt-oss:20b;
historical Qwen/Groq artifacts remain readable and installed model files are not deleted.
