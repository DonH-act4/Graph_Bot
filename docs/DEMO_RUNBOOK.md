# EvidenceGraph local portfolio demo

## Scope

This is a single-paper research demo, not a production multi-user service. The normal
workspace is `http://127.0.0.1:3000/`; the one-page tutorial is
`http://127.0.0.1:3000/?view=tutorial` (old `view=showcase` links still work).
Keep Docker, the remote Ollama server, Tailscale,
and the Mac SSH tunnel running. Do not expose Ollama or this locally authenticated site
publicly. The Windows repository is independent of this macOS checkout.

The default model is `ollama/gpt-oss:20b` for both graph extraction and paper chat.
No cloud API key is needed. Qwen is no longer in this demo's model catalog; its model
files and previous artifacts have not been deleted.

## User journey

1. Start a new conversation and upload a PDF (up to 25 MiB).
2. See parsing and numbered extraction-section progress. Successful parsing triggers
   graph generation automatically; failures display an actionable error and retry control.
3. Sign in or create an account before asking a question. The backend retrieves source
   blocks from that paper; manual evidence selection is optional.
4. Open the graph at any point, drag nodes, inspect a node or relationship, and read the
   original passage in source cards. A page link opens the stored PDF at the source page.
5. Select up to 12 source cards and return to the conversation for a focused question.
   If still a guest, signing in preserves the draft and selected block IDs in this tab.
6. Switch conversations or refresh: paper association, selected blocks, and a draft are
   restored from the API. Answers and prior evidence context come from saved checkpoints.

The tutorial loads the saved real graph and three question/answer pairs on the same page.
Its graph is draggable and its nodes/relationships open original evidence in a panel;
there is no separate sample-conversation jump in the normal flow. The tutorial is read-only
and does not overwrite the sample, impersonate its owner, or change the browser's personal
user identifier. **Back to chat** restores the personal workspace; **Start with your own PDF**
opens the existing upload flow. The featured Water Research paper is co-authored by the
project creator; its DOI is `10.1016/j.watres.2026.126464`. This label/DOI is shown only
when the saved sample points to that exact PDF hash.

## What is stored

- `/app/data/stage0-smoke.db`: LangGraph SQLite checkpoints, including message history.
- `/app/data/evidencegraph/conversations.sqlite3`: conversation title, owner identifier,
  paper association, draft, selected evidence IDs, and update times.
- `/app/data/evidencegraph/accounts.sqlite3`: Argon2id password hashes and revocable
  login-session hashes; raw passwords and login tokens are not stored there.
- `/app/data/evidencegraph/paper-access.sqlite3`: guest and account PDF grants.
- `/app/data/evidencegraph/request-quotas.sqlite3`: bounded request counters for
  registration, login, uploads, graph jobs, and chat (created by the new code).
- `/app/data/evidencegraph/showcase.json`: an index pointing to an existing saved conversation,
  not a hard-coded transcript.
- `/app/data/evidencegraph/<document-sha256>/`: PDF, parsed source blocks, task state,
  and versioned graphs.

All are in the existing `stage0_sqlite` Compose volume. Routine container replacement or
`docker compose ... stop` preserves the volume. **Do not use `down -v`** to stop the demo.
With `EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT=true`, a server-issued account ID controls
private chat and history; the old browser-generated ID is not an authentication credential.
The current local account implementation is **not** a public multi-user deployment gate.
The Mac source now checks unsafe-request Origin/Referer, persists provisional IP/account
request quotas, and fails startup in public mode when core HTTPS/Cookie/proxy settings are
missing. These changes have **not** been loaded into the running containers. Account recovery,
retention/deletion policy, real HTTPS and client-IP checks, PDF parser resource isolation,
and deployment tests remain open. Open registration was chosen for eventual resume-visitor
trial; provisional site-wide cost ceilings are 30 uploads and 30 graph requests per day,
plus 240 chats per hour, and need tuning against target-machine measurements.
The new source optionally requires emailed verification for self-service registration;
it is disabled in the current local configuration and has not sent a real email. Existing
username/password accounts remain usable. No automatic account or paper deletion is planned;
the owner needs saved data for demonstrations.

## Generate a reproducible sample

With the local services and model tunnel available, run:

```bash
uv run python scripts/create_showcase.py \
  --pdf '/absolute/path/to/paper.pdf' \
  --output data/evidencegraph/showcase-run.json \
  --title 'Full paper title'
```

This command uploads through the real API, waits for Docling and the graph worker, and
asks three real questions using the configured local model. It saves a resumable manifest
in a Git-ignored directory. It does not invent answers, publish to the internet, or change
the current graph of another paper. Each stage has a bounded wait; retries preserve the PDF.
The bundled questions concern the transfer-learning example; adjust them in the script
before using an unrelated paper. Do not rerun against a published read-only sample without
first choosing a different sample thread.

After inspecting the successful run, create the local showcase index from its manifest:

```bash
docker compose -f compose.stage0.yaml exec -T agent_service python -c \
  'import json,sys; from core import settings; from evidencegraph.conversations import ConversationStore; p=json.load(sys.stdin); assert p["complete"]; ConversationStore(settings.EVIDENCEGRAPH_DATA_DIR).publish_showcase(p["thread_id"],p["user_id"],p["title"],"A real paper, generated graph, and evidence-backed conversation.",p["model"])' \
  < data/evidencegraph/showcase-run.json
```

This writes only the local showcase index. Opening the presentation page then reads the
sample's actual stored messages and graph. Do not check transcripts, PDF text, or runtime
manifests into the repository merely to reproduce a local demo.

## Design and limits worth explaining in an interview

- Docling preserves source blocks and page locations. GPT-OSS graph extraction samples
  representative abstract, methods, findings, and conclusion passages (two sections in
  the featured paper), rather than promising exhaustive coverage. Unrecognized headings
  use a bounded evenly spaced fallback; source IDs and full parsed text remain unchanged.
  Full-paper evidence remains available to question retrieval. The model extracts bounded sections;
  deterministic merging deduplicates nodes and keeps a balanced overview of methods,
  observations, claims, and research objects (at most 25 nodes and 40 relations).
- A local-model normalization clears optional measurement fields misplaced on non-observation
  nodes. It never changes node types, labels, facts, edges, or evidence IDs. Actual observations
  keep values/units. JSON shape, relation endpoints, and real source references remain checked.
- Missing relationship IDs receive deterministic program-generated identifiers before
  validation. This does not change endpoint IDs, relation meaning, or citations; malformed
  non-string identifiers are still rejected. Merging subsequently assigns canonical IDs.
- GPT-OSS uses compact JSON mode and low reasoning for graphs, with a maximum of one
  repair per section. The full-schema constraint with low reasoning sometimes produced
  an empty answer in this environment. Chat keeps normal reasoning. Internal reasoning
  is not displayed or saved by the graph adapter or the new paper-answer checkpoint path.
- Disconnected graph components are omitted together with their internal relationships,
  so pruning does not leave dangling edges. No edges are invented to connect them.
- Citation validity is not semantic truth. The model can miss findings, misunderstand a
  relation, or attach a real but weak supporting passage. The graph is a browsable overview,
  not a clinical/regulatory decision engine. Missing edges are not fabricated to improve appearance.
- Question retrieval is a small lexical/section-intent baseline with Chinese term expansion,
  not vector or graph retrieval. It is bounded and can miss paraphrases or table context.
  The answering model is instructed to cite source blocks; semantic citation support still
  requires evaluation, rather than being guaranteed by a prompt.
- SQLite and versioned JSON satisfy the current demo without introducing Neo4j/Qdrant just
  for the stack. Model serving is separate from the API and parsing worker.
- Parsing/extraction are asynchronous jobs; long documents still take minutes. A failed
  section can fail the whole job after one bounded repair; there is no hidden infinite retry.
- Local account sign-in, document authorization, and deletion are implemented. The new
  source has CSRF source checks, provisional quotas, and an isolated backup verification
  script, but not a verified public deployment or systematic cross-domain quality evaluation.

## Local account rollout verified on 2026-10-07

- The Mac `.env` (Git-ignored) enables `EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT=true`.
  `agent_service` and `web_app` were rebuilt and replaced; `paper_worker`, Streamlit,
  and the `stage0_sqlite` volume were not replaced or cleared. The API is healthy.
- The pre-switch SQLite online backup is in the volume at
  `/app/data/backups/pre-account-20261007T032211Z/`, with a second copy at
  `data/evidencegraph/pre-account-20261007T032211Z/` in the Mac checkout (Git-ignored).
  Both contain the checkpoint database, conversation database, and Tutorial index.
- A separate full copy of `/app/data` (including saved PDFs and graph artifacts) is at
  `data/evidencegraph/pre-account-full-20261007T032211Z/` in the Mac checkout
  (Git-ignored, about 21 MiB). This full copy was taken **after** activation; the SQLite
  online snapshot above was taken **before** activation. Both have since passed an isolated
  copy/integrity/reference check; neither has been restored into a running service.
- Eight old conversation rows across three browser-selected owner IDs remain in the live
  database and backup. They were **not** automatically assigned to a new account. New
  accounts start with empty private history; older PDF links without an access grant ask
  for re-upload. No old PDF or conversation was deleted.
- A guest can upload a PDF, generate/view its graph, inspect source cards, and read the
  Tutorial. The API returns 401 for guest history/chat. Register or sign in to ask and
  save; logging in atomically moves this browser's guest PDF grants to the account, so
  its papers remain available across devices after the guest cookie expires. Logging
  out leaves those papers private to the account.
- To verify manually: open `http://127.0.0.1:3000/`; inspect the Tutorial without signing
  in; upload a **new or re-uploaded** PDF and wait for its graph; choose evidence and type
  a question; select **Sign in to ask questions** and create your own account; check that
  the draft/evidence remain; send a question, refresh to check history; sign out and
  confirm private history is no longer visible. The model connection is needed only for
  new graph generation and answers, not the saved Tutorial.
- The Mac loopback SSH tunnel was restored and the worker confirmed that Ollama's
  `/api/tags` is reachable and `gpt-oss:20b` is installed. This is a connectivity
  check, not a new end-to-end generation or answer-quality test.

Useful interview questions: Why are provenance checks separate from answer correctness?
Why persist workspace metadata separately from model checkpoints? How does the worker
protect the API from parser failures? Why chunk inputs and merge deterministically? What
would justify adding vector retrieval, graph queries, or a production database?

## Featured run verified on 2026-10-05

The Water Research paper "Explainable simulation-to-field transfer learning for
transient-based leak detection in water distribution networks" has 20 pages and
258 parsed source blocks. The current real GPT-OSS overview contains 9 nodes and
12 relationships, with no structural warnings. The final graph regeneration took
44.5 seconds with parsed input and a warm model; this is not a cold upload benchmark.
The showcase has three real saved question/answer pairs. Browser verification also
created a separate personal conversation, clicked a relationship, selected its page-3
source card, and completed a real selected-evidence question.

Verified references point to actual saved blocks, but the first sample answer includes
an overbroad baseline-comparison statement. Do not present the sample as perfectly
correct or manually replace model text while claiming it is unedited output. Saved
sample browsing works without model inference; new extraction or questions require
the remote model connection. See the roadmap for exact acceptance scope and limits.
