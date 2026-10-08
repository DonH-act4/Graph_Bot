import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

// Isolate the original upload/graph assertions from the new workspace endpoints.
// Persistence behavior has its own App.persistence suite.
function stubWorkspaceFetch(fetchMock: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/session") {
      return Promise.resolve(new Response(JSON.stringify({ user_id: "test-browser", identity_enforced: false }), {
        status: 200, headers: { "Content-Type": "application/json" },
      }));
    }
    if (url === "/api/history") {
      return Promise.resolve(new Response(JSON.stringify({ messages: [], conversation: null }), {
        status: 200, headers: { "Content-Type": "application/json" },
      }));
    }
    if (url.startsWith("/api/conversations/")) {
      const state = JSON.parse(String(init?.body ?? "{}"));
      return Promise.resolve(new Response(JSON.stringify({ ...state, thread_id: url.split("/").at(-1) }), {
        status: 200, headers: { "Content-Type": "application/json" },
      }));
    }
    return fetchMock(input, init);
  });
}

afterEach(() => {
  window.localStorage.clear();
  window.history.replaceState(null, "", "/");
  vi.unstubAllGlobals();
});

describe("App upload limits", () => {
  it("shows the real Docling milestone instead of a generic preparing state", async () => {
    const documentId = "a".repeat(64);
    window.history.replaceState(null, "", `/?document_id=${documentId}`);
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/papers/configuration")) {
        return Promise.resolve(new Response(JSON.stringify({
          max_pdf_bytes: 25 * 1024 * 1024,
          graph_models: ["gemini-fast"],
          default_graph_model: "gemini-fast",
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      if (url.startsWith("/api/threads?")) {
        return Promise.resolve(new Response(JSON.stringify({ threads: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      }
      if (url === `/api/papers/${documentId}`) {
        return Promise.resolve(new Response(JSON.stringify({
          document_id: documentId,
          state: "processing",
          stage: "docling_parsing",
          progress_percent: 20,
          page_count: null,
          block_count: null,
          error: null,
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    stubWorkspaceFetch(fetchMock);

    render(<App />);

    expect(await screen.findByText("Docling is parsing the paper")).toBeInTheDocument();
    expect(screen.getByText(/Reading layout, text, tables/)).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "8");
    expect(screen.getByText(/Milestone progress/)).toBeInTheDocument();
  });

  it("shows exact file and limit sizes without uploading an oversized PDF", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/papers/configuration")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              max_pdf_bytes: 25 * 1024 * 1024,
              graph_models: ["gemini-fast"],
              default_graph_model: "gemini-fast",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      if (url.startsWith("/api/threads?")) {
        return Promise.resolve(
          new Response(JSON.stringify({ threads: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    stubWorkspaceFetch(fetchMock);
    render(<App />);
    await screen.findByText(/up to 25.00 MiB/);

    const file = new File(["%PDF-test"], "large-paper.pdf", { type: "application/pdf" });
    Object.defineProperty(file, "size", { value: 26 * 1024 * 1024 });
    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();
    fireEvent.change(input!, { target: { files: [file] } });

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("PDF is too large");
    expect(alert).toHaveTextContent("large-paper.pdf is 26.00 MiB");
    expect(alert).toHaveTextContent("maximum allowed size is 25.00 MiB");
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });
  });


  it("stops after parsing so the user can choose a model before graph generation", async () => {
    const documentId = "b".repeat(64);
    window.history.replaceState(null, "", `/?document_id=${documentId}`);
    const failure = "invalid ontology v2 endpoints for uses: concept -> artifact";
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/papers/configuration")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              max_pdf_bytes: 25 * 1024 * 1024,
              graph_models: ["gemini-fast", "gemini-pro"],
              default_graph_model: "gemini-fast",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      if (url.startsWith("/api/threads?")) {
        return Promise.resolve(
          new Response(JSON.stringify({ threads: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      if (url === `/api/papers/${documentId}`) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              document_id: documentId,
              state: "ready",
              page_count: 20,
              block_count: 397,
              error: null,
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      if (url === `/api/papers/${documentId}/graph`) {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: "Graph is failed" }), {
            status: 409,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      if (url === `/api/papers/${documentId}/graph/status`) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              document_id: documentId,
              state: "failed",
              requested_model: "gemini-fast",
              error: failure,
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    stubWorkspaceFetch(fetchMock);

    render(<App />);

    expect(await screen.findByText("The last graph attempt failed")).toBeInTheDocument();
    expect(screen.getByText(failure)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /Extraction model/i })).toHaveValue(
      "gemini-fast",
    );
    expect(screen.getByRole("button", { name: "Retry with selected model" })).toBeEnabled();
    expect(
      fetchMock.mock.calls.some(([, init]) => (init as RequestInit | undefined)?.method === "POST"),
    ).toBe(false);
  });

  it("polls after generation and surfaces a model failure instead of waiting forever", async () => {
    const documentId = "c".repeat(64);
    window.history.replaceState(null, "", `/?document_id=${documentId}`);
    let graphRequested = false;
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith("/api/papers/configuration")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              max_pdf_bytes: 25 * 1024 * 1024,
              graph_models: ["gemini-3.6-flash", "gemini-fast"],
              default_graph_model: "gemini-3.6-flash",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      if (url.startsWith("/api/threads?")) {
        return Promise.resolve(
          new Response(JSON.stringify({ threads: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      if (url === `/api/papers/${documentId}`) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              document_id: documentId,
              state: "ready",
              page_count: 20,
              block_count: 397,
              error: null,
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      if (url === `/api/papers/${documentId}/graph` && init?.method === "POST") {
        graphRequested = true;
        return Promise.resolve(
          new Response(
            JSON.stringify({
              document_id: documentId,
              state: "queued",
              requested_model: "gemini-3.6-flash",
              error: null,
            }),
            { status: 202, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      if (url === `/api/papers/${documentId}/graph`) {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: graphRequested ? "Graph is queued" : "Graph not found" }), {
            status: graphRequested ? 409 : 404,
            headers: { "Content-Type": "application/json" },
          }),
        );
      }
      if (url === `/api/papers/${documentId}/graph/status`) {
        if (!graphRequested) {
          return Promise.resolve(
            new Response(JSON.stringify({ detail: "Graph status not found" }), {
              status: 404,
              headers: { "Content-Type": "application/json" },
            }),
          );
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              document_id: documentId,
              state: "failed",
              requested_model: "gemini-3.6-flash",
              error: "Gemini API request failed (code=503, status=UNAVAILABLE)",
            }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    stubWorkspaceFetch(fetchMock);

    render(<App />);

    const generate = await screen.findByRole("button", { name: "Generate graph" });
    await waitFor(() => expect(screen.getAllByRole("combobox")).toHaveLength(1));
    await waitFor(() => expect(generate).toBeEnabled());
    expect(screen.getByRole("button", { name: /New chat/ })).toBeInTheDocument();
    expect(screen.getByText("Recent conversations")).toBeInTheDocument();
    fireEvent.click(generate);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Graph generation failed");
    expect(alert).toHaveTextContent("gemini-3.6-flash");
    expect(alert).toHaveTextContent("503");
    expect(alert).toHaveTextContent("Choose another model or retry later");
    expect(screen.getByRole("button", { name: "Retry with selected model" })).toBeEnabled();
  });

  it("shows a non-blocking warning when a recovered graph is ready", async () => {
    const documentId = "d".repeat(64);
    window.history.replaceState(null, "", `/?document_id=${documentId}`);
    const warning = "Omitted 1 relation whose endpoint types could not be repaired safely.";
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/papers/configuration")) {
        return Promise.resolve(new Response(JSON.stringify({
          max_pdf_bytes: 25 * 1024 * 1024,
          graph_models: ["gemini-fast"],
          default_graph_model: "gemini-fast",
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      if (url.startsWith("/api/threads?")) {
        return Promise.resolve(new Response(JSON.stringify({ threads: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      }
      if (url === `/api/papers/${documentId}`) {
        return Promise.resolve(new Response(JSON.stringify({
          document_id: documentId,
          state: "ready",
          page_count: 1,
          block_count: 1,
          error: null,
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      if (url === `/api/papers/${documentId}/graph`) {
        return Promise.resolve(new Response(JSON.stringify({
          graph_version: 1,
          document_sha256: documentId,
          extractor_name: "fake",
          extractor_version: "1",
          warnings: [{
            code: "invalid_relations_omitted",
            message: warning,
            relation_ids: ["bad-edge"],
          }],
          graph: {
            schema_version: "2",
            nodes: [{
              node_id: "paper",
              node_type: "paper",
              name: "Recovered paper",
              domain_type: null,
              evidence: [],
              value: null,
              unit: null,
              uncertainty: null,
              conditions: null,
            }],
            relations: [],
          },
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      if (url === `/api/papers/${documentId}/graph/status`) {
        return Promise.resolve(new Response(JSON.stringify({
          document_id: documentId,
          state: "ready",
          node_count: 1,
          relation_count: 0,
          requested_model: "gemini-fast",
          current_version: 1,
          warnings: [],
          error: null,
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    stubWorkspaceFetch(fetchMock);

    render(<App />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Graph generated with warnings");
    expect(alert).toHaveTextContent(warning);
    expect(screen.getByRole("button", { name: /Open graph/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Open graph/i })).toHaveClass("is-ready");
    expect(screen.getByText("Ready · explore evidence")).toBeInTheDocument();
  });

  it("does not label an older failed attempt as the newly selected local model", async () => {
    const documentId = "e".repeat(64);
    window.history.replaceState(null, "", `/?document_id=${documentId}`);
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/papers/configuration")) {
        return Promise.resolve(new Response(JSON.stringify({
          max_pdf_bytes: 25 * 1024 * 1024,
          graph_models: ["ollama/qwen3:14b-q4_K_M"],
          default_graph_model: "ollama/qwen3:14b-q4_K_M",
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      if (url.startsWith("/api/threads?")) {
        return Promise.resolve(new Response(JSON.stringify({ threads: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }));
      }
      if (url === `/api/papers/${documentId}`) {
        return Promise.resolve(new Response(JSON.stringify({
          document_id: documentId,
          state: "ready",
          page_count: 1,
          block_count: 1,
          error: null,
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      if (url === `/api/papers/${documentId}/graph`) {
        return Promise.resolve(new Response(JSON.stringify({
          graph_version: 4,
          document_sha256: documentId,
          extractor_name: "gemini",
          extractor_version: "old",
          warnings: [],
          graph: {
            schema_version: "3",
            nodes: [{ node_id: "paper", node_type: "paper", name: "Saved paper", evidence: [] }],
            relations: [],
          },
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      if (url === `/api/papers/${documentId}/graph/status`) {
        return Promise.resolve(new Response(JSON.stringify({
          document_id: documentId,
          state: "failed",
          requested_model: null,
          current_version: 4,
          warnings: [],
          error: "Gemini API request failed (503)",
        }), { status: 200, headers: { "Content-Type": "application/json" } }));
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    stubWorkspaceFetch(fetchMock);

    render(<App />);

    await waitFor(() => expect(screen.getByRole("button", { name: /Open graph/i })).toBeEnabled());
    await waitFor(() => expect(fetchMock.mock.calls.some(
      ([input]) => String(input) === `/api/papers/${documentId}/graph/status`,
    )).toBe(true));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("combobox")).toHaveValue("ollama/qwen3:14b-q4_K_M");
  });
});
