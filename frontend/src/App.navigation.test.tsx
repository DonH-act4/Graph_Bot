import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import * as api from "./api";
import type { ConversationState, GraphArtifact } from "./types";

vi.mock("./api", async (importOriginal) => ({
  ...await importOriginal<typeof import("./api")>(),
  getHistory: vi.fn(), getPaper: vi.fn(), getGraph: vi.fn(), getGraphStatus: vi.fn(),
  getPaperConfiguration: vi.fn(), getShowcase: vi.fn(), listThreads: vi.fn(),
  getSession: vi.fn(),
  saveConversation: vi.fn(), deleteConversation: vi.fn(), requestGraph: vi.fn(), streamChat: vi.fn(),
}));
vi.mock("./components/GraphWorkspace", () => ({ GraphWorkspace: ({ inline }: { inline?: boolean }) => inline ? <div role="region" aria-label="Tutorial graph" /> : null }));

const documentId = "b0c74a6f6dc80e8c29cba07ba56a1b2de213bc9dbc41604a8e5081507a35fcc3";
const title = "A real research paper title";
const graph: GraphArtifact = {
  graph_version: 1, document_sha256: documentId, extractor_name: "ollama",
  extractor_version: "gpt-oss:20b:test", warnings: [],
  graph: { schema_version: "3", relations: [], nodes: [{
    node_id: "paper", node_type: "paper", name: title, domain_type: null,
    evidence: [], value: null, unit: null, uncertainty: null, conditions: null,
  }] },
};

function workspace(threadId: string, userId: string): ConversationState {
  return {
    thread_id: threadId, user_id: userId, agent_id: "research-assistant", title,
    document_id: documentId, document_name: title, selected_block_ids: [],
    draft_message: "", created_at: "2026-10-05", updated_at: "2026-10-05",
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  window.history.replaceState(null, "", `?user_id=personal-user&thread_id=personal-thread&document_id=${documentId}`);
  vi.mocked(api.getHistory).mockImplementation(async (threadId, userId) => ({
    conversation: workspace(threadId, userId ?? "personal-user"),
    messages: threadId === "sample-thread" ? [
      { type: "human", content: "Compare the results", custom_data: { evidence_context: {
        document_id: documentId, source_mode: "automatic" as const,
        blocks: [
          { block_id: `blk_${"1".repeat(24)}`, pages: [12], text: "First original passage" },
          { block_id: `blk_${"2".repeat(24)}`, pages: [13], text: "Second original passage" },
        ],
      } } },
      { type: "ai", content: "| Task | Before | After |\n| --- | --- | --- |\n| Accuracy | 30% | 62.5% |" },
    ] : [],
  }));
  vi.mocked(api.getPaper).mockResolvedValue({
    document_id: documentId, state: "ready", page_count: 20, block_count: 258,
    stage: "complete", progress_percent: 100, error: null,
  });
  vi.mocked(api.getGraph).mockResolvedValue(graph);
  vi.mocked(api.getGraphStatus).mockResolvedValue({
    document_id: documentId, state: "ready", node_count: 1, relation_count: 0,
    requested_model: "ollama/gpt-oss:20b", current_version: 1, stage: "complete",
    progress_percent: 100, progress_detail: null, warnings: [], error: null,
  });
  vi.mocked(api.getPaperConfiguration).mockResolvedValue({
    max_pdf_bytes: 25 * 1024 * 1024, graph_models: ["ollama/gpt-oss:20b"],
    default_graph_model: "ollama/gpt-oss:20b",
  });
  vi.mocked(api.getSession).mockResolvedValue({ user_id: "personal-user", identity_enforced: false });
  vi.mocked(api.getShowcase).mockResolvedValue({
    title, description: "Real saved sample", thread_id: "sample-thread",
    user_id: "sample-owner", document_id: documentId, model: "ollama/gpt-oss:20b",
  });
  vi.mocked(api.listThreads).mockResolvedValue({ threads: [] });
  vi.mocked(api.saveConversation).mockImplementation(async (threadId, state) => ({
    ...workspace(threadId, state.user_id), ...state,
  }));
  vi.mocked(api.deleteConversation).mockResolvedValue(undefined);
});

describe("same-paper conversation navigation and grounded chat", () => {
  it("shows the operations link only for the server-designated admin account", async () => {
    window.history.replaceState(null, "", "/");
    vi.mocked(api.getSession).mockResolvedValue({
      user_id: "owner", identity_enforced: true, authenticated: true,
      chat_requires_login: true, is_admin: true,
    });
    render(<App />);
    expect(await screen.findByRole("link", { name: /Operations dashboard/ })).toHaveAttribute("href", "/?view=admin");
  });

  it("keeps a draft but does not call a model without an attached parsed paper", async () => {
    window.history.replaceState(null, "", "?user_id=personal-user&thread_id=personal-thread");
    vi.mocked(api.getHistory).mockResolvedValue({
      messages: [], conversation: { ...workspace("personal-thread", "personal-user"), document_id: null, document_name: null },
    });
    render(<App />);
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toBeEnabled());
    fireEvent.change(screen.getByRole("textbox", { name: "Message" }), { target: { value: "Explain this paper" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Attach a paper first");
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("Explain this paper");
    expect(api.streamChat).not.toHaveBeenCalled();
  });

  it("shows the saved graph and answer inside Tutorial without switching the personal conversation", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByRole("button", { name: /Open graph/ })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /Interactive tutorial/ }));
    expect(await screen.findByRole("region", { name: "Tutorial graph" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /DOI 10\.1016\/j\.watres\.2026\.126464/ })).toHaveAttribute("href", "https://doi.org/10.1016/j.watres.2026.126464");
    expect(screen.getByRole("table")).toHaveTextContent("Accuracy");
    expect(screen.getByRole("table")).toHaveTextContent("62.5%");
    expect(screen.queryByRole("button", { name: /Open sample conversation/ })).not.toBeInTheDocument();
    expect(screen.queryByText("Research source")).not.toBeInTheDocument();
    expect(window.location.search).toBe("?view=tutorial");
    expect(window.localStorage.getItem("evidencegraph:user_id")).toBe("personal-user");
    expect(window.localStorage.getItem("evidencegraph:thread_id")).toBe("personal-thread");
    fireEvent.click(screen.getByRole("button", { name: /Back to chat/ }));
    const query = new URLSearchParams(window.location.search);
    expect(query.get("user_id")).toBe("personal-user");
    expect(query.get("thread_id")).toBe("personal-thread");
    expect(query.get("document_id")).toBe(documentId);
    expect(query.get("view")).toBeNull();
    expect(screen.getByRole("button", { name: /Open graph/ })).toBeEnabled();
    expect(api.requestGraph).not.toHaveBeenCalled();
  });

  it("asks before deleting another conversation and keeps the current chat", async () => {
    let rows = [{ thread_id: "old-thread", agent_id: "research-assistant", title: "Old draft", updated_at: "2026-10-04" }];
    vi.mocked(api.listThreads).mockImplementation(async () => ({ threads: rows }));
    vi.mocked(api.deleteConversation).mockImplementation(async () => { rows = []; });
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete conversation Old draft" }));
    expect(screen.getByRole("alertdialog", { name: "Delete this conversation?" })).toHaveTextContent("stops its active answer and permanently removes the conversation");
    expect(api.deleteConversation).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete conversation Old draft" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete everything" }));
    await waitFor(() => expect(api.deleteConversation).toHaveBeenCalledWith("old-thread", "personal-user"));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Delete conversation Old draft" })).not.toBeInTheDocument());
    expect(window.localStorage.getItem("evidencegraph:thread_id")).toBe("personal-thread");
  });

  it("starts a clean chat after deleting the currently open conversation", async () => {
    let rows = [{ thread_id: "personal-thread", agent_id: "research-assistant", title: "Current draft", document_id: documentId, updated_at: "2026-10-05" }];
    vi.mocked(api.listThreads).mockImplementation(async () => ({ threads: rows }));
    vi.mocked(api.getHistory).mockImplementation(async (id) => ({
      conversation: id === "personal-thread" ? workspace(id, "personal-user") : null,
      messages: [],
    }));
    vi.mocked(api.deleteConversation).mockImplementation(async () => { rows = []; });
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete conversation Current draft" }));
    expect(screen.getByRole("alertdialog")).toHaveTextContent("permanently removes the conversation, PDF, extracted text, graph, and evidence");
    fireEvent.click(screen.getByRole("button", { name: "Delete everything" }));
    await waitFor(() => expect(api.deleteConversation).toHaveBeenCalledWith("personal-thread", "personal-user"));
    await waitFor(() => expect(window.localStorage.getItem("evidencegraph:thread_id")).not.toBe("personal-thread"));
    expect(new URLSearchParams(window.location.search).get("document_id")).toBeNull();
    expect(screen.queryByRole("button", { name: "Delete conversation Current draft" })).not.toBeInTheDocument();
  });

  it("keeps the confirmation open with an actionable error when deletion fails", async () => {
    vi.mocked(api.listThreads).mockResolvedValue({ threads: [
      { thread_id: "old-thread", agent_id: "research-assistant", title: "Old draft", updated_at: "2026-10-04" },
    ] });
    vi.mocked(api.deleteConversation).mockRejectedValue(new Error("Network unavailable"));
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Delete conversation Old draft" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete everything" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Network unavailable");
    expect(screen.getByRole("alertdialog")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete conversation Old draft" })).toBeInTheDocument();
  });

  it("uses the server identity and omits the old URL identity in strict mode", async () => {
    vi.mocked(api.getSession).mockResolvedValue({ user_id: "server-guest", identity_enforced: true });
    render(<App />);

    await waitFor(() => expect(api.listThreads).toHaveBeenCalledWith("server-guest"));
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("user_id")).toBeNull());
    expect(window.localStorage.getItem("evidencegraph:user_id")).toBeNull();
    expect(window.localStorage.getItem("evidencegraph:thread_id:server-guest")).toBeTruthy();
    expect(api.getHistory).toHaveBeenCalledWith(expect.any(String), "server-guest");
  });
});
