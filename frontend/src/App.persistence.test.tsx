import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import * as api from "./api";
import App from "./App";
import type { ConversationState, GraphArtifact } from "./types";

vi.mock("./api", async (original) => ({
  ...await original<typeof import("./api")>(),
  getPaperConfiguration: vi.fn(), getHistory: vi.fn(), listThreads: vi.fn(),
  getSession: vi.fn(),
  authenticate: vi.fn(), logout: vi.fn(),
  getPaper: vi.fn(), getGraph: vi.fn(), getGraphStatus: vi.fn(), getEvidenceBlock: vi.fn(),
  saveConversation: vi.fn(), getShowcase: vi.fn(), requestGraph: vi.fn(), streamChat: vi.fn(),
}));
vi.mock("./components/GraphWorkspace", () => ({ GraphWorkspace: ({ inline }: { inline?: boolean }) => inline ? <div role="region" aria-label="Tutorial graph" /> : null }));

const userId = "11111111-1111-4111-8111-111111111111";
const threadId = "22222222-2222-4222-8222-222222222222";
const documentId = "a".repeat(64);
const blockId = "blk_" + "b".repeat(24);
const state: ConversationState = {
  thread_id: threadId, user_id: userId, agent_id: "research-assistant", title: "Saved research conversation",
  document_id: documentId, document_name: "research.pdf", selected_block_ids: [blockId],
  draft_message: "Explain the findings", created_at: "2026-10-04T10:00:00Z", updated_at: "2026-10-04T10:00:00Z",
};
const artifact: GraphArtifact = {
  graph_version: 1, document_sha256: documentId, extractor_name: "ollama", extractor_version: "test", warnings: [],
  graph: { schema_version: "3", nodes: [{ node_id: "paper", node_type: "paper", name: "Research paper", domain_type: null,
    evidence: [], value: null, unit: null, uncertainty: null, conditions: null }], relations: [] },
};

beforeEach(() => {
  vi.resetAllMocks();
  window.history.replaceState(null, "", `/?user_id=${userId}&thread_id=${threadId}`);
  vi.mocked(api.getPaperConfiguration).mockResolvedValue({ max_pdf_bytes: 25 * 1024 * 1024, graph_models: ["ollama/gpt-oss:20b"], default_graph_model: "ollama/gpt-oss:20b" });
  vi.mocked(api.getSession).mockResolvedValue({ user_id: userId, identity_enforced: false });
  vi.mocked(api.listThreads).mockResolvedValue({ threads: [] });
  vi.mocked(api.getHistory).mockResolvedValue({ messages: [], conversation: null });
  vi.mocked(api.getPaper).mockResolvedValue({ document_id: documentId, state: "ready", page_count: 10, block_count: 120, stage: "complete", progress_percent: 100, error: null });
  vi.mocked(api.getGraph).mockResolvedValue(artifact);
  vi.mocked(api.getGraphStatus).mockResolvedValue({ document_id: documentId, state: "ready", node_count: 1, relation_count: 0,
    requested_model: "ollama/gpt-oss:20b", current_version: 1, stage: "complete", progress_percent: 100, progress_detail: null, warnings: [], error: null });
  vi.mocked(api.getEvidenceBlock).mockResolvedValue({ block_id: blockId, source_ref: "#/texts/1", label: "text", text: "The exact research finding from the paper.", locations: [{ page_number: 2, bounding_box: { left: 0, top: 0, right: 1, bottom: 1, coordinate_origin: "TOPLEFT" }, character_start: 0, character_end: 40 }] });
  vi.mocked(api.saveConversation).mockResolvedValue(state);
});

afterEach(() => { window.localStorage.clear(); window.history.replaceState(null, "", "/"); });

describe("persisted research conversations", () => {
  it("explains a legacy paper link without deleting its saved files", async () => {
    vi.mocked(api.getSession).mockResolvedValue({ user_id: "new-guest", identity_enforced: true, chat_requires_login: true, authenticated: false });
    vi.mocked(api.getPaper).mockRejectedValue(new api.ApiError("Paper not found", 404));
    window.history.replaceState(null, "", `/?document_id=${documentId}`);
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Paper needs to be reattached");
    expect(screen.getByRole("alert")).toHaveTextContent("saved local files were not deleted");
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document_id")).toBeNull());
  });

  it("keeps graph browsing open to guests and restores a question after account creation", async () => {
    vi.mocked(api.getSession)
      .mockResolvedValueOnce({ user_id: "guest-1", identity_enforced: true, chat_requires_login: true, authenticated: false })
      .mockResolvedValueOnce({ user_id: userId, identity_enforced: true, chat_requires_login: true, authenticated: true });
    vi.mocked(api.authenticate).mockResolvedValue({ user_id: userId });
    window.history.replaceState(null, "", `/?document_id=${documentId}`);
    render(<App />);
    await screen.findByRole("button", { name: /Sign in to ask questions/ });
    expect(api.getHistory).not.toHaveBeenCalled();
    expect(api.listThreads).not.toHaveBeenCalled();
    await screen.findByRole("button", { name: /Open graph/ });
    fireEvent.change(screen.getByRole("textbox", { name: "Message" }), { target: { value: "How was the model evaluated?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByRole("dialog", { name: "Sign in to ask the paper" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "New here? Create an account" }));
    fireEvent.change(screen.getByLabelText("Username"), { target: { value: "researcher" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "long-test-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Create account" }));
    await waitFor(() => expect(api.authenticate).toHaveBeenCalledWith("register", "researcher", "long-test-password"));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("How was the model evaluated?"));
    expect(api.getHistory).toHaveBeenCalled();
    expect(api.streamChat).not.toHaveBeenCalled();
  });

  it("restores the paper, draft, and selected evidence before saving them", async () => {
    vi.mocked(api.getHistory).mockResolvedValue({ messages: [], conversation: state });
    render(<App />);
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("Explain the findings"));
    await screen.findByRole("button", { name: "Remove evidence from page 2" });
    expect(api.getEvidenceBlock).toHaveBeenCalledWith(documentId, blockId);
    await waitFor(() => expect(api.saveConversation).toHaveBeenCalledWith(threadId, expect.objectContaining({
      user_id: userId, document_id: documentId, selected_block_ids: [blockId], draft_message: "Explain the findings",
    })));
    expect(new URLSearchParams(window.location.search).get("document_id")).toBe(documentId);
  });

  it("starts a clean new conversation instead of carrying the previous paper and sources", async () => {
    vi.mocked(api.getHistory).mockImplementation(async (id) => ({ messages: [], conversation: id === threadId ? state : null }));
    render(<App />);
    await screen.findByRole("button", { name: "Remove evidence from page 2" });
    fireEvent.click(screen.getByRole("button", { name: /New chat/ }));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue(""));
    expect(screen.queryByRole("button", { name: "Remove evidence from page 2" })).not.toBeInTheDocument();
    expect(new URLSearchParams(window.location.search).get("document_id")).toBeNull();
    expect(screen.getByRole("button", { name: /Open graph/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Open graph/ })).not.toHaveClass("is-ready");
  });

  it("flushes the latest draft when the page is being closed", async () => {
    vi.mocked(api.getHistory).mockResolvedValue({ messages: [], conversation: { ...state, selected_block_ids: [], draft_message: "" } });
    render(<App />);
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toBeEnabled());
    fireEvent.change(screen.getByRole("textbox", { name: "Message" }), { target: { value: "A just-typed question" } });
    fireEvent(window, new Event("pagehide"));
    expect(api.saveConversation).toHaveBeenCalledWith(threadId, expect.objectContaining({ draft_message: "A just-typed question", document_id: documentId }), { keepalive: true });
  });

  it("shows the saved tutorial read-only without replacing the personal identity", async () => {
    const sample = { title: "My published research", description: "A real paper and saved conversation", user_id: "demo-owner", thread_id: "demo-thread", document_id: documentId, model: "ollama/gpt-oss:20b" };
    vi.mocked(api.getShowcase).mockResolvedValue(sample);
    vi.mocked(api.getHistory).mockImplementation(async (id) => ({ messages: id === sample.thread_id ? [{ type: "human", content: "What did the paper find?" }, { type: "ai", content: "The saved sample answer." }] : [], conversation: id === sample.thread_id ? { ...state, thread_id: sample.thread_id, user_id: sample.user_id, selected_block_ids: [], draft_message: "" } : null }));
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /Interactive tutorial/ }));
    await screen.findByText("The saved sample answer.");
    expect(await screen.findByRole("region", { name: "Tutorial graph" })).toBeInTheDocument();
    expect(api.getHistory).toHaveBeenCalledWith(sample.thread_id, sample.user_id);
    expect(screen.queryByRole("textbox", { name: "Message" })).not.toBeInTheDocument();
    expect(window.localStorage.getItem("evidencegraph:user_id")).toBe(userId);
    expect(api.saveConversation).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Back to chat/ }));
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Message" })).toBeEnabled());
    expect(new URLSearchParams(window.location.search).get("thread_id")).toBe(threadId);
    expect(api.saveConversation).not.toHaveBeenCalled();
  });

  it("reloads a demo URL without replacing the last personal thread in storage", async () => {
    window.localStorage.setItem("evidencegraph:thread_id", threadId);
    const sample = { title: "My published research", description: "Real research", user_id: "demo-owner", thread_id: "demo-thread", document_id: documentId, model: "ollama/gpt-oss:20b" };
    window.history.replaceState(null, "", `/?user_id=${userId}&thread_id=demo-thread&view=demo`);
    vi.mocked(api.getShowcase).mockResolvedValue(sample);
    vi.mocked(api.getHistory).mockResolvedValue({ messages: [{ type: "ai", content: "Restored demo answer." }], conversation: { ...state, thread_id: sample.thread_id, user_id: sample.user_id, selected_block_ids: [], draft_message: "" } });
    render(<App />);
    await screen.findByText("Restored demo answer.");
    expect(api.getHistory).toHaveBeenCalledWith(sample.thread_id, sample.user_id);
    expect(window.localStorage.getItem("evidencegraph:thread_id")).toBe(threadId);
    expect(api.saveConversation).not.toHaveBeenCalled();
  });
});
