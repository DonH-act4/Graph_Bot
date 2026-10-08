import { afterEach, describe, expect, it, vi } from "vitest";

import { deleteConversation, getPaperConfiguration, getSession, rebuildGraph, requestGraph } from "./api";

const documentId = "a".repeat(64);

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("paper configuration and graph model requests", () => {
  it("loads the server-issued browser session", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      user_id: "server-guest", identity_enforced: true,
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(getSession()).resolves.toEqual({ user_id: "server-guest", identity_enforced: true });
    expect(fetchMock).toHaveBeenCalledWith("/api/session", undefined);
  });
  it("accepts an empty 204 response when deleting a conversation", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(deleteConversation("thread one", "user")).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("/api/conversations/thread%20one?user_id=user", { method: "DELETE" });
  });
  it("loads the server-owned upload and model configuration", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          max_pdf_bytes: 25 * 1024 * 1024,
          graph_models: ["gemini-fast", "gemini-pro"],
          default_graph_model: "gemini-fast",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const configuration = await getPaperConfiguration();

    expect(configuration.max_pdf_bytes).toBe(25 * 1024 * 1024);
    expect(fetchMock).toHaveBeenCalledWith("/api/papers/configuration", undefined);
  });

  it.each([
    ["request", requestGraph, `/api/papers/${documentId}/graph`],
    ["rebuild", rebuildGraph, `/api/papers/${documentId}/graph/rebuild`],
  ])("sends the selected model for %s", async (_name, action, url) => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          document_id: documentId,
          state: "queued",
          requested_model: "gemini-pro",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await action(documentId, "gemini-pro");

    expect(fetchMock).toHaveBeenCalledWith(
      url,
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model: "gemini-pro" }),
      }),
    );
  });
});
