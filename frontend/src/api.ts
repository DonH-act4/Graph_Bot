import { decodeSseEvent } from "./lib/sse";
import type {
  ChatHistory,
  ChatMessage,
  ConversationState,
  EvidenceAttachment,
  EvidenceBlock,
  GraphArtifact,
  GraphRecord,
  PaperConfiguration,
  PaperRecord,
  BrowserSession,
  ShowcaseRecord,
  ThreadList,
} from "./types";

const API_ROOT = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the bounded status-only fallback for non-JSON errors.
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function uploadPaper(file: File): Promise<PaperRecord> {
  return request("/papers", {
    method: "POST",
    headers: { "Content-Type": "application/pdf" },
    body: file,
  });
}

export function getPaperConfiguration(): Promise<PaperConfiguration> {
  return request("/papers/configuration");
}

export function getSession(): Promise<BrowserSession> {
  return request("/session");
}

export function authenticate(mode: "login" | "register", username: string, password: string): Promise<{ user_id: string }> {
  return request(`/auth/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
}

export function logout(): Promise<void> {
  return request("/auth/logout", { method: "POST" });
}

export function getPaper(documentId: string): Promise<PaperRecord> {
  return request(`/papers/${documentId}`);
}

export function requestGraph(documentId: string, model?: string): Promise<GraphRecord> {
  return request(`/papers/${documentId}/graph`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model || null }),
  });
}

export function rebuildGraph(documentId: string, model?: string): Promise<GraphRecord> {
  return request(`/papers/${documentId}/graph/rebuild`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model || null }),
  });
}

export function getGraphStatus(documentId: string): Promise<GraphRecord> {
  return request(`/papers/${documentId}/graph/status`);
}

export function getGraph(documentId: string): Promise<GraphArtifact> {
  return request(`/papers/${documentId}/graph`);
}

export function getEvidenceBlock(
  documentId: string,
  blockId: string,
): Promise<EvidenceBlock> {
  return request(`/papers/${documentId}/blocks/${blockId}`);
}

export function listThreads(userId: string): Promise<ThreadList> {
  const params = new URLSearchParams({ user_id: userId, limit: "30" });
  return request(`/threads?${params.toString()}`);
}

export function deleteConversation(threadId: string, userId: string): Promise<void> {
  const params = new URLSearchParams({ user_id: userId });
  return request(`/conversations/${encodeURIComponent(threadId)}?${params.toString()}`, {
    method: "DELETE",
  });
}

export function getHistory(threadId: string, userId?: string): Promise<ChatHistory> {
  return request("/history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, ...(userId ? { user_id: userId } : {}) }),
  });
}

export function getShowcase(): Promise<ShowcaseRecord | null> {
  return request("/showcase");
}

export function saveConversation(
  threadId: string,
  state: Pick<ConversationState, "user_id" | "document_id" | "document_name" | "selected_block_ids" | "draft_message">,
  options?: { keepalive?: boolean },
): Promise<ConversationState> {
  return request(`/conversations/${threadId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
    ...(options?.keepalive ? { keepalive: true } : {}),
  });
}

interface StreamChatOptions {
  message: string;
  threadId: string;
  userId: string;
  evidence: EvidenceAttachment[];
  documentId?: string | null;
  onToken: (token: string) => void;
  onMessage: (message: ChatMessage) => void;
}

export async function streamChat(options: StreamChatOptions): Promise<void> {
  const documentIds = new Set(options.evidence.map((item) => item.documentId));
  if (documentIds.size > 1) {
    throw new Error("Selected evidence must come from one paper per question.");
  }
  const evidenceContext = options.evidence.length
    ? {
        document_id: options.evidence[0].documentId,
        block_ids: options.evidence.map((item) => item.blockId),
      }
    : undefined;
  const response = await fetch(`${API_ROOT}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: options.message,
      thread_id: options.threadId,
      user_id: options.userId,
      stream_tokens: true,
      evidence_context: evidenceContext,
      document_id: options.documentId ?? options.evidence[0]?.documentId ?? null,
    }),
  });
  if (!response.ok || !response.body) {
    let message = `Chat request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Preserve the bounded fallback.
    }
    throw new ApiError(message, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const payload = decodeSseEvent(event);
      if (!payload || payload === "done") continue;
      if (payload.type === "token" && typeof payload.content === "string") {
        options.onToken(payload.content);
      } else if (payload.type === "message") {
        options.onMessage(payload.content as ChatMessage);
      } else if (payload.type === "error") {
        throw new Error(String(payload.content));
      }
    }
    if (done) break;
  }
}
