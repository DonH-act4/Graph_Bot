export interface SsePayload {
  type: "token" | "message" | "error";
  content: unknown;
}

export function decodeSseEvent(event: string): SsePayload | "done" | null {
  const data = event
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data) return null;
  if (data.trim() === "[DONE]") return "done";
  return JSON.parse(data) as SsePayload;
}
