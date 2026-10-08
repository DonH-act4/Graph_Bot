import { describe, expect, it } from "vitest";

import { decodeSseEvent } from "./sse";

describe("decodeSseEvent", () => {
  it("decodes token events", () => {
    expect(decodeSseEvent('data: {"type":"token","content":"hello"}')).toEqual({
      type: "token",
      content: "hello",
    });
  });

  it("recognizes completion and empty heartbeat events", () => {
    expect(decodeSseEvent("data: [DONE]")).toBe("done");
    expect(decodeSseEvent(": keep-alive")).toBeNull();
  });
});
