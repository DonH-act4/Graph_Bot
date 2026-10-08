import { describe, expect, it } from "vitest";

import { dedupeEvidence, historyAttachments, readableCitations, toAttachment } from "./evidence";
import type { ChatMessage, EvidenceAttachment, EvidenceBlock } from "../types";

const block: EvidenceBlock = {
  block_id: "blk_123",
  source_ref: "#/texts/1",
  label: "text",
  text: "A source-located result.",
  locations: [
    {
      page_number: 3,
      bounding_box: { left: 0, top: 0, right: 1, bottom: 1, coordinate_origin: "bottom-left" },
      character_start: 0,
      character_end: 10,
    },
    {
      page_number: 2,
      bounding_box: { left: 0, top: 0, right: 1, bottom: 1, coordinate_origin: "bottom-left" },
      character_start: 0,
      character_end: 10,
    },
    {
      page_number: 3,
      bounding_box: { left: 0, top: 0, right: 1, bottom: 1, coordinate_origin: "bottom-left" },
      character_start: 0,
      character_end: 10,
    },
  ],
};

describe("evidence helpers", () => {
  it("normalizes pages and removes duplicate attachments", () => {
    const attachment = toAttachment("a".repeat(64), block, "Result relation");
    expect(attachment.pages).toEqual([2, 3]);
    expect(dedupeEvidence([attachment, { ...attachment }])).toEqual([attachment]);
  });

  it("restores evidence attached to a historical message", () => {
    const message: ChatMessage = {
      type: "human",
      content: "Explain this",
      custom_data: {
        evidence_context: {
          document_id: "a".repeat(64),
          blocks: [{ block_id: "blk_123", pages: [2], text: "Source text" }],
        },
      },
    };
    const restored: EvidenceAttachment[] = historyAttachments(message);
    expect(restored[0]).toMatchObject({ blockId: "blk_123", pages: [2], text: "Source text" });
  });

  it("links model citation variants using the actual source pages", () => {
    const id = "1".repeat(24);
    const source = { ...toAttachment("a".repeat(64), block, "Result"), blockId: `blk_${id}` };
    const link = `[Page 2, 3](/api/papers/${"a".repeat(64)}/source#page=2)`;
    for (const citation of [`[Page 99 · blk_${id}]`, `【blk_${id}】`, `【${id}】`, `[blk_${id}]`]) {
      expect(readableCitations(`Result ${citation}`, [source])).toBe(`Result ${link}`);
    }
  });

  it("does not invent a link for unknown citations or sources without pages", () => {
    const content = `Result 【blk_${"f".repeat(24)}】`;
    expect(readableCitations(content, [])).toBe(content);
    const source = { ...toAttachment("a".repeat(64), block, "Result"), blockId: `blk_${"f".repeat(24)}`, pages: [] };
    expect(readableCitations(content, [source])).toBe(content);
  });

  it("recognizes Unicode spaces in a live selected-source answer", () => {
    const id = "fcc48a4ab507d73dfeda34fc";
    const source = { ...toAttachment("a".repeat(64), block, "Selected evidence"), blockId: `blk_${id}`, pages: [3] };
    for (const space of ["\u202f", "\u00a0", "\t"]) {
      expect(readableCitations(`[Page${space}3 · blk_${id}]`, [source]))
        .toBe(`[Page 3](/api/papers/${"a".repeat(64)}/source#page=3)`);
    }
  });
});
