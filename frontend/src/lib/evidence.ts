import type { ChatMessage, EvidenceAttachment, EvidenceBlock } from "../types";

export function evidenceKey(
  evidence: Pick<EvidenceAttachment, "documentId" | "blockId">,
): string {
  return `${evidence.documentId}:${evidence.blockId}`;
}

export function toAttachment(
  documentId: string,
  block: EvidenceBlock,
  sourceLabel: string,
): EvidenceAttachment {
  return {
    documentId,
    blockId: block.block_id,
    pages: [...new Set(block.locations.map((location) => location.page_number))].sort(
      (left, right) => left - right,
    ),
    text: block.text,
    sourceLabel,
  };
}

export function dedupeEvidence(
  evidence: EvidenceAttachment[],
): EvidenceAttachment[] {
  return [...new Map(evidence.map((item) => [evidenceKey(item), item])).values()];
}

export function historyAttachments(message: ChatMessage): EvidenceAttachment[] {
  const context = message.custom_data?.evidence_context;
  if (!context) return [];
  return context.blocks.map((block) => ({
    documentId: context.document_id,
    blockId: block.block_id,
    pages: block.pages,
    text: block.text,
    sourceLabel: context.source_mode === "automatic" ? "Evidence found in the paper" : "Selected evidence",
  }));
}

/** Link only exact cited IDs found in this turn's evidence; never guess a page. */
export function readableCitations(content: string, sources: EvidenceAttachment[]): string {
  const byId = new Map(sources.map((source) => [source.blockId, source]));
  return content.replace(
    /\[(?:Page\s+\d+\s*[·|:]\s*)?(?:blk_)?([a-f0-9]{24})\]|【(?:blk_)?([a-f0-9]{24})】/gi,
    (original: string, squareId: string | undefined, wideId: string | undefined) => {
      const source = byId.get(`blk_${(squareId ?? wideId ?? "").toLowerCase()}`);
      if (!source || !source.pages.length) return original;
      return `[Page ${source.pages.join(", ")}](/api/papers/${source.documentId}/source#page=${source.pages[0]})`;
    },
  );
}
