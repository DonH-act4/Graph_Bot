import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Composer } from "./Composer";
import type { EvidenceAttachment } from "../types";

const evidence: EvidenceAttachment = {
  documentId: "a".repeat(64),
  blockId: "blk_123",
  pages: [3],
  text: "A source-located result.",
  sourceLabel: "reports",
};

function renderComposer(overrides: Partial<React.ComponentProps<typeof Composer>> = {}) {
  const props: React.ComponentProps<typeof Composer> = {
    value: "Explain this",
    evidence: [evidence],
    disabled: false,
    canOpenGraph: true,
    onChange: vi.fn(),
    onEvidenceChange: vi.fn(),
    onOpenGraph: vi.fn(),
    onSubmit: vi.fn(),
    ...overrides,
  };
  render(<Composer {...props} />);
  return props;
}

describe("Composer", () => {
  it("shows and removes a selected evidence chip", () => {
    const props = renderComposer();
    expect(screen.getByText(/A source-located result/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Remove evidence/ }));
    expect(props.onEvidenceChange).toHaveBeenCalledWith([]);
  });

  it("submits on Enter but preserves Shift+Enter", () => {
    const props = renderComposer();
    const input = screen.getByRole("textbox");
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(props.onSubmit).not.toHaveBeenCalled();
    fireEvent.keyDown(input, { key: "Enter", shiftKey: false });
    expect(props.onSubmit).toHaveBeenCalledOnce();
  });

  it("disables graph evidence selection when no graph is ready", () => {
    renderComposer({ canOpenGraph: false, evidence: [] });
    expect(screen.getByRole("button", { name: /Add evidence/ })).toBeDisabled();
  });
});
