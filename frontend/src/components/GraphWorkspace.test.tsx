import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { forwardRef, useImperativeHandle } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getEvidenceBlock } from "../api";
import type { EvidenceAttachment, GraphArtifact, GraphNode } from "../types";
import { GraphWorkspace } from "./GraphWorkspace";

const graphMethods = vi.hoisted(() => ({
  d3Force: vi.fn(() => ({ distance: vi.fn(), strength: vi.fn() })),
  d3ReheatSimulation: vi.fn(), zoomToFit: vi.fn(), centerAt: vi.fn(),
}));

interface MockGraphProps {
  width: number;
  height: number;
  graphData: { nodes: Array<{ id: string; item: GraphNode }> };
  onNodeClick: (node: { id: string; item: GraphNode }) => void;
}
vi.mock("react-force-graph-2d", () => ({ default: forwardRef<unknown, MockGraphProps>((props, ref) => {
  useImperativeHandle(ref, () => graphMethods);
  return <div data-testid="force-graph" data-width={props.width} data-height={props.height}>
    <button onClick={() => props.onNodeClick(props.graphData.nodes[0])}>Select paper node</button>
  </div>;
}) }));
vi.mock("../api", () => ({ getEvidenceBlock: vi.fn() }));

const documentId = "a".repeat(64);
const blockId = "blk_" + "b".repeat(24);
const artifact: GraphArtifact = { graph_version: 1, document_sha256: documentId, extractor_name: "ollama", extractor_version: "test", warnings: [],
  graph: { schema_version: "3", nodes: [{ node_id: "paper", node_type: "paper", name: "Example paper", domain_type: null,
    evidence: [{ source_sha256: documentId, block_id: blockId }], value: null, unit: null, uncertainty: null, conditions: null }], relations: [] } };

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("ResizeObserver", class {
    constructor(private callback: (entries: Array<{ contentRect: { width: number; height: number } }>) => void) {}
    observe() { this.callback([{ contentRect: { width: 1050, height: 640 } }]); }
    disconnect() {}
  });
  vi.mocked(getEvidenceBlock).mockResolvedValue({ block_id: blockId, source_ref: "#/texts/0", label: "text", text: "The original source passage.", locations: [{ page_number: 3, bounding_box: { left: 0, top: 0, right: 1, bottom: 1, coordinate_origin: "TOPLEFT" }, character_start: 0, character_end: 28 }] });
});
afterEach(() => vi.unstubAllGlobals());

describe("graph evidence workspace", () => {
  it("measures the real canvas when a previously closed graph is opened", async () => {
    const props = { artifact, selectedEvidence: [], onSelectedEvidenceChange: vi.fn(), onClose: vi.fn() };
    const { rerender } = render(<GraphWorkspace {...props} open={false} />);
    expect(screen.queryByTestId("force-graph")).not.toBeInTheDocument();
    rerender(<GraphWorkspace {...props} open />);
    await waitFor(() => expect(screen.getByTestId("force-graph")).toHaveAttribute("data-width", "1050"));
    expect(screen.getByTestId("force-graph")).toHaveAttribute("data-height", "640");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(props.onClose).toHaveBeenCalled();
  });

  it("allows exact evidence browsing in a read-only sample without selection controls", async () => {
    render(<GraphWorkspace artifact={artifact} open readOnly selectedEvidence={[]} onSelectedEvidenceChange={vi.fn()} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Select paper node" }));
    await screen.findByText("The original source passage.");
    expect(getEvidenceBlock).toHaveBeenCalledWith(documentId, blockId);
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Open source PDF/ })).toHaveAttribute("href", `/api/papers/${documentId}/source#page=3`);
  });

  it("embeds the same interactive evidence graph in a page without modal navigation", async () => {
    const onClose = vi.fn();
    render(<GraphWorkspace artifact={artifact} open inline readOnly selectedEvidence={[]} onSelectedEvidenceChange={vi.fn()} onClose={onClose} />);
    expect(screen.getByRole("region", { name: "Knowledge graph" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Knowledge graph" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Close graph" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Select paper node" }));
    await screen.findByText("The original source passage.");
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).not.toHaveBeenCalled();
  });

  it("prevents adding a thirteenth source while retaining existing selections", async () => {
    const selected: EvidenceAttachment[] = Array.from({ length: 12 }, (_, index) => ({ documentId, blockId: `blk_${index}`, pages: [1], text: `Source ${index}`, sourceLabel: "Source" }));
    const change = vi.fn();
    render(<GraphWorkspace artifact={artifact} open selectedEvidence={selected} onSelectedEvidenceChange={change} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Select paper node" }));
    await screen.findByText("The original source passage.");
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(screen.getByText("12/12 sources selected")).toBeInTheDocument();
    expect(change).not.toHaveBeenCalled();
  });
});
