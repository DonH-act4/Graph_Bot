import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getGraph, getHistory } from "../api";
import type { GraphArtifact, ShowcaseRecord } from "../types";
import { Showcase } from "./Showcase";

vi.mock("../api", () => ({ getGraph: vi.fn(), getHistory: vi.fn() }));
vi.mock("./GraphWorkspace", () => ({ GraphWorkspace: () => <div role="region" aria-label="Tutorial graph" /> }));

const featuredId = "b0c74a6f6dc80e8c29cba07ba56a1b2de213bc9dbc41604a8e5081507a35fcc3";
const sample: ShowcaseRecord = {
  title: "Water-network leak detection", description: "Saved sample",
  document_id: featuredId, thread_id: "sample-thread", user_id: "sample-owner", model: "ollama/gpt-oss:20b",
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(getGraph).mockResolvedValue({
    graph_version: 1, document_sha256: featuredId, extractor_name: "ollama", extractor_version: "test", warnings: [],
    graph: { schema_version: "3", nodes: [], relations: [] },
  } satisfies GraphArtifact);
  vi.mocked(getHistory).mockResolvedValue({ messages: [
    { type: "human", content: "What is the contribution?" }, { type: "ai", content: "A transfer-learning framework." },
    { type: "human", content: "What changed in accuracy?" }, { type: "ai", content: "Accuracy rose in the saved result." },
    { type: "human", content: "How is it explained?" }, { type: "ai", content: "With Integrated Gradients." },
  ] });
});

describe("one-page tutorial", () => {
  it("shows the real saved graph and all saved question/answer pairs without navigation", async () => {
    render(<Showcase sample={sample} loading={false} onUpload={vi.fn()} />);
    expect(await screen.findByRole("region", { name: "Tutorial graph" })).toBeInTheDocument();
    expect(screen.getAllByText(/Question 0[1-3]/)).toHaveLength(3);
    expect(screen.getByText("A transfer-learning framework.")).toBeInTheDocument();
    expect(screen.getByText("Accuracy rose in the saved result.")).toBeInTheDocument();
    expect(screen.getByText("With Integrated Gradients.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /DOI 10\.1016\/j\.watres\.2026\.126464/ })).toHaveAttribute("href", "https://doi.org/10.1016/j.watres.2026.126464");
    expect(screen.getByText(/one of my MPhil papers published in the top-tier journal Water Research/)).toBeInTheDocument();
    expect(screen.getByText("My MPhil research · Water Research")).toBeInTheDocument();
    expect(screen.getByText(/select source cards to include their passages in your next question and answer/)).toBeInTheDocument();
    expect(screen.getByText(/select cards to include their original passages as context for the next answer/)).toBeInTheDocument();
    expect(getHistory).toHaveBeenCalledWith(sample.thread_id, sample.user_id);
    expect(screen.queryByRole("button", { name: /Open sample conversation/ })).not.toBeInTheDocument();
  });

  it("does not attribute the featured paper's DOI to a different saved sample", async () => {
    render(<Showcase sample={{ ...sample, document_id: "a".repeat(64) }} loading={false} onUpload={vi.fn()} />);
    await screen.findByRole("region", { name: "Tutorial graph" });
    expect(screen.queryByRole("link", { name: /DOI/ })).not.toBeInTheDocument();
    expect(screen.getByText("Real saved research paper")).toBeInTheDocument();
  });
});
