import { lazy, Suspense, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { getGraph, getHistory } from "../api";
import { historyAttachments, readableCitations } from "../lib/evidence";
import type { ChatMessage, GraphArtifact, ShowcaseRecord } from "../types";

const GraphWorkspace = lazy(() => import("./GraphWorkspace").then((module) => ({ default: module.GraphWorkspace })));

const PAPER_DOI = "10.1016/j.watres.2026.126464";
const FEATURED_DOCUMENT_ID = "b0c74a6f6dc80e8c29cba07ba56a1b2de213bc9dbc41604a8e5081507a35fcc3";

interface ShowcaseProps {
  sample: ShowcaseRecord | null;
  loading: boolean;
  onUpload: () => void;
}

interface Exchange {
  question: ChatMessage;
  answer: ChatMessage;
}

function savedExchanges(messages: ChatMessage[]): Exchange[] {
  const exchanges: Exchange[] = [];
  let question: ChatMessage | null = null;
  for (const message of messages) {
    if (message.type === "human") question = message;
    if (message.type === "ai" && question) {
      exchanges.push({ question, answer: message });
      question = null;
    }
  }
  return exchanges;
}

export function Showcase({ sample, loading, onUpload }: ShowcaseProps) {
  const isFeaturedPaper = sample?.document_id === FEATURED_DOCUMENT_ID;
  const [graph, setGraph] = useState<GraphArtifact | null>(null);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [sampleLoading, setSampleLoading] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);

  useEffect(() => {
    setGraph(null);
    setExchanges([]);
    setSampleError(null);
    if (!sample) return;
    let cancelled = false;
    setSampleLoading(true);
    void Promise.all([
      getGraph(sample.document_id),
      getHistory(sample.thread_id, sample.user_id),
    ]).then(([artifact, history]) => {
      if (cancelled) return;
      setGraph(artifact);
      setExchanges(savedExchanges(history.messages));
    }).catch((error) => {
      if (!cancelled) setSampleError(error instanceof Error ? error.message : "The saved tutorial could not be loaded.");
    }).finally(() => {
      if (!cancelled) setSampleLoading(false);
    });
    return () => { cancelled = true; };
  }, [sample]);

  return (
    <section className="showcase-page">
      <div className="showcase-intro">
        <span className="eyebrow">Interactive tutorial · real research</span>
        <h1>See the paper.<br /><em>Follow the evidence.</em></h1>
        <p>Using one of my MPhil papers published in the top-tier journal Water Research as an example, explore its generated graph, inspect original evidence cards, and read the saved questions and answers below.</p>
      </div>

      <div className="showcase-feature">
        <div className="showcase-feature-copy">
          <span className="sample-badge">{isFeaturedPaper ? "My MPhil research · Water Research" : "Real saved research paper"}</span>
          <h2>{sample?.title ?? (loading ? "Loading the research paper…" : "Research tutorial unavailable")}</h2>
          <p>{isFeaturedPaper
            ? "Published in Water Research, this study uses hydraulic transient signals and simulation-to-field transfer learning to diagnose leaks in water distribution networks when field data are limited. The graph and answers below are generated from its saved PDF."
            : sample?.description ?? "The tutorial will appear when its saved paper and conversation are available."}</p>
          {sample && <div className="showcase-meta">
            {isFeaturedPaper && <><span>Water Research · 2026</span><a href={`https://doi.org/${PAPER_DOI}`} target="_blank" rel="noreferrer">DOI {PAPER_DOI} ↗</a></>}
            <span>Saved graph · {sample.model}</span>
          </div>}
        </div>
        <div className="showcase-guide">
          <span className="eyebrow">How to explore</span>
          <ol>
            <li><b>Move the graph.</b> Drag nodes and see their relationships respond.</li>
            <li><b>Touch a connection.</b> Click a node or line to reveal exact PDF passages.</li>
            <li><b>Bring evidence into chat.</b> In a research conversation, select source cards to include their passages in your next question and answer.</li>
            <li><b>Read the conversation.</b> The saved questions and model answers sit right below.</li>
          </ol>
        </div>
      </div>

      {(loading || sampleLoading) && <p className="showcase-state" role="status">Loading the saved graph and conversation…</p>}
      {!loading && !sample && <p className="showcase-state" role="alert">No tutorial has been published for this local demo yet.</p>}
      {sampleError && <p className="showcase-state error" role="alert">The tutorial could not be loaded: {sampleError}</p>}

      {graph && <div className="showcase-graph-section">
        <div className="showcase-section-heading">
          <div><span className="eyebrow">01 · Explore</span><h2>The generated knowledge graph</h2></div>
          <p>{graph.graph.nodes.length} entities · {graph.graph.relations.length} relationships · click a line or node for source evidence</p>
        </div>
        <Suspense fallback={<p className="showcase-state" role="status">Opening the interactive graph…</p>}>
          <GraphWorkspace artifact={graph} open inline readOnly selectedEvidence={[]} onSelectedEvidenceChange={() => {}} onClose={() => {}} />
        </Suspense>
        <p className="showcase-caveat">This tutorial lets you inspect the source cards. In a research conversation, you can select cards to include their original passages as context for the next answer. This model-generated graph is an overview, not an exhaustive or independently verified reading of the paper.</p>
      </div>}

      {exchanges.length > 0 && <div className="showcase-qa-section">
        <div className="showcase-section-heading">
          <div><span className="eyebrow">02 · Ask</span><h2>Questions already asked about this paper</h2></div>
          <p>Real saved answers · citations open the original PDF page</p>
        </div>
        <div className="showcase-exchanges">
          {exchanges.map(({ question, answer }, index) => {
            const sources = historyAttachments(question);
            return <article className="showcase-exchange" key={`${index}-${question.content}`}>
              <div className="showcase-question"><span>Question {String(index + 1).padStart(2, "0")}</span><h3>{question.content}</h3></div>
              <div className="showcase-answer"><span>EvidenceGraph answer</span><div className="message-body"><ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ children, ...props }) => <a {...props} target="_blank" rel="noreferrer">{children}</a> }}>{readableCitations(answer.content, sources)}</ReactMarkdown></div></div>
            </article>;
          })}
        </div>
      </div>}
      {sample && !sampleLoading && !sampleError && exchanges.length === 0 && <p className="showcase-state" role="status">No saved questions are available for this tutorial yet.</p>}

      <div className="showcase-upload"><p>Ready to try a different paper?</p><button onClick={onUpload}>Start with your own PDF <span>↑</span></button></div>
    </section>
  );
}
