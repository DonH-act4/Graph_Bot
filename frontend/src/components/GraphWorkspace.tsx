import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, {
  type ForceGraphMethods,
  type LinkObject,
  type NodeObject,
} from "react-force-graph-2d";

import { getEvidenceBlock } from "../api";
import { evidenceKey, toAttachment } from "../lib/evidence";
import type {
  EvidenceAttachment,
  EvidenceBlock,
  GraphArtifact,
  GraphNode,
  GraphRelation,
  NodeType,
} from "../types";

const NODE_COLORS: Record<NodeType, string> = {
  paper: "#8b7cf6",
  actor: "#f08f74",
  concept: "#43b5d9",
  artifact: "#36c58b",
  process: "#f0b44f",
  observation: "#ff8b67",
  claim: "#ef6e9d",
  context: "#8fa7bd",
  method: "#f0b44f",
  dataset: "#36c58b",
  result: "#ff8b67",
};

interface CanvasNode extends NodeObject {
  id: string;
  item: GraphNode;
}

interface CanvasLink extends LinkObject<CanvasNode> {
  id: string;
  source: string | CanvasNode;
  target: string | CanvasNode;
  item: GraphRelation;
}

type Selection =
  | { kind: "node"; item: GraphNode }
  | { kind: "relation"; item: GraphRelation };

interface GraphWorkspaceProps {
  artifact: GraphArtifact;
  open: boolean;
  selectedEvidence: EvidenceAttachment[];
  onSelectedEvidenceChange: (evidence: EvidenceAttachment[]) => void;
  onClose: () => void;
  readOnly?: boolean;
  inline?: boolean;
}

function useElementSize(open: boolean) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 650 });
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.max(320, Math.floor(entry.contentRect.width)),
        height: Math.max(360, Math.floor(entry.contentRect.height)),
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [open]);
  return { ref, size };
}

function relationEndpoint(
  endpoint: string | CanvasNode,
  nodesById: Map<string, GraphNode>,
): GraphNode | undefined {
  return typeof endpoint === "string" ? nodesById.get(endpoint) : endpoint.item;
}

function nodeLabel(node: GraphNode): string {
  const max = node.node_type === "paper" ? 29 : 22;
  return node.name.length > max ? `${node.name.slice(0, max - 1)}…` : node.name;
}

export function GraphWorkspace({
  artifact,
  open,
  selectedEvidence,
  onSelectedEvidenceChange,
  onClose,
  readOnly = false,
  inline = false,
}: GraphWorkspaceProps) {
  const graphRef = useRef<ForceGraphMethods<CanvasNode, CanvasLink> | undefined>(undefined);
  const { ref: canvasRef, size } = useElementSize(open);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [blocks, setBlocks] = useState<EvidenceBlock[]>([]);
  const [loadingEvidence, setLoadingEvidence] = useState(false);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const cache = useRef(new Map<string, EvidenceBlock>());
  const shellRef = useRef<HTMLDivElement>(null);
  const didFit = useRef(false);
  const [search, setSearch] = useState("");
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const [reducedMotion] = useState(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);

  const nodesById = useMemo(
    () => new Map(artifact.graph.nodes.map((node) => [node.node_id, node])),
    [artifact],
  );
  const graphData = useMemo(
    () => ({
      nodes: artifact.graph.nodes.map((item) => ({ id: item.node_id, item })),
      links: artifact.graph.relations.map((item) => ({
        id: item.relation_id,
        source: item.source_node_id,
        target: item.target_node_id,
        item,
      })),
    }),
    [artifact],
  );

  useEffect(() => {
    setSelection(null);
    setBlocks([]);
    setSearch("");
    didFit.current = false;
  }, [artifact.document_sha256, artifact.graph_version]);

  useEffect(() => {
    if (!open || inline) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    shellRef.current?.querySelector<HTMLButtonElement>('button[aria-label="Close graph"]')?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab") return;
      const targets = [...(shellRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), a[href], summary') ?? [])];
      const first = targets[0];
      const last = targets.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    window.addEventListener("keydown", onKey);
    return () => { window.removeEventListener("keydown", onKey); previousFocus?.focus(); };
  }, [open, inline, onClose]);

  const searchResults = search.trim() ? artifact.graph.nodes.filter((node) => node.name.toLowerCase().includes(search.toLowerCase())).slice(0, 6) : [];
  const selectNode = (item: GraphNode) => {
    setSelection({ kind: "node", item });
    setSearch("");
    const node = graphData.nodes.find((value) => value.id === item.node_id) as CanvasNode | undefined;
    if (node?.x != null && node.y != null) graphRef.current?.centerAt(node.x, node.y, 450);
  };
  const endpointId = (value: string | CanvasNode) => typeof value === "string" ? value : value.id;
  const highlightedNodes = new Set<string>();
  if (hoverNode) {
    highlightedNodes.add(hoverNode);
    graphData.links.forEach((link) => {
      const source = endpointId(link.source);
      const target = endpointId(link.target);
      if (source === hoverNode || target === hoverNode) { highlightedNodes.add(source); highlightedNodes.add(target); }
    });
  }

  useEffect(() => {
    if (!open) return;
    didFit.current = false;
    const tuneTimer = window.setTimeout(() => {
      const graph = graphRef.current;
      const linkForce = graph?.d3Force("link") as
        | { distance: (distance: number) => unknown }
        | undefined;
      const chargeForce = graph?.d3Force("charge") as
        | { strength: (strength: number) => unknown }
        | undefined;
      linkForce?.distance(artifact.graph.nodes.length > 20 ? 120 : 100);
      chargeForce?.strength(-340);
      graph?.d3ReheatSimulation();
    }, 0);
    const fitTimers = [300, 1200].map((delay) =>
      window.setTimeout(() => graphRef.current?.zoomToFit(500, 70), delay),
    );
    return () => {
      window.clearTimeout(tuneTimer);
      fitTimers.forEach((timer) => window.clearTimeout(timer));
    };
  }, [open, artifact.document_sha256, artifact.graph_version]);

  useEffect(() => {
    if (!selection) {
      setBlocks([]);
      return;
    }
    let cancelled = false;
    const refs = selection.item.evidence;
    const load = async () => {
      setLoadingEvidence(true);
      setEvidenceError(null);
      try {
        const resolved = await Promise.all(
          refs.map(async (ref) => {
            const key = `${ref.source_sha256}:${ref.block_id}`;
            const cached = cache.current.get(key);
            if (cached) return cached;
            const block = await getEvidenceBlock(ref.source_sha256, ref.block_id);
            cache.current.set(key, block);
            return block;
          }),
        );
        if (!cancelled) setBlocks(resolved);
      } catch (error) {
        if (!cancelled) {
          setEvidenceError(error instanceof Error ? error.message : "Unable to load evidence");
        }
      } finally {
        if (!cancelled) setLoadingEvidence(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [selection]);

  if (!open) return null;

  const selectionTitle = (() => {
    if (!selection) return "Select a node or relationship";
    if (selection.kind === "node") return selection.item.name;
    const source = nodesById.get(selection.item.source_node_id)?.name ?? "Source";
    const target = nodesById.get(selection.item.target_node_id)?.name ?? "Target";
    return `${source} · ${selection.item.domain_relation ?? selection.item.relation_type} · ${target}`;
  })();

  const toggleEvidence = (block: EvidenceBlock) => {
    const attachment = toAttachment(
      artifact.document_sha256,
      block,
      selectionTitle,
    );
    const key = evidenceKey(attachment);
    const exists = selectedEvidence.some((item) => evidenceKey(item) === key);
    if (!exists && selectedEvidence.length >= 12) return;
    onSelectedEvidenceChange(
      exists
        ? selectedEvidence.filter((item) => evidenceKey(item) !== key)
        : [...selectedEvidence, attachment],
    );
  };

  return (
    <div className={inline ? "graph-inline" : "graph-overlay"} role={inline ? "region" : "dialog"} aria-modal={inline ? undefined : true} aria-label="Knowledge graph">
      <div className="graph-shell" ref={shellRef}>
        <header className="graph-header">
          <div>
            <div className="eyebrow">The paper, connected</div>
            <h2>Knowledge graph</h2>
            <p>
              {artifact.graph.nodes.length} entities · {artifact.graph.relations.length} relationships
            </p>
          </div>
          <div className="graph-header-actions">
            {!readOnly && <button className="use-sources-button" onClick={onClose}>{selectedEvidence.length ? `Use ${selectedEvidence.length} source${selectedEvidence.length > 1 ? "s" : ""} in chat` : "Back to conversation"} <span>→</span></button>}
            <button className="ghost-button" onClick={() => graphRef.current?.zoomToFit(500, 70)}>
              Fit graph
            </button>
            {!inline && <button className="icon-button" aria-label="Close graph" onClick={onClose}>×</button>}
          </div>
        </header>

        <div className="graph-body">
          <div className="graph-canvas" ref={canvasRef}>
            <div className="graph-search"><input aria-label="Find an entity" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find an idea in the graph…" onKeyDown={(event) => { if (event.key === "Enter" && searchResults[0]) selectNode(searchResults[0]); }} />{search && <div className="graph-search-results">{searchResults.length ? searchResults.map((node) => <button key={node.node_id} onClick={() => selectNode(node)}><i style={{ background: NODE_COLORS[node.node_type] }} />{node.name}<small>{node.domain_type ?? node.node_type}</small></button>) : <p>No matching entity</p>}</div>}</div>
            <div className="graph-legend">
              {[...new Set(artifact.graph.nodes.map((node) => node.node_type))].map(
                (type) => (
                  <span key={type}>
                    <i style={{ background: NODE_COLORS[type] }} /> {type}
                  </span>
                ),
              )}
            </div>
            <ForceGraph2D<CanvasNode, CanvasLink>
              ref={graphRef}
              width={size.width}
              height={size.height}
              graphData={graphData}
              backgroundColor="#0b0d16"
              nodeRelSize={5}
              d3AlphaDecay={0.025}
              d3VelocityDecay={0.28}
              cooldownTicks={180}
              linkCurvature={0.08}
              linkColor={(link) => selection?.kind === "relation" && selection.item.relation_id === link.id ? "rgba(196,182,255,.85)" : hoverNode && endpointId(link.source) !== hoverNode && endpointId(link.target) !== hoverNode ? "rgba(159,176,218,.08)" : "rgba(159,176,218,.38)"}
              linkHoverPrecision={8}
              linkWidth={(link) =>
                selection?.kind === "relation" && selection.item.relation_id === link.item.relation_id
                  ? 2.4
                  : 1.1
              }
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={0.88}
              linkDirectionalParticles={reducedMotion ? 0 : 2}
              linkDirectionalParticleWidth={(link) =>
                selection?.kind === "relation" && selection.item.relation_id === link.item.relation_id
                  ? 3
                  : 1.4
              }
              linkDirectionalParticleSpeed={0.004}
              linkPointerAreaPaint={(link, color, context, scale) => {
                if (
                  typeof link.source === "string" ||
                  typeof link.target === "string" ||
                  link.source.x == null ||
                  link.source.y == null ||
                  link.target.x == null ||
                  link.target.y == null
                ) return;
                context.beginPath();
                context.moveTo(link.source.x, link.source.y);
                context.lineTo(link.target.x, link.target.y);
                context.strokeStyle = color;
                context.lineWidth = 14 / scale;
                context.stroke();
                const label = (link.item.domain_relation ?? link.item.relation_type).replaceAll(
                  "_",
                  " ",
                );
                const x = (link.source.x + link.target.x) / 2;
                const y = (link.source.y + link.target.y) / 2;
                let angle = Math.atan2(
                  link.target.y - link.source.y,
                  link.target.x - link.source.x,
                );
                if (angle > Math.PI / 2 || angle < -Math.PI / 2) angle += Math.PI;
                const fontSize = 8.5 / scale;
                context.save();
                context.translate(x, y);
                context.rotate(angle);
                context.font = `650 ${fontSize}px Inter, ui-sans-serif`;
                const width = context.measureText(label).width;
                const padX = 7 / scale;
                const height = 20 / scale;
                context.fillStyle = color;
                context.fillRect(
                  -width / 2 - padX,
                  -height / 2,
                  width + padX * 2,
                  height,
                );
                context.restore();
              }}
              linkCanvasObjectMode={() => "after"}
              linkCanvasObject={(link, context, scale) => {
                if (
                  typeof link.source === "string" ||
                  typeof link.target === "string" ||
                  link.source.x == null ||
                  link.source.y == null ||
                  link.target.x == null ||
                  link.target.y == null
                ) return;
                const label = link.item.domain_relation ?? link.item.relation_type;
                const x = (link.source.x + link.target.x) / 2;
                const y = (link.source.y + link.target.y) / 2;
                let angle = Math.atan2(link.target.y - link.source.y, link.target.x - link.source.x);
                if (angle > Math.PI / 2 || angle < -Math.PI / 2) angle += Math.PI;
                const fontSize = 8.5 / scale;
                context.save();
                context.translate(x, y);
                context.rotate(angle);
                context.font = `650 ${fontSize}px Inter, ui-sans-serif`;
                const width = context.measureText(label).width;
                const padX = 5 / scale;
                const height = 16 / scale;
                context.beginPath();
                context.roundRect(
                  -width / 2 - padX,
                  -height / 2,
                  width + padX * 2,
                  height,
                  6 / scale,
                );
                context.fillStyle = "rgba(18, 21, 34, .9)";
                context.fill();
                context.strokeStyle = "rgba(135, 146, 190, .32)";
                context.lineWidth = 0.6 / scale;
                context.stroke();
                context.textAlign = "center";
                context.textBaseline = "middle";
                context.fillStyle = "rgba(208, 214, 236, .88)";
                context.fillText(label.replaceAll("_", " "), 0, 0);
                context.restore();
              }}
              nodeCanvasObject={(node, context, scale) => {
                const screenRadius = node.item.node_type === "paper" ? 22 : 15;
                const radius = screenRadius / scale;
                const selected =
                  selection?.kind === "node" && selection.item.node_id === node.item.node_id;
                context.save();
                if (hoverNode && !highlightedNodes.has(node.id)) context.globalAlpha = .22;
                context.beginPath();
                context.arc(
                  node.x ?? 0,
                  node.y ?? 0,
                  radius + (selected ? 3 / scale : 0),
                  0,
                  Math.PI * 2,
                );
                context.fillStyle = selected ? "rgba(255,255,255,.22)" : "rgba(255,255,255,.08)";
                context.fill();
                context.beginPath();
                context.arc(node.x ?? 0, node.y ?? 0, radius, 0, Math.PI * 2);
                context.fillStyle = NODE_COLORS[node.item.node_type] ?? "#8fa7bd";
                context.fill();
                context.strokeStyle = "rgba(255,255,255,.72)";
                context.lineWidth = 0.7 / scale;
                context.stroke();
                const label = nodeLabel(node.item);
                const fontSize = 11 / scale;
                context.font = `600 ${fontSize}px Inter, ui-sans-serif`;
                context.textAlign = "center";
                context.textBaseline = "top";
                context.fillStyle = "rgba(244,247,255,.92)";
                context.fillText(
                  label,
                  node.x ?? 0,
                  (node.y ?? 0) + radius + 5 / scale,
                );
                context.restore();
              }}
              nodePointerAreaPaint={(node, color, context, scale) => {
                context.beginPath();
                context.arc(node.x ?? 0, node.y ?? 0, 18 / scale, 0, Math.PI * 2);
                context.fillStyle = color;
                context.fill();
              }}
              nodeLabel={(node) => `${node.item.name} · ${node.item.domain_type ?? node.item.node_type}`}
              linkLabel={(link) => {
                const source = relationEndpoint(link.source, nodesById)?.name ?? "Source";
                const target = relationEndpoint(link.target, nodesById)?.name ?? "Target";
                return `${source} · ${link.item.domain_relation ?? link.item.relation_type} · ${target}`;
              }}
              onNodeClick={(node) => selectNode(node.item)}
              onNodeHover={(node) => setHoverNode(node?.id ?? null)}
              onLinkClick={(link) => setSelection({ kind: "relation", item: link.item })}
              onEngineStop={() => { if (!didFit.current) { didFit.current = true; graphRef.current?.zoomToFit(400, 85); } }}
            />
            <div className="canvas-hint">Drag to move · scroll to zoom · click a line or node</div>
            <details className="graph-accessible-index"><summary>Browse relationships</summary><div>{artifact.graph.relations.map((relation) => <button key={relation.relation_id} onClick={() => setSelection({ kind: "relation", item: relation })}>{nodesById.get(relation.source_node_id)?.name} <b>{(relation.domain_relation ?? relation.relation_type).replaceAll("_", " ")}</b> {nodesById.get(relation.target_node_id)?.name}</button>)}</div></details>
          </div>

          <aside className="evidence-inspector">
            <div className="inspector-kicker">
              {selection?.kind === "relation" ? "Relationship evidence" : "Source evidence"}
            </div>
            <h3>{selectionTitle}</h3>
            {selection && <p className="inspector-description">Original passages from the paper{!readOnly ? " · select a card to use it in your next question" : ""}.</p>}
            {!selection && (
              <div className="inspector-empty">
                <div className="empty-orbit">◎</div>
                <p>Choose any entity or relationship to inspect the exact source text.</p>
              </div>
            )}
            {loadingEvidence && <div className="loading-row"><span /> Loading source text</div>}
            {evidenceError && <div className="inline-error">{evidenceError}</div>}
            <div className="evidence-list">
              {blocks.map((block) => {
                const attachment = toAttachment(artifact.document_sha256, block, selectionTitle);
                const checked = selectedEvidence.some(
                  (item) => evidenceKey(item) === evidenceKey(attachment),
                );
                const pages = attachment.pages.join(", ");
                return (
                  <article className={`evidence-card ${checked ? "selected" : ""}`} key={block.block_id}>
                    <div className="evidence-card-topline">
                      <span>Page {pages}</span>
                      {!readOnly && <label>
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={!checked && selectedEvidence.length >= 12}
                          onChange={() => toggleEvidence(block)}
                        />
                        <b>{checked ? "Added" : "Use in chat"}</b>
                      </label>}
                    </div>
                    <p>{block.text}</p>
                    <a
                      href={`/api/papers/${artifact.document_sha256}/source#page=${attachment.pages[0] ?? 1}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open source PDF ↗
                    </a>
                  </article>
                );
              })}
            </div>
            {!readOnly && selectedEvidence.length > 0 && <div className="inspector-selection-summary"><span>{selectedEvidence.length}/12 sources selected</span><button onClick={onClose}>Continue in chat →</button></div>}
          </aside>
        </div>
      </div>
    </div>
  );
}
