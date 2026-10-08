export type PaperState = "processing" | "ready" | "failed";
export type GraphState = "queued" | "processing" | "ready" | "failed";
export type PaperStage = "queued" | "docling_parsing" | "saving_evidence" | "complete" | "failed";
export type GraphStage =
  | "queued"
  | "model_generation"
  | "chunk_extraction"
  | "rate_limit_wait"
  | "graph_merge"
  | "ontology_validation"
  | "ontology_repair"
  | "saving_graph"
  | "complete"
  | "failed";

export interface PaperRecord {
  document_id: string;
  state: PaperState;
  page_count: number | null;
  block_count: number | null;
  stage: PaperStage | null;
  progress_percent: number | null;
  error: string | null;
}

export interface GraphRecord {
  document_id: string;
  state: GraphState;
  node_count: number | null;
  relation_count: number | null;
  requested_model: string | null;
  current_version: number | null;
  stage: GraphStage | null;
  progress_percent: number | null;
  progress_detail: string | null;
  warnings: GraphRepairWarning[];
  error: string | null;
}

export interface GraphRepairWarning {
  code:
    | "model_output_repaired"
    | "invalid_relations_omitted"
    | "unsupported_relations_omitted"
    | "disconnected_nodes_omitted";
  message: string;
  relation_ids: string[];
}

export interface PaperConfiguration {
  max_pdf_bytes: number;
  graph_models: string[];
  default_graph_model: string | null;
}

export interface BrowserSession {
  user_id: string;
  identity_enforced: boolean;
  chat_requires_login?: boolean;
  authenticated?: boolean;
  email_verification_required?: boolean;
}

export interface EvidenceRef {
  source_sha256: string;
  block_id: string;
}

export type NodeType =
  | "paper"
  | "actor"
  | "concept"
  | "artifact"
  | "process"
  | "observation"
  | "claim"
  | "context"
  | "method"
  | "dataset"
  | "result";

export interface GraphNode {
  node_id: string;
  node_type: NodeType;
  name: string;
  domain_type: string | null;
  evidence: EvidenceRef[];
  value: string | number | null;
  unit: string | null;
  uncertainty: string | null;
  conditions: string | null;
}

export interface GraphRelation {
  relation_id: string;
  source_node_id: string;
  target_node_id: string;
  relation_type: string;
  domain_relation: string | null;
  evidence: EvidenceRef[];
}

export interface GraphArtifact {
  graph_version: number;
  document_sha256: string;
  extractor_name: string;
  extractor_version: string;
  warnings: GraphRepairWarning[];
  graph: {
    schema_version: "1" | "2" | "3";
    nodes: GraphNode[];
    relations: GraphRelation[];
  };
}

export interface SourceLocation {
  page_number: number;
  bounding_box: {
    left: number;
    top: number;
    right: number;
    bottom: number;
    coordinate_origin: string;
  };
  character_start: number;
  character_end: number;
}

export interface EvidenceBlock {
  block_id: string;
  source_ref: string;
  label: string;
  text: string;
  locations: SourceLocation[];
}

export interface EvidenceAttachment {
  documentId: string;
  blockId: string;
  pages: number[];
  text: string;
  sourceLabel: string;
}

export interface ChatMessage {
  type: "human" | "ai" | "tool" | "custom";
  content: string;
  run_id?: string | null;
  custom_data?: {
    evidence_context?: {
      document_id: string;
      source_mode?: "automatic" | "selected";
      blocks: Array<{ block_id: string; pages: number[]; text: string }>;
    };
  };
  attachments?: EvidenceAttachment[];
  pending?: boolean;
}

export interface ThreadSummary {
  thread_id: string;
  agent_id: string;
  updated_at: string | null;
  title: string | null;
  document_id?: string | null;
  document_name?: string | null;
  selected_block_ids?: string[];
}

export interface ThreadList {
  threads: ThreadSummary[];
}

export interface ChatHistory {
  messages: ChatMessage[];
  conversation?: ConversationState | null;
}

export interface ConversationState {
  thread_id: string;
  user_id: string;
  agent_id: string;
  title: string | null;
  document_id: string | null;
  document_name: string | null;
  selected_block_ids: string[];
  draft_message: string;
  created_at: string;
  updated_at: string;
}

export interface ShowcaseRecord {
  title: string;
  description: string;
  thread_id: string;
  user_id: string;
  document_id: string;
  model: string;
}
