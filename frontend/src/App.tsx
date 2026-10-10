import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  ApiError,
  deleteConversation,
  getEvidenceBlock,
  getGraph,
  getGraphStatus,
  getHistory,
  getPaper,
  getPaperConfiguration,
  getSession,
  getShowcase,
  listThreads,
  logout,
  rebuildGraph,
  requestGraph,
  saveConversation,
  streamChat,
  uploadPaper,
} from "./api";
import { Composer } from "./components/Composer";
import { AuthDialog } from "./components/AuthDialog";
import { AdminDashboard } from "./components/AdminDashboard";
import { Showcase } from "./components/Showcase";
import { historyAttachments, readableCitations, toAttachment } from "./lib/evidence";
import type {
  ChatMessage,
  BrowserSession,
  ConversationState,
  EvidenceAttachment,
  GraphArtifact,
  GraphRecord,
  PaperConfiguration,
  PaperRecord,
  ShowcaseRecord,
  ThreadSummary,
} from "./types";

const GraphWorkspace = lazy(() =>
  import("./components/GraphWorkspace").then((module) => ({
    default: module.GraphWorkspace,
  })),
);

function idFromStorage(key: string): string {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get(key);
  if (fromUrl) {
    if (!(key === "thread_id" && params.get("view") === "demo")) window.localStorage.setItem(`evidencegraph:${key}`, fromUrl);
    return fromUrl;
  }
  const stored = window.localStorage.getItem(`evidencegraph:${key}`);
  if (stored) return stored;
  const created = crypto.randomUUID();
  window.localStorage.setItem(`evidencegraph:${key}`, created);
  return created;
}

function threadFromSession(userId: string, enforced: boolean): string {
  if (!enforced) return idFromStorage("thread_id");
  const key = `evidencegraph:thread_id:${userId}`;
  const stored = window.localStorage.getItem(key);
  if (stored) return stored;
  const created = crypto.randomUUID();
  window.localStorage.setItem(key, created);
  return created;
}

function setUrlState(userId: string, threadId: string, documentId: string | null, view: string, enforced: boolean) {
  if (view === "tutorial") {
    window.history.replaceState(null, "", "?view=tutorial");
    return;
  }
  const params = new URLSearchParams();
  if (!enforced) params.set("user_id", userId);
  params.set("thread_id", threadId);
  if (documentId) params.set("document_id", documentId);
  if (view !== "workspace") params.set("view", view);
  window.history.replaceState(null, "", `?${params.toString()}`);
}

function messageAttachments(message: ChatMessage): EvidenceAttachment[] {
  return message.attachments ?? historyAttachments(message);
}

interface ErrorNotice {
  title: string;
  detail: string;
  tone?: "error" | "warning";
}

interface PipelineState {
  activeStep: number;
  detail: string;
  percent: number;
  shortLabel: string;
  title: string;
}

const PIPELINE_STEPS = ["Upload", "Docling", "Evidence", "Sections", "Validate", "Save"];

function PipelineProgress({ state }: { state: PipelineState }) {
  return (
    <div className="pipeline-card">
      <div className="pipeline-heading">
        <div className="stage-spinner" />
        <div>
          <strong>{state.title}</strong>
          <p>{state.detail}</p>
        </div>
        <span>{state.percent}%</span>
      </div>
      <div
        className="pipeline-track"
        role="progressbar"
        aria-label="Document pipeline milestone progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={state.percent}
      >
        <i style={{ width: `${state.percent}%` }} />
      </div>
      <div className="pipeline-steps">
        {PIPELINE_STEPS.map((step, index) => (
          <span
            className={index < state.activeStep ? "done" : index === state.activeStep ? "active" : ""}
            key={step}
          >
            <b>{index < state.activeStep ? "✓" : index + 1}</b>{step}
          </span>
        ))}
      </div>
      <small>Milestone progress · the active stage animates while work continues.</small>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function graphModelLabel(model: string): string {
  if (model === "groq/openai/gpt-oss-120b") return "Groq · GPT-OSS 120B";
  if (model === "groq/openai/gpt-oss-20b") return "Groq · GPT-OSS 20B (faster)";
  if (model.startsWith("groq/")) return `Groq · ${model.slice(5)}`;
  if (model === "ollama/gpt-oss:20b") return "GPT-OSS 20B · Local";
  if (model.startsWith("ollama/")) return `Local · ${model.slice(7)}`;
  if (model.startsWith("gemini/")) return `Gemini · ${model.slice(7)}`;
  return model;
}

function isGroqModel(model: string): boolean {
  return model.startsWith("groq/");
}

const PENDING_AUTH_KEY = "evidencegraph:pending-auth-question";

function readPendingAuth(): { documentId: string | null; question: string; blockIds: string[] } | null {
  const value = window.sessionStorage.getItem(PENDING_AUTH_KEY);
  if (!value) return null;
  window.sessionStorage.removeItem(PENDING_AUTH_KEY);
  try {
    return JSON.parse(value) as { documentId: string | null; question: string; blockIds: string[] };
  } catch {
    return null;
  }
}

function WorkspaceApp({ userId, identityEnforced, canChat, authEnabled, authenticated, isAdmin, emailVerificationRequired, onAuthChanged }: {
  userId: string; identityEnforced: boolean; canChat: boolean; authEnabled: boolean;
  authenticated: boolean; isAdmin: boolean; emailVerificationRequired: boolean; onAuthChanged: () => void;
}) {
  const [threadId, setThreadId] = useState(() => threadFromSession(userId, identityEnforced));
  const [documentId, setDocumentId] = useState<string | null>(() =>
    new URLSearchParams(window.location.search).get("document_id"),
  );
  const [paper, setPaper] = useState<PaperRecord | null>(null);
  const [graphStatus, setGraphStatus] = useState<GraphRecord | null>(null);
  const [graph, setGraph] = useState<GraphArtifact | null>(null);
  const [graphOpen, setGraphOpen] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<EvidenceAttachment[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [deleteTarget, setDeleteTarget] = useState<ThreadSummary | null>(null);
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(true);
  const [view, setView] = useState<"workspace" | "showcase">(() =>
    ["showcase", "tutorial"].includes(new URLSearchParams(window.location.search).get("view") ?? "") ? "showcase" : "workspace",
  );
  const [demoOwner, setDemoOwner] = useState<string | null>(null);
  const [showcase, setShowcase] = useState<ShowcaseRecord | null>(null);
  const [showcaseLoading, setShowcaseLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [conversationReady, setConversationReady] = useState(false);
  const [conversationSaving, setConversationSaving] = useState(false);
  const [conversationTitle, setConversationTitle] = useState<string | null>(null);
  const [previewEvidence, setPreviewEvidence] = useState<EvidenceAttachment | null>(null);
  const [question, setQuestion] = useState("");
  const [authOpen, setAuthOpen] = useState(false);
  const [chatBusy, setChatBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [appError, setAppError] = useState<ErrorNotice | null>(null);
  const [paperConfig, setPaperConfig] = useState<PaperConfiguration | null>(null);
  const [selectedGraphModel, setSelectedGraphModel] = useState("");
  const [graphPollRevision, setGraphPollRevision] = useState(0);
  const [graphLookupReady, setGraphLookupReady] = useState(false);
  const [documentName, setDocumentName] = useState(() =>
    documentId ? window.localStorage.getItem(`evidencegraph:paper:${documentId}`) : null,
  );
  const fileInputRef = useRef<HTMLInputElement>(null);
  const deleteCancelRef = useRef<HTMLButtonElement>(null);
  const deleteConfirmRef = useRef<HTMLButtonElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const initialDocumentRef = useRef(documentId);
  const automaticGraphRequests = useRef(new Set<string>());
  const demoFromUrl = useRef(new URLSearchParams(window.location.search).get("view") === "demo");
  const pendingConversation = useRef<{ threadId: string; state: Pick<ConversationState, "user_id" | "document_id" | "document_name" | "selected_block_ids" | "draft_message"> } | null>(null);
  const threadStorageKey = identityEnforced ? `evidencegraph:thread_id:${userId}` : "evidencegraph:thread_id";

  const flushConversation = useCallback(() => {
    if (!canChat) return;
    const pending = pendingConversation.current;
    if (!pending) return;
    pendingConversation.current = null;
    void saveConversation(pending.threadId, pending.state, { keepalive: true }).catch(() => {
      setAppError({ title: "Conversation could not be saved", detail: "Your latest draft could not be saved. Please check the connection." });
    });
  }, [canChat]);

  useEffect(() => {
    window.addEventListener("pagehide", flushConversation);
    return () => window.removeEventListener("pagehide", flushConversation);
  }, [flushConversation]);

  useEffect(() => {
    if (!deleteTarget) return;
    deleteCancelRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !deletingThreadId) setDeleteTarget(null);
      if (event.key !== "Tab") return;
      const first = deleteCancelRef.current;
      const last = deleteConfirmRef.current;
      if (event.shiftKey && document.activeElement === first && last) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last && first) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteTarget, deletingThreadId]);

  const paperTitle = useMemo(
    () =>
      graph?.graph.nodes.find((node) => node.node_type === "paper")?.name ??
      documentName ??
      "Current paper",
    [documentName, graph],
  );

  const refreshThreads = useCallback(async () => {
    if (!canChat) return;
    try {
      setThreads((await listThreads(userId)).threads);
    } catch {
      setAppError({ title: "History unavailable", detail: "Conversation history could not be loaded. Try refreshing the page." });
    }
  }, [userId, canChat]);

  useEffect(() => {
    void refreshThreads();
  }, [refreshThreads]);

  useEffect(() => {
    if (view !== "showcase" && !demoFromUrl.current) return;
    let cancelled = false;
    setShowcaseLoading(true);
    void getShowcase().then((sample) => {
      if (cancelled) return;
      setShowcase(sample);
      if (sample && demoFromUrl.current) {
        demoFromUrl.current = false;
        setDemoOwner(sample.user_id);
        setThreadId(sample.thread_id);
        setDocumentId(sample.document_id);
        setConversationTitle(sample.title);
      } else if (!sample && demoFromUrl.current) {
        demoFromUrl.current = false;
        setView("showcase");
        setThreadId(window.localStorage.getItem(threadStorageKey) ?? crypto.randomUUID());
      }
    }).catch((error) => {
      if (!cancelled) setAppError({ title: "Tutorial unavailable", detail: error instanceof Error ? error.message : "The saved tutorial could not be loaded." });
    }).finally(() => { if (!cancelled) setShowcaseLoading(false); });
    return () => { cancelled = true; };
  }, [view, threadStorageKey]);

  useEffect(() => {
    if (demoFromUrl.current) return;
    if (!canChat && !demoOwner) {
      setConversationReady(true);
      return;
    }
    let cancelled = false;
    setConversationReady(false);
    void getHistory(threadId, demoOwner ?? userId).then(async (history) => {
      if (cancelled) return;
      const restoredMessages = history.messages.filter((message) => message.type === "human" || message.type === "ai")
        .map((message) => ({ ...message, attachments: historyAttachments(message) }));
      const state = history.conversation;
      const fallbackDocument = initialDocumentRef.current;
      initialDocumentRef.current = null;
      const pendingAuth = state ? null : readPendingAuth();
      const restoredDocument = state ? state.document_id : fallbackDocument ?? pendingAuth?.documentId ??
        restoredMessages.flatMap(messageAttachments).at(-1)?.documentId ?? null;
      setMessages(restoredMessages);
      setDocumentId(restoredDocument);
      setDocumentName(state?.document_name ?? (restoredDocument ? window.localStorage.getItem(`evidencegraph:paper:${restoredDocument}`) : null));
      setQuestion(state?.draft_message ?? pendingAuth?.question ?? "");
      setConversationTitle(state?.title ?? (demoOwner ? showcase?.title ?? null : null));
      const selectedDocument = state?.document_id ?? pendingAuth?.documentId;
      const selectedBlocks = state?.selected_block_ids ?? pendingAuth?.blockIds ?? [];
      if (selectedDocument && selectedBlocks.length) {
        const blocks = await Promise.all(selectedBlocks.map((blockId) => getEvidenceBlock(selectedDocument, blockId)));
        if (!cancelled) setSelectedEvidence(blocks.map((block) => toAttachment(selectedDocument, block, "Saved evidence")));
      } else if (!cancelled) setSelectedEvidence([]);
      if (!cancelled) setConversationReady(true);
    }).catch((error) => {
      if (!cancelled) setAppError({ title: "Unable to restore conversation", detail: error instanceof Error ? error.message : "Conversation history is unavailable." });
    });
    return () => { cancelled = true; };
  }, [threadId, userId, demoOwner, canChat]);

  useEffect(() => {
    if (!canChat || !conversationReady || demoOwner) return;
    if (!documentId && !question && !messages.length && !selectedEvidence.length) return;
    const snapshot = {
      user_id: userId, document_id: documentId, document_name: documentName,
      selected_block_ids: selectedEvidence.map((item) => item.blockId), draft_message: question,
    };
    pendingConversation.current = { threadId, state: snapshot };
    const timer = window.setTimeout(() => {
      setConversationSaving(true);
      void saveConversation(threadId, snapshot).then(() => {
        if (pendingConversation.current?.state === snapshot) pendingConversation.current = null;
        return refreshThreads();
      }).catch((error) => {
        setAppError({ title: "Conversation could not be saved", detail: error instanceof Error ? error.message : "Please retry before leaving this page." });
      }).finally(() => setConversationSaving(false));
    }, 500);
    return () => window.clearTimeout(timer);
  }, [canChat, conversationReady, demoOwner, documentId, documentName, messages.length, question, refreshThreads, selectedEvidence, threadId, userId]);

  useEffect(() => {
    let cancelled = false;
    void getPaperConfiguration()
      .then((configuration) => {
        if (cancelled) return;
        setPaperConfig(configuration);
        setSelectedGraphModel((current) => current || configuration.default_graph_model || "");
      })
      .catch((error) => {
        if (cancelled) return;
        setAppError({
          title: "Configuration unavailable",
          detail: error instanceof Error ? error.message : "Unable to load upload settings",
        });
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    setUrlState(userId, threadId, documentId, view === "showcase" ? "tutorial" : demoOwner ? "demo" : "workspace", identityEnforced);
  }, [documentId, threadId, userId, view, demoOwner, identityEnforced]);

  useEffect(() => {
    setGraphLookupReady(false);
    if (!documentId) return;
    let cancelled = false;
    let timer: number | undefined;

    const refresh = async () => {
      let needsPolling = false;
      let hasSavedGraph = false;
      try {
        const paperRecord = await getPaper(documentId);
        if (cancelled) return;
        setPaper(paperRecord);
        setAppError(null);
        if (paperRecord.state === "processing") needsPolling = true;
        if (paperRecord.state === "failed") {
          setAppError({
            title: "Paper processing failed",
            detail: paperRecord.error ?? "The PDF could not be parsed",
          });
          return;
        }
        if (paperRecord.state === "ready") {
          try {
            const artifact = await getGraph(documentId);
            hasSavedGraph = true;
            if (!cancelled) {
              setGraph(artifact);
              if (artifact.warnings.length > 0) {
                setAppError({
                  title: "Graph generated with warnings",
                  detail: artifact.warnings.map((warning) => warning.message).join(" "),
                  tone: "warning",
                });
              }
            }
          } catch (error) {
            if (!(error instanceof ApiError) || ![404, 409].includes(error.status)) throw error;
          }
          try {
            const status = await getGraphStatus(documentId);
            if (!cancelled) {
              setGraphStatus(status);
              if (status.requested_model) {
                setSelectedGraphModel((current) => current || status.requested_model || "");
              }
              if (status.state === "failed" && (!hasSavedGraph || graphPollRevision > 0)) {
                const modelPrefix = status.requested_model ? `${status.requested_model}: ` : "";
                setAppError({
                  title: "Graph generation failed",
                  detail: `${modelPrefix}${status.error ?? "The model did not return a usable graph."} ${status.requested_model?.startsWith("ollama/") ? "Please retry graph generation." : "Choose another model or retry later."}`,
                });
              }
            }
            needsPolling = status.state === "queued" || status.state === "processing";
          } catch (error) {
            if (error instanceof ApiError && error.status === 404) {
              if (!cancelled) setGraphStatus(null);
            } else {
              throw error;
            }
          }
          if (!cancelled) setGraphLookupReady(true);
        }
      } catch (error) {
        if (!cancelled) {
          if (error instanceof ApiError && error.status === 404 && authEnabled) {
            setDocumentId(null);
            setDocumentName(null);
            setPaper(null);
            setGraph(null);
            setGraphStatus(null);
            setGraphLookupReady(true);
            setAppError({
              title: "Paper needs to be reattached",
              detail: "This older paper is not linked to your current account. Upload the PDF again; the saved local files were not deleted.",
            });
            return;
          }
          setAppError({
            title: "Unable to load paper",
            detail: error instanceof Error ? error.message : "The paper service is unavailable",
          });
        }
      } finally {
        if (!cancelled && needsPolling) timer = window.setTimeout(refresh, 1800);
      }
    };
    void refresh();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [documentId, graphPollRevision, threadId, authEnabled]);

  useEffect(() => {
    if (view !== "workspace" || demoOwner || !documentId || !graphLookupReady || paper?.state !== "ready" || graph || graphStatus || !selectedGraphModel) return;
    if (automaticGraphRequests.current.has(documentId)) return;
    automaticGraphRequests.current.add(documentId);
    void requestGraph(documentId, selectedGraphModel).then((status) => {
      setGraphStatus(status);
      setGraphPollRevision((revision) => revision + 1);
    }).catch((error) => {
      setAppError({ title: "Unable to generate graph", detail: error instanceof Error ? error.message : "Graph generation could not be started." });
    });
  }, [demoOwner, documentId, graph, graphLookupReady, graphStatus, paper?.state, selectedGraphModel, view]);

  const choosePaper = () => fileInputRef.current?.click();

  const handleUpload = async (file: File) => {
    if (paperConfig && file.size > paperConfig.max_pdf_bytes) {
      setAppError({
        title: "PDF is too large",
        detail: `${file.name} is ${formatBytes(file.size)}. The maximum allowed size is ${formatBytes(paperConfig.max_pdf_bytes)}.`,
      });
      return;
    }
    setUploadBusy(true);
    setView("workspace");
    setSidebarOpen(false);
    setAppError(null);
    try {
      const record = await uploadPaper(file);
      if (demoOwner || (documentId && documentId !== record.document_id)) {
        flushConversation();
        setConversationReady(false);
        setDemoOwner(null);
        const nextThreadId = crypto.randomUUID();
        window.localStorage.setItem(threadStorageKey, nextThreadId);
        initialDocumentRef.current = record.document_id;
        setThreadId(nextThreadId);
        setMessages([]);
      }
      window.localStorage.setItem(`evidencegraph:paper:${record.document_id}`, file.name);
      setDocumentName(file.name);
      setDocumentId(record.document_id);
      setPaper(record);
      setGraph(null);
      setGraphStatus(null);
      setGraphPollRevision(0);
      setSelectedEvidence([]);
    } catch (error) {
      const serverDetail = error instanceof Error ? error.message : "The upload could not be completed";
      setAppError({
        title: error instanceof ApiError && error.status === 413 ? "PDF is too large" : "Upload failed",
        detail:
          error instanceof ApiError && error.status === 413 && paperConfig
            ? `${file.name} is ${formatBytes(file.size)}. The maximum allowed size is ${formatBytes(paperConfig.max_pdf_bytes)}.`
            : `${file.name}: ${serverDetail}`,
      });
    } finally {
      setUploadBusy(false);
    }
  };

  const generateGraph = async () => {
    if (!documentId) return;
    setAppError(null);
    try {
      const status = graph
        ? await rebuildGraph(documentId, selectedGraphModel)
        : await requestGraph(documentId, selectedGraphModel);
      setGraphStatus(status);
      setGraphPollRevision((revision) => revision + 1);
    } catch (error) {
      setAppError({
        title: graph ? "Unable to regenerate graph" : "Unable to generate graph",
        detail: error instanceof Error ? error.message : "Graph generation could not be queued",
      });
    }
  };

  const resetToNewChat = () => {
    setConversationReady(false);
    const nextThreadId = crypto.randomUUID();
    window.localStorage.setItem(threadStorageKey, nextThreadId);
    setThreadId(nextThreadId);
    setView("workspace");
    setDemoOwner(null);
    setSidebarOpen(false);
    setDocumentId(null);
    setDocumentName(null);
    setPaper(null);
    setGraph(null);
    setGraphStatus(null);
    setGraphPollRevision(0);
    setGraphOpen(false);
    setPreviewEvidence(null);
    setConversationTitle(null);
    initialDocumentRef.current = null;
    setMessages([]);
    setSelectedEvidence([]);
    setQuestion("");
  };

  const newChat = () => {
    flushConversation();
    resetToNewChat();
  };

  const confirmThreadDeletion = async () => {
    const target = deleteTarget;
    if (!target || deletingThreadId) return;
    const isCurrent = !demoOwner && target.thread_id === threadId;
    if (isCurrent) {
      pendingConversation.current = null;
      setConversationReady(false);
    }
    setDeletingThreadId(target.thread_id);
    setDeleteError(null);
    try {
      await deleteConversation(target.thread_id, userId);
      if (target.document_id) {
        window.localStorage.removeItem(`evidencegraph:paper:${target.document_id}`);
      }
      setThreads((current) => current.filter((item) => item.thread_id !== target.thread_id));
      setDeleteTarget(null);
      if (isCurrent) resetToNewChat();
      void refreshThreads();
    } catch (error) {
      if (isCurrent) setConversationReady(true);
      setDeleteError(error instanceof Error ? error.message : "Please try again.");
    } finally {
      setDeletingThreadId(null);
    }
  };

  const openThread = (thread: ThreadSummary) => {
    flushConversation();
    setConversationReady(false);
    setAppError(null);
    setView("workspace");
    setDemoOwner(null);
    setSidebarOpen(false);
    setGraphOpen(false);
    setGraph(null);
    setPaper(null);
    setGraphStatus(null);
    setSelectedEvidence([]);
    setQuestion("");
    setMessages([]);
    setDocumentId(thread.document_id ?? null);
    initialDocumentRef.current = thread.document_id ?? null;
    window.localStorage.setItem(threadStorageKey, thread.thread_id);
    setThreadId(thread.thread_id);
  };

  const exploreShowcase = () => {
    const source = showcase?.document_id ?? documentId;
    if (!source) return;
    newChat();
    initialDocumentRef.current = source;
    setDocumentId(source);
    setDocumentName(showcase?.title ?? paperTitle);
  };

  const sendMessage = async () => {
    const prompt = question.trim();
    if (!canChat && authEnabled) {
      setAuthOpen(true);
      return;
    }
    if (!prompt || chatBusy || demoOwner || !conversationReady) return;
    if (!documentId || paper?.state !== "ready") {
      setAppError({ title: "Attach a paper first", detail: "Upload a PDF and wait for parsing to finish. Your question stays in the draft." });
      return;
    }
    const attached = [...selectedEvidence];
    const human: ChatMessage = { type: "human", content: prompt, attachments: attached };
    const pending: ChatMessage = { type: "ai", content: "", pending: true };
    setMessages((current) => [...current, human, pending]);
    setQuestion("");
    setSelectedEvidence([]);
    setChatBusy(true);
    let streamed = "";
    try {
      await streamChat({
        message: prompt,
        threadId,
        userId,
        evidence: attached,
        documentId,
        onToken: (token) => {
          streamed += token;
          setMessages((current) => [
            ...current.slice(0, -1),
            { type: "ai", content: streamed, pending: true },
          ]);
        },
        onMessage: (message) => {
          if (message.type !== "ai") return;
          setMessages((current) => [
            ...current.slice(0, -1),
            { ...message, pending: false },
          ]);
        },
      });
      setMessages((current) => {
        const last = current.at(-1);
        if (!last?.pending) return current;
        return [
          ...current.slice(0, -1),
          { ...last, content: last.content || streamed, pending: false },
        ];
      });
      void refreshThreads();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Chat request failed";
      setMessages((current) => [
        ...current.slice(0, -1),
        { type: "ai", content: `I couldn't complete that request: ${message}`, pending: false },
      ]);
    } finally {
      setChatBusy(false);
    }
  };
  const closeGraph = useCallback(() => setGraphOpen(false), []);

  const onAuthSuccess = () => {
    window.sessionStorage.setItem(PENDING_AUTH_KEY, JSON.stringify({
      documentId, question, blockIds: selectedEvidence.map((item) => item.blockId),
    }));
    setAuthOpen(false);
    onAuthChanged();
  };

  const pipelineState: PipelineState | null = (() => {
    if (paper?.state === "processing") {
      const stage = paper.stage ?? "queued";
      const percent = Math.round((paper.progress_percent ?? 5) * 0.4);
      if (stage === "docling_parsing") return {
        activeStep: 1,
        detail: "Reading layout, text, tables, and page coordinates on CPU.",
        percent,
        shortLabel: "Docling is reading the PDF",
        title: "Docling is parsing the paper",
      };
      if (stage === "saving_evidence") return {
        activeStep: 2,
        detail: "Creating stable, source-located evidence blocks for the graph and chat.",
        percent,
        shortLabel: "Saving evidence blocks",
        title: "Saving traceable evidence",
      };
      return {
        activeStep: 0,
        detail: "The document worker will start Docling as soon as it is available.",
        percent,
        shortLabel: "Waiting for document worker",
        title: "Paper queued",
      };
    }
    if (graphStatus?.state === "queued") return {
      activeStep: 3,
      detail: `Waiting to call ${graphStatus.requested_model ?? selectedGraphModel}. Parsed evidence is already safe.`,
      percent: 40 + Math.round((graphStatus.progress_percent ?? 5) * 0.6),
      shortLabel: "Waiting for graph worker",
      title: "Graph generation queued",
    };
    if (graphStatus?.state === "processing") {
      const stage = graphStatus.stage ?? "model_generation";
      const model = graphStatus.requested_model ?? selectedGraphModel;
      const percent = 40 + Math.round((graphStatus.progress_percent ?? 25) * 0.6);
      if (stage === "chunk_extraction") return {
        activeStep: 3,
        detail: graphStatus.progress_detail ?? "Extracting a source-located section with the selected model.",
        percent,
        shortLabel: graphStatus.progress_detail ?? `Extracting sections · ${model}`,
        title: "Building section graphs",
      };
      if (stage === "rate_limit_wait") return {
        activeStep: 3,
        detail: graphStatus.progress_detail ?? "Waiting for the Groq token window before continuing safely.",
        percent,
        shortLabel: "Waiting for Groq free-tier quota",
        title: "Pausing between evidence sections",
      };
      if (stage === "graph_merge") return {
        activeStep: 3,
        detail: graphStatus.progress_detail ?? "De-duplicating entities and combining evidence-backed relationships.",
        percent,
        shortLabel: "Merging section graphs",
        title: "Assembling the paper graph",
      };
      if (stage === "ontology_validation") return {
        activeStep: 4,
        detail: "Checking entity types, relation endpoints, and every evidence citation.",
        percent,
        shortLabel: "Validating Ontology v3",
        title: "Validating the generated graph",
      };
      if (stage === "ontology_repair") return {
        activeStep: 4,
        detail: "Repairing a small number of unsupported relationships, then validating again.",
        percent,
        shortLabel: "Repairing graph relationships",
        title: "Applying bounded graph repair",
      };
      if (stage === "saving_graph") return {
        activeStep: 5,
        detail: "Persisting the verified graph, provenance, model metadata, and warnings.",
        percent,
        shortLabel: "Saving verified graph",
        title: "Saving the knowledge graph",
      };
      return {
        activeStep: 3,
        detail: `Asking ${graphModelLabel(model)} for an evidence-linked Ontology v3 graph.`,
        percent,
        shortLabel: `Generating · ${graphModelLabel(model)}`,
        title: "Generating the knowledge graph",
      };
    }
    return null;
  })();

  const processingLabel = (() => {
    if (pipelineState) return pipelineState.shortLabel;
    if (graphStatus?.state === "failed" && !graph) return "Graph needs attention";
    if (graph) return "Graph ready";
    return null;
  })();

  return (
    <div className={`app-shell ${sidebarOpen ? "sidebar-is-open" : ""}`}>
      {sidebarOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
      <aside className="sidebar" aria-label="Research navigation">
        <div className="brand">
          <span className="brand-mark"><i /><i /></span>
          <div><strong>EvidenceGraph</strong><small>Research, grounded</small></div>
        </div>
        <button className="new-chat-button" onClick={newChat}><span>＋</span> New chat</button>
        <nav className="workspace-nav">
          <button className={`tutorial-nav-button ${view === "showcase" ? "active" : ""}`} onClick={() => { setView("showcase"); setSidebarOpen(false); }}><span>◈</span><span><strong>Interactive tutorial</strong><small>Real paper · graph · answers</small></span><b>Explore</b></button>
        </nav>

        {authEnabled && <div className="account-control">
          {isAdmin && <a className="admin-nav-link" href="/?view=admin">Operations dashboard <span>↗</span></a>}
          {authenticated ? <button onClick={() => void logout().then(onAuthChanged).catch((error) => setAppError({ title: "Sign out failed", detail: error instanceof Error ? error.message : "Please try again." }))}>Sign out</button>
            : <button onClick={() => setAuthOpen(true)}>Sign in to ask questions <span>→</span></button>}
        </div>}

        <div className="sidebar-section">
          <button className="section-toggle" onClick={() => setHistoryOpen((value) => !value)}>
            <span>Recent conversations</span><b>{historyOpen ? "−" : "+"}</b>
          </button>
          {historyOpen && !canChat && <p className="empty-caption">Sign in to save and reopen conversations. Graphs are free to explore.</p>}
          {historyOpen && canChat && (
            <div className="thread-list">
              {threads.length === 0 && <p className="empty-caption">{authEnabled
                ? "Your new account conversations will appear here. Older local chats are preserved separately, not imported automatically."
                : "Your conversations will appear here."}</p>}
              {threads.map((thread) => (
                <div className="thread-row" key={thread.thread_id}>
                  <button className={`thread-open ${!demoOwner && thread.thread_id === threadId ? "active" : ""}`} onClick={() => void openThread(thread)}>
                    <span>{thread.title || "Untitled conversation"}</span>
                    <small>{thread.document_id ? "▧ Paper attached · " : ""}{thread.updated_at ? new Date(thread.updated_at).toLocaleDateString() : ""}</small>
                  </button>
                  <button className="thread-delete" aria-label={`Delete conversation ${thread.title || "Untitled conversation"}`} title="Delete conversation" onClick={() => { setDeleteError(null); setDeleteTarget(thread); }}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M4 7h16M10 4h4M6.5 7l.8 13h9.4l.8-13M10 11v5m4-5v5" /></svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

      </aside>

      <main className="chat-main">
        <header className="topbar">
          <button className="mobile-nav-toggle" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}>☰</button>
          <div className="conversation-title">
            <span className="status-dot" />
            <div><strong>{view === "showcase" ? "Interactive tutorial" : conversationTitle || messages[0]?.content.slice(0, 58) || "New research conversation"}</strong><small>{view === "showcase" ? "Real paper · live graph · saved questions" : demoOwner ? "Sample conversation · read only" : conversationSaving ? "Saving conversation…" : documentId ? paperTitle : "Your research workspace"}</small></div>
          </div>
          <div className="topbar-actions">
            {view === "showcase" && <button className="back-to-chat-button" onClick={() => setView("workspace")}>← Back to chat</button>}
            {view === "workspace" && documentId && !demoOwner && paperConfig && (
              <label className="model-picker">
                <span>Graph model</span>
                <select aria-label="Extraction model" value={selectedGraphModel}
                  disabled={graphStatus?.state === "queued" || graphStatus?.state === "processing"}
                  onChange={(event) => setSelectedGraphModel(event.target.value)}>
                  {paperConfig.graph_models.map((model) => <option key={model} value={model}>{graphModelLabel(model)}</option>)}
                </select>
              </label>
            )}
            {view === "workspace" && processingLabel && (!graph || graphStatus?.state === "queued" || graphStatus?.state === "processing") && <span className={`status-pill ${graph ? "ready" : ""}`}>{processingLabel}</span>}
            {view === "workspace" && graph && !demoOwner && (
              <button
                className="retry-button"
                disabled={graphStatus?.state === "queued" || graphStatus?.state === "processing"}
                onClick={() => void generateGraph()}
              >
                Regenerate
              </button>
            )}
            {view === "workspace" && <button className={`open-graph-button ${graph ? "is-ready" : ""}`} disabled={!graph} onClick={() => setGraphOpen(true)}>
              <span className="graph-glyph" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M6 7.5 12 12l6-5M12 12v6" /><circle cx="6" cy="6" r="2" /><circle cx="18" cy="6" r="2" /><circle cx="12" cy="19" r="2" /><circle cx="12" cy="12" r="2" /></svg></span>
              <span className="graph-button-label"><strong>Open graph</strong><small>{graph ? "Ready · explore evidence" : "Available after generation"}</small></span>
              {graph && <b>{graph.graph.nodes.length}</b>}
            </button>}
          </div>
        </header>
        {view === "showcase" ? (
          <Showcase sample={showcase} loading={showcaseLoading} onUpload={() => { setView("workspace"); choosePaper(); }} />
        ) : <>
        {documentId && <div className="paper-context-strip">
          <span className="strip-paper-icon">▧</span><div><strong>{paperTitle}</strong><small>{paper?.page_count ? `${paper.page_count} pages` : "PDF source"}{graph ? ` · ${graph.graph.nodes.length} entities · ${graph.graph.relations.length} relationships` : ""}</small></div>
          <a href={`/api/papers/${documentId}/source`} target="_blank" rel="noreferrer">View PDF ↗</a>
          {!demoOwner && <button className="replace-paper-button" disabled={uploadBusy} onClick={choosePaper}>{uploadBusy ? "Uploading…" : "Replace PDF"}</button>}
        </div>}
        {demoOwner && <div className="sample-conversation-banner"><span><b>Sample conversation</b> Browse the graph and original evidence, or start your own questions.</span><button onClick={exploreShowcase}>Explore this paper <span>→</span></button></div>}
        <section className="conversation">
          {messages.length === 0 ? (
            <div className={`welcome ${documentId ? "document-context" : ""}`}>
              <div className="welcome-orbit"><span /><span /><span /></div>
              <div className="eyebrow">Research, grounded</div>
              <h1>{documentId ? <>Understand the paper.<br /><em>Follow the evidence.</em></> : <>Ask the paper.<br /><em>See the connections.</em></>}</h1>
              <p>
                Explore relationships in a living knowledge graph, select the exact source passages,
                and bring them into the conversation.
              </p>
              {!documentId ? (
                <div className="welcome-actions"><button className="primary-upload" onClick={choosePaper} disabled={uploadBusy}>
                  <span>↑</span> {uploadBusy ? "Uploading…" : "Upload a research paper"}
                </button>{paperConfig && <small className="upload-limit-inline">Text-based PDF · up to {formatBytes(paperConfig.max_pdf_bytes)}</small>}<button className="welcome-sample" onClick={() => setView("showcase")}>Explore the tutorial <span>→</span></button></div>
              ) : paper?.state === "processing" ? (
                pipelineState && <PipelineProgress state={pipelineState} />
              ) : graph ? (
                <><button className="primary-upload" onClick={() => setGraphOpen(true)}>
                  <span>⌘</span> Explore knowledge graph
                </button><div className="starter-questions">{["What is the main contribution of this paper?", "How was the study carried out?", "What are the key findings?"].map((prompt) => <button key={prompt} onClick={() => setQuestion(prompt)}>{prompt}<span>↗</span></button>)}</div></>
              ) : graphStatus?.state === "queued" || graphStatus?.state === "processing" ? (
                pipelineState && <PipelineProgress state={pipelineState} />
              ) : (
                <div className="graph-setup-card">
                  <div className="graph-setup-copy">
                    <span>GPT-OSS 20B · Local</span>
                    <strong>{graphStatus?.state === "failed" ? "The last graph attempt failed" : "Preparing your knowledge graph"}</strong>
                    <p>
                      {graphStatus?.state === "failed"
                        ? graphStatus.error
                        : "Your paper has been parsed. Its core ideas and source-linked relationships are being prepared."}
                    </p>
                  </div>
                  {graphStatus?.state === "failed" && <div className="graph-model-name">{graphModelLabel(selectedGraphModel)}</div>}
                  <button
                    className="primary-upload graph-generate-button"
                    disabled={!selectedGraphModel}
                    onClick={() => void generateGraph()}
                  >
                    {graphStatus?.state === "failed" ? "Retry with selected model" : "Generate graph"}
                  </button>
                  <small>
                    {selectedGraphModel.startsWith("ollama/")
                      ? "Generated with your local model. Larger papers can take a few minutes."
                      : isGroqModel(selectedGraphModel)
                        ? "Groq may pause between evidence sections when its quota is full."
                        : "The selected provider may use its API quota."}
                  </small>
                </div>
              )}
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message, index) => {
                const attachments = messageAttachments(message);
                const citationSources = message.type === "ai"
                  ? messageAttachments(messages.slice(0, index).reverse().find((item) => item.type === "human") ?? message)
                  : attachments;
                const sourceCards = attachments.length > 0 && (
                  <div className="message-sources">
                    {attachments.map((item) => (
                      <button title={item.sourceLabel} key={`${item.documentId}:${item.blockId}`} onClick={() => setPreviewEvidence(item)}><b>p.{item.pages.join(",")}</b> {item.text.slice(0, 72)}{item.text.length > 72 ? "…" : ""}<i>↗</i></button>
                    ))}
                  </div>
                );
                return (
                  <article className={`message ${message.type}`} key={`${message.type}-${index}`}>
                    <div className="avatar">{message.type === "human" ? "You" : "EG"}</div>
                    <div className="message-body">
                      {sourceCards && (message.custom_data?.evidence_context?.source_mode === "automatic"
                        ? <details className="message-source-disclosure"><summary>{attachments.length} paper sources</summary>{sourceCards}</details>
                        : sourceCards)}
                      {message.pending && !message.content ? (
                        <div className="thinking"><i /><i /><i /></div>
                      ) : (
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ children, ...props }) => <a {...props} target="_blank" rel="noreferrer">{children}</a> }}>{readableCitations(message.content, citationSources)}</ReactMarkdown>
                      )}
                    </div>
                  </article>
                );
              })}
              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        {demoOwner ? <div className="demo-footer"><div><b>Make this paper your own.</b><span>Start a new conversation and bring evidence from the graph.</span></div><button onClick={exploreShowcase}>Ask your own questions <span>→</span></button></div> : <Composer
          value={question}
          evidence={selectedEvidence}
          disabled={chatBusy || !conversationReady}
          canOpenGraph={Boolean(graph)}
          onChange={setQuestion}
          onEvidenceChange={setSelectedEvidence}
          onOpenGraph={() => setGraphOpen(true)}
          onSubmit={() => void sendMessage()}
        />}
        </>}
        {appError && (
          <div className={`app-error ${appError.tone ?? "error"}`} role="alert">
            <span>!</span>
            <div><strong>{appError.title}</strong><p>{appError.detail}</p></div>
            <button aria-label="Dismiss error" onClick={() => setAppError(null)}>×</button>
          </div>
        )}
      </main>

      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept="application/pdf,.pdf"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void handleUpload(file);
          event.currentTarget.value = "";
        }}
      />

      {authOpen && <AuthDialog emailVerificationRequired={emailVerificationRequired} onClose={() => setAuthOpen(false)} onSuccess={onAuthSuccess} />}
      {graph && (
        <Suspense fallback={graphOpen ? <div className="graph-loading">Opening graph…</div> : null}>
          <GraphWorkspace
            artifact={graph}
            open={graphOpen}
            selectedEvidence={selectedEvidence}
            onSelectedEvidenceChange={setSelectedEvidence}
            onClose={closeGraph}
            readOnly={Boolean(demoOwner)}
          />
        </Suspense>
      )}
      {deleteTarget && <div className="delete-conversation-overlay">
        <section className="delete-conversation-dialog" role="alertdialog" aria-modal="true" aria-labelledby="delete-conversation-title" aria-describedby="delete-conversation-description">
          <span className="delete-conversation-icon" aria-hidden="true">×</span>
          <h2 id="delete-conversation-title">Delete this conversation?</h2>
          <p className="delete-conversation-name">{deleteTarget.title || "Untitled conversation"}</p>
          <p id="delete-conversation-description">{deleteTarget.document_id
            ? "This stops its active work and permanently removes the conversation, PDF, extracted text, graph, and evidence. If another conversation or the tutorial uses this PDF, deletion will be refused."
            : "This stops its active answer and permanently removes the conversation, messages, and draft."}</p>
          {deleteError && <p className="delete-conversation-error" role="alert">Could not delete conversation: {deleteError}</p>}
          <div className="delete-conversation-actions">
            <button ref={deleteCancelRef} disabled={Boolean(deletingThreadId)} onClick={() => setDeleteTarget(null)}>Cancel</button>
            <button ref={deleteConfirmRef} className="danger" disabled={Boolean(deletingThreadId)} onClick={() => void confirmThreadDeletion()}>{deletingThreadId ? "Stopping and deleting…" : "Delete everything"}</button>
          </div>
        </section>
      </div>}
      {previewEvidence && <div className="source-preview-overlay" role="dialog" aria-modal="true" aria-label="Source evidence">
        <article className="source-preview-card"><div className="source-preview-header"><span className="eyebrow">Original source · Page {previewEvidence.pages.join(", ")}</span><button aria-label="Close source evidence" onClick={() => setPreviewEvidence(null)}>×</button></div><h2>{previewEvidence.sourceLabel}</h2><p>{previewEvidence.text}</p><div className="source-preview-actions"><a href={`/api/papers/${previewEvidence.documentId}/source#page=${previewEvidence.pages[0] ?? 1}`} target="_blank" rel="noreferrer">Open source PDF ↗</a>{!demoOwner && <button disabled={selectedEvidence.length >= 12} onClick={() => { setSelectedEvidence((current) => current.some((item) => item.blockId === previewEvidence.blockId && item.documentId === previewEvidence.documentId) ? current : [...current, previewEvidence]); setPreviewEvidence(null); }}>Use in next question</button>}</div></article>
      </div>}
    </div>
  );
}

function WorkspaceRoot() {
  const [session, setSession] = useState<BrowserSession | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const loadSession = useCallback(() => {
    setSessionError(null);
    void getSession().then(setSession).catch((error) => {
      setSessionError(error instanceof Error ? error.message : "Unable to connect to the workspace");
    });
  }, []);
  useEffect(loadSession, [loadSession]);

  if (sessionError) return <main className="session-state" role="alert"><p>Workspace session unavailable: {sessionError}</p><button onClick={loadSession}>Try again</button></main>;
  if (!session) return <main className="session-state" role="status">Connecting to your research workspace…</main>;
  const userId = session.identity_enforced ? session.user_id : idFromStorage("user_id");
  const authEnabled = Boolean(session.chat_requires_login);
  const authenticated = Boolean(session.authenticated);
  return <WorkspaceApp key={`${userId}:${session.identity_enforced}`} userId={userId} identityEnforced={session.identity_enforced}
    canChat={!authEnabled || authenticated} authEnabled={authEnabled} authenticated={authenticated} isAdmin={Boolean(session.is_admin)}
    emailVerificationRequired={Boolean(session.email_verification_required)} onAuthChanged={loadSession} />;
}

export default function App() {
  return new URLSearchParams(window.location.search).get("view") === "admin"
    ? <AdminDashboard />
    : <WorkspaceRoot />;
}
