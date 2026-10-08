import { useRef } from "react";

import { evidenceKey } from "../lib/evidence";
import type { EvidenceAttachment } from "../types";

interface ComposerProps {
  value: string;
  evidence: EvidenceAttachment[];
  disabled: boolean;
  canOpenGraph: boolean;
  onChange: (value: string) => void;
  onEvidenceChange: (evidence: EvidenceAttachment[]) => void;
  onOpenGraph: () => void;
  onSubmit: () => void;
}

export function Composer({
  value,
  evidence,
  disabled,
  canOpenGraph,
  onChange,
  onEvidenceChange,
  onOpenGraph,
  onSubmit,
}: ComposerProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const removeEvidence = (key: string) => {
    onEvidenceChange(evidence.filter((item) => evidenceKey(item) !== key));
    inputRef.current?.focus();
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        {evidence.length > 0 && (
          <div className="attachment-row">
            {evidence.map((item) => (
              <span className="attachment-chip" key={evidenceKey(item)}>
                <i>p.{item.pages.join(",")}</i>
                {item.text.slice(0, 48)}{item.text.length > 48 ? "…" : ""}
                <button
                  aria-label={`Remove evidence from page ${item.pages.join(", ")}`}
                  onClick={() => removeEvidence(evidenceKey(item))}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        <textarea
          ref={inputRef}
          value={value}
          disabled={disabled}
          rows={1}
          aria-label="Message"
          placeholder={
            evidence.length
              ? "Ask about the selected evidence…"
              : "Ask about the paper…"
          }
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (value.trim() && !disabled) onSubmit();
            }
          }}
        />
        <div className="composer-actions">
          <button
            className="evidence-button"
            disabled={!canOpenGraph || disabled}
            onClick={onOpenGraph}
          >
            <span>⌘</span> Add evidence
          </button>
          <span className="composer-note">
            {evidence.length ? `${evidence.length} source${evidence.length > 1 ? "s" : ""} attached` : "Enter to send · Shift + Enter for a new line"}
          </span>
          <button
            className="send-button"
            aria-label="Send message"
            disabled={!value.trim() || disabled}
            onClick={onSubmit}
          >
            ↑
          </button>
        </div>
      </div>
      <p className="grounding-note">Answers stay connected to the paper. Open the graph to follow the evidence.</p>
    </div>
  );
}
