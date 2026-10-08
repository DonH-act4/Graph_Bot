import { useState, type FormEvent } from "react";

import { authenticate } from "../api";

interface AuthDialogProps {
  onClose: () => void;
  onSuccess: () => void;
}

export function AuthDialog({ onClose, onSuccess }: AuthDialogProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      await authenticate(mode, username, password);
      onSuccess();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Please try again.");
    } finally {
      setBusy(false);
    }
  };

  return <div className="auth-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="auth-dialog" role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <button className="auth-close" aria-label="Close sign in" onClick={onClose}>×</button>
      <span className="auth-kicker">Keep researching</span>
      <h2 id="auth-title">{mode === "login" ? "Sign in to ask the paper" : "Create your research account"}</h2>
      <p>Explore and generate graphs without an account. Sign in only when you want to ask questions and save conversations.</p>
      <form onSubmit={(event) => void submit(event)}>
        <label>Username<input autoFocus autoComplete="username" minLength={3} maxLength={32} pattern="[A-Za-z0-9_.-]+" value={username} onChange={(event) => setUsername(event.target.value)} required /></label>
        <label>Password<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} minLength={12} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
        {error && <p className="auth-error" role="alert">{error}</p>}
        <button className="auth-submit" type="submit" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
      </form>
      <button className="auth-switch" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
        {mode === "login" ? "New here? Create an account" : "Already have an account? Sign in"}
      </button>
    </section>
  </div>;
}
