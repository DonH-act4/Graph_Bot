import { useState, type FormEvent } from "react";

import { authenticate, beginEmailRegistration, verifyEmailRegistration } from "../api";

interface AuthDialogProps {
  emailVerificationRequired: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export function AuthDialog({ emailVerificationRequired, onClose, onSuccess }: AuthDialogProps) {
  const [mode, setMode] = useState<"login" | "register" | "verify">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      if (mode === "verify") {
        await verifyEmailRegistration(username, code.trim());
        onSuccess();
      } else if (mode === "register" && emailVerificationRequired) {
        await beginEmailRegistration(username, email, password);
        setPassword("");
        setMode("verify");
      } else {
        await authenticate(mode, username, password);
        onSuccess();
      }
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
      <h2 id="auth-title">{mode === "login" ? "Sign in to ask the paper" : mode === "register" ? "Create your research account" : "Verify your email"}</h2>
      <p>{mode === "verify" ? `Enter the code sent to ${email}. It expires in 20 minutes.` : "Explore and generate graphs without an account. Sign in only when you want to ask questions and save conversations."}</p>
      <form onSubmit={(event) => void submit(event)}>
        {mode !== "verify" && <>
          <label>Username<input autoFocus autoComplete="username" minLength={3} maxLength={32} pattern="[A-Za-z0-9_.-]+" value={username} onChange={(event) => setUsername(event.target.value)} required /></label>
          {mode === "register" && emailVerificationRequired && <label>Email<input type="email" autoComplete="email" maxLength={254} value={email} onChange={(event) => setEmail(event.target.value)} required /></label>}
          <label>Password<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} minLength={12} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
        </>}
        {mode === "verify" && <label>Verification code<input autoFocus autoComplete="one-time-code" maxLength={64} value={code} onChange={(event) => setCode(event.target.value)} required /></label>}
        {error && <p className="auth-error" role="alert">{error}</p>}
        <button className="auth-submit" type="submit" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : mode === "register" ? "Create account" : "Verify and sign in"}</button>
      </form>
      <button className="auth-switch" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
        {mode === "login" ? "New here? Create an account" : mode === "verify" ? "Back to sign in" : "Already have an account? Sign in"}
      </button>
    </section>
  </div>;
}
