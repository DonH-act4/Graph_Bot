import { useCallback, useEffect, useState } from "react";
import type { FormEvent } from "react";

import { ApiError, authenticate, getAdminSummary, getManagedAccounts, setAccountBan } from "../api";
import type { AdminSummary, ManagedAccount } from "../types";

const cards: Array<{ label: string; key: keyof AdminSummary; note: string }> = [
  { label: "Accounts", key: "accounts_total", note: "All created accounts, including older accounts" },
  { label: "Verified email accounts", key: "verified_email_accounts", note: "Accounts with a verified email address" },
  { label: "Upload requests today", key: "uploads_allowed_today", note: "Allowed by quota; not necessarily successful uploads" },
  { label: "Graph requests today", key: "graphs_allowed_today", note: "Allowed by quota; not necessarily completed graphs" },
  { label: "Chat requests this hour", key: "chats_allowed_this_hour", note: "Allowed by quota; not necessarily completed answers" },
  { label: "Verification emails today", key: "verification_emails_allowed_today", note: "Provider-call budget consumed; delivery is not guaranteed" },
];

const accountMetrics: Array<{ label: string; key: "uploads_allowed_today" | "graphs_allowed_today" | "chats_allowed_this_hour" }> = [
  { label: "Uploads today", key: "uploads_allowed_today" },
  { label: "Graphs today", key: "graphs_allowed_today" },
  { label: "Chats this hour", key: "chats_allowed_this_hour" },
];

type PendingAction = { account: ManagedAccount; banned: boolean };

export function AdminDashboard() {
  const [summary, setSummary] = useState<AdminSummary | null>(null);
  const [accounts, setAccounts] = useState<ManagedAccount[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [reason, setReason] = useState("");
  const [moderationBusy, setModerationBusy] = useState(false);
  const [moderationError, setModerationError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    setErrorStatus(null);
    void Promise.all([getAdminSummary(), getManagedAccounts()]).then(([nextSummary, nextAccounts]) => {
      setSummary(nextSummary);
      setAccounts(nextAccounts.accounts);
    }).catch((cause) => {
      setSummary(null);
      setAccounts([]);
      setError(cause instanceof Error ? cause.message : "Could not load the dashboard");
      setErrorStatus(cause instanceof ApiError ? cause.status : null);
    }).finally(() => setLoading(false));
  }, []);

  useEffect(refresh, [refresh]);

  function submitLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoginBusy(true);
    setLoginError(null);
    void authenticate("login", username, password).then(() => {
      setPassword("");
      refresh();
    }).catch((cause) => {
      setLoginError(cause instanceof Error ? cause.message : "Sign in failed");
    }).finally(() => setLoginBusy(false));
  }

  function submitModeration() {
    if (!pending) return;
    setModerationBusy(true);
    setModerationError(null);
    void setAccountBan(pending.account.account_id, pending.banned, reason).then(() => {
      setPending(null);
      setReason("");
      refresh();
    }).catch((cause) => {
      setModerationError(cause instanceof Error ? cause.message : "Account action failed");
    }).finally(() => setModerationBusy(false));
  }

  const unattributed = summary ? {
    uploads: Math.max(0, summary.uploads_allowed_today - accounts.reduce((sum, item) => sum + item.uploads_allowed_today, 0)),
    graphs: Math.max(0, summary.graphs_allowed_today - accounts.reduce((sum, item) => sum + item.graphs_allowed_today, 0)),
    chats: Math.max(0, summary.chats_allowed_this_hour - accounts.reduce((sum, item) => sum + item.chats_allowed_this_hour, 0)),
  } : null;

  return <main className="admin-page">
    <header className="admin-header">
      <div><span className="admin-kicker">EvidenceGraph · owner view</span><h1>Operations summary</h1>
        <p>Account-level usage and reversible moderation. No visitor documents, questions, or email addresses are shown.</p></div>
      <a href="/">Back to workspace</a>
    </header>
    {loading && <p role="status">Loading dashboard…</p>}
    {error && <section className="admin-error" role="alert"><p>{error}</p>
      {(errorStatus === 401 || errorStatus === 403) && <>
        <p>Sign in with the designated owner account. This replaces the current workspace login in this browser.</p>
        <form className="admin-login-form" onSubmit={submitLogin}>
          <label>Username<input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" required minLength={3} /></label>
          <label>Password<input value={password} onChange={(event) => setPassword(event.target.value)} type="password" autoComplete="current-password" required /></label>
          {loginError && <p role="alert">{loginError}</p>}
          <button disabled={loginBusy} type="submit">{loginBusy ? "Signing in…" : "Sign in as owner"}</button>
        </form>
      </>}
      {errorStatus !== 401 && errorStatus !== 403 && <button onClick={refresh}>Try again</button>}
    </section>}
    {summary && <>
      {!summary.quotas_enabled && <p className="admin-warning">Request quotas are disabled; request counters may be zero or incomplete.</p>}
      <div className="admin-grid">{cards.map((card) => <section className="admin-card" key={card.key}>
        <span>{card.label}</span><strong>{summary[card.key].toString()}</strong><p>{card.note}</p>
      </section>)}</div>
      <section className="admin-accounts" aria-label="Account request comparison">
        <h2>Usage by account</h2>
        <p>Current UTC quota windows, latest 100 accounts. Guest requests are not assigned to an account. Bars compare accounts within each request type; they do not show successful jobs.</p>
        {accounts.length === 0 && <p>No accounts to display.</p>}
        {accounts.map((account) => <article className="admin-account" key={account.account_id}>
          <div className="admin-account-heading"><div><strong>{account.username}</strong>{account.is_self && <span className="admin-account-self">You</span>}
            <small>{account.email_verified ? "Verified email" : "No verified email"} · Joined {new Date(account.created_at).toLocaleDateString()} · {account.banned ? "Banned" : "Active"}</small></div>
            {!account.is_self && <button className={account.banned ? "" : "danger"} onClick={() => { setPending({ account, banned: !account.banned }); setReason(""); setModerationError(null); }}>{account.banned ? "Unban" : "Ban"}</button>}
          </div>
          <div className="admin-account-metrics">{accountMetrics.map((metric) => {
            const maximum = Math.max(1, ...accounts.map((item) => item[metric.key]));
            const value = account[metric.key];
            return <div className="admin-meter" key={metric.key}><span>{metric.label}</span><div className="admin-meter-track"><i style={{ width: `${value === 0 ? 0 : Math.max(5, value / maximum * 100)}%` }} /></div><b>{value}</b></div>;
          })}</div>
        </article>)}
        {unattributed && Object.values(unattributed).some((value) => value > 0) && <p className="admin-unattributed">Not attributed to these accounts: {unattributed.uploads} uploads, {unattributed.graphs} graphs, {unattributed.chats} chats. This can include guests or accounts outside the latest 100.</p>}
      </section>
      <p className="admin-footnote">“Today” uses UTC calendar days; “this hour” uses the current UTC hour. These are quota counters, not unique visitors or successful jobs. Banning an account revokes its sessions but does not delete saved data or prevent guest access.</p>
      <button className="admin-refresh" onClick={refresh}>Refresh counts</button>
    </>}
    {pending && <div className="admin-confirm-overlay" role="presentation"><section className="admin-confirm" role="alertdialog" aria-modal="true" aria-labelledby="admin-confirm-title">
      <h2 id="admin-confirm-title">{pending.banned ? "Ban" : "Unban"} {pending.account.username}?</h2>
      <p>{pending.banned ? "Their current login sessions will stop working. Saved papers and conversations remain." : "They can sign in again, but old sessions stay revoked."}</p>
      {pending.banned && <label>Reason (optional, kept in the audit log)<input maxLength={200} value={reason} onChange={(event) => setReason(event.target.value)} /></label>}
      {moderationError && <p role="alert">{moderationError}</p>}
      <div><button disabled={moderationBusy} onClick={() => setPending(null)}>Cancel</button><button className={pending.banned ? "danger" : ""} disabled={moderationBusy} onClick={submitModeration}>{moderationBusy ? "Working…" : `Confirm ${pending.banned ? "ban" : "unban"}`}</button></div>
    </section></div>}
  </main>;
}
