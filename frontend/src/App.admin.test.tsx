import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";

import App from "./App";

afterEach(() => {
  window.history.replaceState(null, "", "/");
  vi.unstubAllGlobals();
});

it("shows aggregate request labels without pretending they are visitor or success counts", async () => {
  window.history.replaceState(null, "", "/?view=admin");
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    if (String(input) === "/api/admin/summary") return Promise.resolve(new Response(JSON.stringify({
      accounts_total: 3, verified_email_accounts: 1, uploads_allowed_today: 2,
      graphs_allowed_today: 1, chats_allowed_this_hour: 4,
      verification_emails_allowed_today: 1, quotas_enabled: true,
    }), { status: 200 }));
    if (String(input) === "/api/admin/accounts") return Promise.resolve(new Response(JSON.stringify({ accounts: [
      { account_id: "owner-id", username: "don", created_at: "2026-10-08T00:00:00Z", email_verified: false, banned: false, is_self: true, uploads_allowed_today: 0, graphs_allowed_today: 0, chats_allowed_this_hour: 0 },
      { account_id: "reader-id", username: "reader", created_at: "2026-10-09T00:00:00Z", email_verified: true, banned: false, is_self: false, uploads_allowed_today: 2, graphs_allowed_today: 1, chats_allowed_this_hour: 4 },
    ] }), { status: 200 }));
    if (String(input) === "/api/admin/accounts/reader-id/ban") return Promise.resolve(new Response(null, { status: 204 }));
    throw new Error(`Unexpected request: ${String(input)}`);
  });
  vi.stubGlobal("fetch", fetchMock);

  render(<App />);
  expect(await screen.findByText("Operations summary")).toBeInTheDocument();
  expect(await screen.findByText("Upload requests today")).toBeInTheDocument();
  expect(screen.getByText(/not necessarily successful uploads/)).toBeInTheDocument();
  expect(screen.getByText(/not unique visitors or successful jobs/)).toBeInTheDocument();
  expect(screen.getByRole("region", { name: "Account request comparison" })).toHaveTextContent("reader");
  fireEvent.click(screen.getByRole("button", { name: "Ban" }));
  expect(screen.getByRole("alertdialog")).toHaveTextContent("Ban reader?");
  expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith("/ban"))).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith("/ban"))).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Ban" }));
  fireEvent.click(screen.getByRole("button", { name: "Confirm ban" }));
  await waitFor(() => expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith("/ban"))).toBe(true));
});

it("does not show counts when the server rejects a non-owner", async () => {
  window.history.replaceState(null, "", "/?view=admin");
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    if (String(input) === "/api/admin/summary") return Promise.resolve(new Response(JSON.stringify({
      detail: "Administrator access required",
    }), { status: 403 }));
    if (String(input) === "/api/admin/accounts") return Promise.resolve(new Response(JSON.stringify({
      detail: "Administrator access required",
    }), { status: 403 }));
    throw new Error(`Unexpected request: ${String(input)}`);
  }));

  render(<App />);
  expect(await screen.findByRole("alert")).toHaveTextContent("Administrator access required");
  expect(screen.getByRole("button", { name: "Sign in as owner" })).toBeInTheDocument();
  expect(screen.queryByText("Accounts")).not.toBeInTheDocument();
});

it("offers a direct login form and loads the dashboard after owner sign-in", async () => {
  window.history.replaceState(null, "", "/?view=admin");
  let signedIn = false;
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path === "/api/auth/login") {
      signedIn = true;
      return Promise.resolve(new Response(JSON.stringify({ user_id: "owner-id" }), { status: 200 }));
    }
    if (!signedIn) return Promise.resolve(new Response(JSON.stringify({ detail: "Sign in to view operations" }), { status: 401 }));
    if (path === "/api/admin/summary") return Promise.resolve(new Response(JSON.stringify({
      accounts_total: 1, verified_email_accounts: 0, uploads_allowed_today: 0,
      graphs_allowed_today: 0, chats_allowed_this_hour: 0,
      verification_emails_allowed_today: 0, quotas_enabled: true,
    }), { status: 200 }));
    if (path === "/api/admin/accounts") return Promise.resolve(new Response(JSON.stringify({ accounts: [] }), { status: 200 }));
    throw new Error(`Unexpected request: ${path}`);
  }));

  render(<App />);
  expect(await screen.findByRole("button", { name: "Sign in as owner" })).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Username"), { target: { value: "don" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "owner-test-password" } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in as owner" }));
  expect(await screen.findByText("Usage by account")).toBeInTheDocument();
  expect(screen.queryByLabelText("Password")).not.toBeInTheDocument();
});
