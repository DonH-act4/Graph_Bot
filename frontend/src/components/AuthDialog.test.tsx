import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";

import * as api from "../api";
import { AuthDialog } from "./AuthDialog";

vi.mock("../api", () => ({
  authenticate: vi.fn(),
  beginEmailRegistration: vi.fn(),
  verifyEmailRegistration: vi.fn(),
}));

beforeEach(() => vi.resetAllMocks());

it("does not sign in until the emailed code is verified", async () => {
  vi.mocked(api.beginEmailRegistration).mockResolvedValue({ status: "verification_required" });
  vi.mocked(api.verifyEmailRegistration).mockResolvedValue({ user_id: "account-1" });
  const onSuccess = vi.fn();
  render(<AuthDialog emailVerificationRequired onClose={vi.fn()} onSuccess={onSuccess} />);
  fireEvent.click(screen.getByRole("button", { name: "New here? Create an account" }));
  fireEvent.change(screen.getByLabelText("Username"), { target: { value: "reader" } });
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "reader@example.test" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "long-password-123" } });
  fireEvent.click(screen.getByRole("button", { name: "Create account" }));
  await screen.findByRole("dialog", { name: "Verify your email" });
  expect(api.beginEmailRegistration).toHaveBeenCalledWith("reader", "reader@example.test", "long-password-123");
  expect(onSuccess).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText("Verification code"), { target: { value: "emailed-code" } });
  fireEvent.click(screen.getByRole("button", { name: "Verify and sign in" }));
  await waitFor(() => expect(api.verifyEmailRegistration).toHaveBeenCalledWith("reader", "emailed-code"));
  expect(onSuccess).toHaveBeenCalledOnce();
});
