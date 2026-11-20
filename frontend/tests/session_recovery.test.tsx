import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionRecovery } from "@/components/SessionRecovery";
import { AUTH_RESTORED_EVENT, RoomApi, getAuthToken, setAuthToken } from "@/lib/api";

afterEach(() => { vi.restoreAllMocks(); setAuthToken(null); });

describe("expired browser session", () => {
  it("stops failed network polling and restores access without losing the open edit", async () => {
    setAuthToken("expired");
    const api = new RoomApi();
    const fetcher = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Token expired" }), { status: 401 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ access_token: "renewed" }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "room", tasks: [] }), { status: 200 }));
    const restored = vi.fn();
    window.addEventListener(AUTH_RESTORED_EVENT, restored);
    render(<><input aria-label="Draft edit" defaultValue="blue leather" /><SessionRecovery /></>);
    await act(async () => { await expect(api.getProject("room")).rejects.toThrow("Token expired"); });
    expect(getAuthToken()).toBeNull();
    expect(screen.getByRole("dialog", { name: "Session expired" })).toBeInTheDocument();
    await expect(api.getProject("room")).rejects.toThrow("Session expired");
    expect(fetcher).toHaveBeenCalledTimes(1);
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "user@example.com" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in and resume" }));
    await waitFor(() => expect(restored).toHaveBeenCalledOnce());
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Draft edit")).toHaveValue("blue leather");
    await api.getProject("room");
    const headers = fetcher.mock.calls[2]?.[1]?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer renewed");
    window.removeEventListener(AUTH_RESTORED_EVENT, restored);
  });

  it("ignores a late 401 from an older session after reauthentication", async () => {
    setAuthToken("old");
    let respond: ((value: Response) => void) | undefined;
    vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise((resolve) => { respond = resolve; }));
    const pending = new RoomApi().getProject("room");
    setAuthToken("new");
    respond?.(new Response("{}", { status: 401 }));
    await expect(pending).rejects.toThrow();
    expect(getAuthToken()).toBe("new");
  });
});
