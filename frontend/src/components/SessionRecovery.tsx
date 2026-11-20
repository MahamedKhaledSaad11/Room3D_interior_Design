"use client";

import { useState, useSyncExternalStore } from "react";
import { AUTH_RESTORED_EVENT, AUTH_STATE_EVENT, isSessionExpired, roomApi } from "@/lib/api";

function subscribe(callback: () => void): () => void {
  window.addEventListener(AUTH_STATE_EVENT, callback);
  return () => window.removeEventListener(AUTH_STATE_EVENT, callback);
}

export function SessionRecovery() {
  const expired = useSyncExternalStore(subscribe, isSessionExpired, () => false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const signIn = async () => {
    setBusy(true);
    setError("");
    try {
      await roomApi.login(email.trim(), password);
      setPassword("");
      window.dispatchEvent(new Event(AUTH_RESTORED_EVENT));
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Sign in failed.");
    } finally {
      setBusy(false);
    }
  };

  if (!expired) return null;
  return <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Session expired">
    <form className="modal" onSubmit={(event) => { event.preventDefault(); void signIn(); }}>
      <h2>Sign in to continue</h2>
      <p>Your session expired. Room processing continues in the background. Your current edit stays open.</p>
      <label>Email<input type="email" autoComplete="username" required value={email} onChange={(event) => setEmail(event.target.value)} /></label>
      <label>Password<input type="password" autoComplete="current-password" required value={password} onChange={(event) => setPassword(event.target.value)} /></label>
      {error && <p role="alert" className="form-error">{error}</p>}
      <button className="button primary" type="submit" disabled={busy}>{busy ? "Signing in…" : "Sign in and resume"}</button>
    </form>
  </div>;
}
