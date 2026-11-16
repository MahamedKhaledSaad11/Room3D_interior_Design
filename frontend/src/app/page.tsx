"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowUpRight, Plus, UploadCloud } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { ApiError, getAuthToken, roomApi, type Project } from "@/lib/api";
import { ThemeToggle } from "@/components/ThemeToggle";

export default function HomePage() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("Living room");
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [authOpen, setAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [signedIn, setSignedIn] = useState(false);

  const loadProjects = useCallback(() => {
    void roomApi.listProjects().then(setProjects).catch(() => undefined);
  }, []);

  useEffect(() => {
    setSignedIn(Boolean(getAuthToken()));
    loadProjects();
  }, [loadProjects]);

  const submitAuth = async () => {
    if (!email.trim() || !password) {
      setAuthError("Enter both email and password.");
      return;
    }
    setAuthBusy(true);
    setAuthError("");
    try {
      const res = authMode === "login"
        ? await roomApi.login(email.trim(), password)
        : await roomApi.register(email.trim(), password);
      roomApi.setToken(res.access_token);
      setSignedIn(true);
      setAuthOpen(false);
      setPassword("");
      loadProjects();
    } catch (reason) {
      setAuthError(reason instanceof ApiError ? reason.message : "Authentication failed.");
    } finally {
      setAuthBusy(false);
    }
  };

  const signOut = () => {
    roomApi.setToken(null);
    setSignedIn(false);
    loadProjects();
  };

  const create = async () => {
    if (!file) {
      setError("Choose a room video first.");
      return;
    }
    setError("");
    try {
      const project = await roomApi.createProject(name.trim() || "Untitled room");
      await roomApi.uploadVideo(project.id, file, (pct) => setUploadPct(pct));
      router.push(`/projects/${project.id}`);
    } catch (reason) {
      setUploadPct(null);
      setError(reason instanceof Error ? reason.message : "Could not start the room build.");
    }
  };

  return (
    <main className="shell">
      <header className="topbar">
        <Link className="brand" href="/">ROOM<span>3D</span></Link>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <ThemeToggle />
          {signedIn ? (
            <button type="button" className="button secondary" onClick={signOut}>
              Sign Out
            </button>
          ) : (
            <button type="button" className="button secondary" onClick={() => setAuthOpen(true)}>
              Sign In / Register
            </button>
          )}
          <span className="header-note">{signedIn ? "SIGNED IN" : "PRIVATE WORKSPACE"}</span>
        </div>
      </header>
      <div className="workspace">
        <div className="workspace-head">
          <div>
            <p className="eyebrow">YOUR LIBRARY</p>
            <h1>Projects</h1>
            <p className="muted">Build a room you can walk through.</p>
          </div>
          <button className="button primary" onClick={() => setOpen(true)}>
            <Plus size={17} /> New room video
          </button>
        </div>
        <div className="project-grid">
          {projects.map((project) => (
            <Link className="project-card" href={`/projects/${project.id}`} key={project.id}>
              <div className="card-mark">{project.name.slice(0, 1).toUpperCase()}</div>
              <div>
                <h2>{project.name}</h2>
                <p>{project.status}</p>
              </div>
              <ArrowUpRight size={18} />
            </Link>
          ))}
          {projects.length === 0 && (
            <button className="empty-card" onClick={() => setOpen(true)}>
              <UploadCloud size={24} />
              <strong>Start with a room video</strong>
              <span>Upload an MP4, MOV, or WebM to create your first scene.</span>
            </button>
          )}
        </div>
      </div>
      {open && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal">
            <button className="close" onClick={() => setOpen(false)} aria-label="Close">×</button>
            <p className="eyebrow">NEW PROJECT</p>
            <h2>Bring a room to life.</h2>
            <label>
              Project name
              <input value={name} onChange={(event) => setName(event.target.value)} />
            </label>
            <label className="dropzone">
              {file ? file.name : "Choose a video file"}
              <input
                type="file"
                accept="video/mp4,video/quicktime,video/webm"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            {uploadPct !== null && <p className="muted">Uploading {uploadPct}%…</p>}
            {error && <p className="form-error">{error}</p>}
            <button className="button primary full" disabled={uploadPct !== null} onClick={() => void create()}>
              Upload and build <ArrowUpRight size={16} />
            </button>
          </div>
        </div>
      )}
      {authOpen && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Account authentication">
          <div className="modal">
            <button className="close" type="button" onClick={() => setAuthOpen(false)} aria-label="Close">×</button>
            <p className="eyebrow">{authMode === "login" ? "SIGN IN" : "REGISTER"}</p>
            <h2>{authMode === "login" ? "Welcome back." : "Create an account."}</h2>
            <div style={{ display: "flex", gap: "8px", marginBottom: "8px" }}>
              <button type="button" aria-pressed={authMode === "login"} onClick={() => { setAuthMode("login"); setAuthError(""); }}>
                Sign In
              </button>
              <button type="button" aria-pressed={authMode === "register"} onClick={() => { setAuthMode("register"); setAuthError(""); }}>
                Register
              </button>
            </div>
            <label>
              Email
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
            </label>
            <label>
              Password
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" />
            </label>
            {authError && <p className="form-error">{authError}</p>}
            <button className="button primary full" type="button" disabled={authBusy} onClick={() => void submitAuth()}>
              {authMode === "login" ? "Sign In" : "Create Account"}
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
