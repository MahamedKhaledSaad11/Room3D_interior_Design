"use client";

import { useState } from "react";
import { roomApi, type ProjectDetail } from "@/lib/api";

interface ProjectControlsProps {
  project: ProjectDetail;
  onRefresh: () => Promise<void>;
  onDeleted: () => void;
}

export default function ProjectControls({ project, onRefresh, onDeleted }: ProjectControlsProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pending = project.tasks.some((task) => ["pausing", "stopping"].includes(task.status));
  const working = project.tasks.some((task) => ["queued", "claimed", "running", "leased", "paused"].includes(task.status));
  const paused = project.processing_status === "paused";
  const stopped = project.processing_status === "stopped";
  const canDelete = !project.tasks.some((task) => ["claimed", "running", "leased", "pausing", "stopping"].includes(task.status));

  const perform = async (action: "pause" | "resume" | "stop" | "delete") => {
    setBusy(true);
    setError("");
    try {
      if (action === "delete") {
        await roomApi.deleteProject(project.id);
        onDeleted();
      } else {
        await roomApi.controlProject(project.id, action);
        setConfirmStop(false);
        await onRefresh();
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to update project.");
    } finally {
      setBusy(false);
    }
  };

  return <section className="project-controls" aria-label="Project controls">
    <div className="control-row">
      <strong>{pending ? "Stopping worker…" : paused ? "Processing paused" : stopped ? "Processing stopped" : "Project controls"}</strong>
      {paused || stopped ? <button disabled={busy || pending} onClick={() => void perform("resume")}>{stopped ? "Enable new edits" : "Resume processing"}</button>
        : <button disabled={busy || pending || !working} onClick={() => void perform("pause")}>Pause processing</button>}
      <button disabled={busy || pending || !working} onClick={() => setConfirmStop(true)}>Stop processing</button>
      <button className="danger-button" disabled={busy} onClick={() => setConfirmDelete(true)}>Delete project</button>
    </div>
    {(paused || pending) && <p className="hint-text">Completed tasks are kept. Resume restarts the interrupted task. Worker acknowledgement can take a heartbeat interval.</p>}
    {error && <p role="alert" className="error-text">{error}</p>}
    {confirmStop && <div className="inline-confirm" role="alertdialog" aria-label="Stop project processing">
      <p>Stop this edit and cancel its remaining tasks? Your completed room versions stay available.</p>
      <button disabled={busy} onClick={() => void perform("stop")}>Stop current work</button>
      <button disabled={busy} onClick={() => setConfirmStop(false)}>Keep processing</button>
    </div>}
    {confirmDelete && <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Delete project permanently">
      <div className="modal">
        <h2>Delete {project.name}?</h2>
        <p>This permanently removes the video, room versions, edits, and task history.</p>
        {!canDelete && <p>Stop processing first. Deletion becomes available after the worker acknowledges.</p>}
        {!canDelete && <button disabled={busy || pending} onClick={() => void perform("stop")}>Stop before deleting</button>}
        <button className="danger-button" disabled={busy || !canDelete} onClick={() => void perform("delete")}>Permanently delete project</button>
        <button disabled={busy} onClick={() => setConfirmDelete(false)}>Keep project</button>
      </div>
    </div>}
  </section>;
}
