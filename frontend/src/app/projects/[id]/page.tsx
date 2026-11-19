"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import ProjectControls from "@/components/ProjectControls";
import { useCallback, useEffect, useState } from "react";
import EditStudio from "@/components/EditStudio";
import { ProgressPanel } from "@/components/ProgressPanel";
import Viewer3D from "@/components/Viewer3D";
import { ThemeToggle } from "@/components/ThemeToggle";
import { API_BASE, roomApi, type Manifest, type ProjectDetail, type TaskSummary, type ViewerController } from "@/lib/api";

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [versionManifest, setVersionManifest] = useState<{ versionId: string; data: Manifest } | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [controller, setController] = useState<ViewerController | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    void params
      .then(({ id }) => {
        if (cancelled) return;
        setProjectId(id);
        return roomApi.getProject(id);
      })
      .then((proj) => {
        if (!cancelled && proj) setProject(proj);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Project unavailable.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [params]);

  const displayVersionId = selectedVersionId ?? project?.active_version_id ?? null;

  useEffect(() => {
    if (!projectId || !displayVersionId) return;
    let cancelled = false;
    void roomApi
      .manifest(projectId, displayVersionId)
      .then((data) => {
        if (!cancelled) setVersionManifest({ versionId: displayVersionId, data });
      })
      .catch(() => {
        if (!cancelled) setVersionManifest(null);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, displayVersionId]);

  const activeManifest =
    displayVersionId && versionManifest?.versionId === displayVersionId ? versionManifest.data : undefined;
  const activeAssetBaseUrl =
    activeManifest && projectId && displayVersionId
      ? `${API_BASE}/api/storage/${encodeURIComponent(projectId)}/${encodeURIComponent(displayVersionId)}`
      : "";

  const refresh = useCallback(async () => {
    if (!projectId) return;
    const updated = await roomApi.getProject(projectId);
    setProject(updated);
  }, [projectId]);

  const activeTask = project?.tasks.find(
    (t) => ["leased", "claimed", "running", "pausing", "stopping"].includes(t.status),
  ) ?? project?.tasks.find((t) => t.status === "queued");
  const task: TaskSummary | undefined = activeTask ?? project?.tasks[0];
  const hasActiveTasks = Boolean(activeTask);

  useEffect(() => {
    if (!projectId || !hasActiveTasks) return;
    let cancelled = false;
    let refreshing = false;
    const timer = window.setInterval(() => {
      if (refreshing) return;
      refreshing = true;
      void roomApi.getProject(projectId)
        .then((next) => { if (!cancelled) setProject(next); })
        .catch(() => undefined)
        .finally(() => { refreshing = false; });
    }, 2500);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [projectId, hasActiveTasks]);

  return (
    <main className="studio-shell">
      <header className="studio-topbar">
        <div className="studio-topbar-left">
          <Link className="brand" href="/">ROOM<span>3D</span></Link>
          <span className="studio-divider">/</span>
          <h1 className="studio-project-title">{project?.name ?? "Loading scene…"}</h1>
          <span className="version-pill">{project?.active_version_id ? "ACTIVE VERSION" : "BASE BUILD"}</span>
        </div>
        <div className="studio-topbar-right">
          <ThemeToggle />
          <Link className="back-link" href="/">← Projects</Link>
        </div>
      </header>
      <div className="studio-body">
        {error ? (
          <div className="studio-error-container">
            <p className="form-error">{error}</p>
          </div>
        ) : (
          <>
            <div className="studio-viewport-container">
              <Viewer3D
                manifest={activeManifest as Parameters<typeof Viewer3D>[0]["manifest"]}
                onControllerReady={setController}
                onVersionChange={setSelectedVersionId}
                assetBaseUrl={activeAssetBaseUrl}
                activeVersionId={displayVersionId ?? undefined}
              />
            </div>
            <aside className="studio-sidebar">
              {project && <ProjectControls project={project} onRefresh={refresh} onDeleted={() => router.push("/")} />}
              {task ? (
                <details className="progress-details" open={hasActiveTasks}>
                  <summary>Processing: {task.status === "completed" ? "finished" : task.status}</summary>
                <ProgressPanel
                  taskType={task.type}
                  progress={task.progress_pct}
                  stage={task.stage}
                  status={task.status}
                  message={task.message}
                  telemetry={task.telemetry}
                />
                </details>
              ) : (
                <ProgressPanel
                  progress={0}
                  status="queued"
                  message="Waiting for the worker to claim this room."
                />
              )}
              {project ? (
                <EditStudio
                  projectId={projectId}
                  project={project}
                  manifest={activeManifest}
                  controller={controller}
                  assetBaseUrl={activeAssetBaseUrl}
                  activeVersionId={displayVersionId}
                  onSelectVersion={setSelectedVersionId}
                  onRefreshProject={refresh}
                  onTasksChanged={() => void refresh().catch(() => undefined)}
                />
              ) : null}
            </aside>
          </>
        )}
      </div>
    </main>
  );
}
