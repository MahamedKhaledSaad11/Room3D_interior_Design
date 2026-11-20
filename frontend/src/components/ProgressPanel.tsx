"use client";

import { AlertTriangle, CheckCircle2, Clock3, RotateCcw } from "lucide-react";
import type { StageId, TaskStatus, Telemetry } from "@/lib/api";

const WEEK1_STAGES: Array<{ id: StageId; label: string; threshold: number }> = [
  { id: "01_extract", label: "Extract keyframes", threshold: 15 },
  { id: "02_poses", label: "Align camera poses", threshold: 45 },
  { id: "03_train", label: "Train Gaussian scene", threshold: 85 },
  { id: "04_export", label: "Export viewer bundle", threshold: 100 },
];

const WEEK2_STAGES: Array<{ id: StageId; label: string; threshold: number }> = [
  { id: "05_track", label: "Track object masks", threshold: 25 },
  { id: "06_diffusion", label: "Apply object edit", threshold: 60 },
  { id: "07_rebuild", label: "Rebuild edited scene", threshold: 100 },
];

export interface ProgressPanelProps {
  progress: number;
  stage?: StageId;
  status: TaskStatus | string;
  message?: string | null;
  telemetry?: Telemetry | null;
  onRetry?: () => void;
  taskType?: string;
  metrics?: Record<string, unknown>;
  uploadMetrics?: Record<string, unknown>;
}

const formatRuntime = (seconds?: number): string =>
  seconds === undefined ? "—" : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;

export function ProgressPanel({ progress, stage, status, message, telemetry, onRetry, taskType, metrics, uploadMetrics }: ProgressPanelProps) {
  const failed = status === "failed";
  const done = status === "completed";
  const controlHeadings: Readonly<Record<string, string>> = { paused: "Processing paused", pausing: "Pausing processing…", stopping: "Stopping processing…", cancelled: "Processing stopped" };
  const controlHeading = controlHeadings[status];
  const isWeek2 = stage?.startsWith("05_") || stage?.startsWith("06_") || stage?.startsWith("07_");
  const doneHeading = stage === "07_rebuild" ? "Candidate built — review required" : !isWeek2 ? "Room ready"
    : taskType === "diffusion_preview" ? "Previews ready" : stage === "05_track" ? "Selection ready" : "Edited views ready";
  const ownStage = isWeek2 && taskType ? WEEK2_STAGES.find((item) => item.id === stage) : undefined;
  const activeStages = ownStage ? [{ ...ownStage, threshold: 100 }] : isWeek2 ? WEEK2_STAGES : WEEK1_STAGES;
  const heading = ownStage?.id === "06_diffusion" ? "Editing your object" : ownStage?.id === "07_rebuild" ? "Rebuilding edited scene" : "Building your room";

  return (
    <section className="progress-panel" aria-label="Scene processing progress">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{isWeek2 ? "OBJECT EDIT" : "ROOM BUILD"}</p>
          <h2>{controlHeading ?? (failed ? "Build needs attention" : done ? doneHeading : heading)}</h2>
        </div>
        <strong>{Math.round(progress)}%</strong>
      </div>
      {(metrics || uploadMetrics) && <details><summary>Measured stage timings</summary>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{JSON.stringify({ upload: uploadMetrics,
          worker_wall_sec: metrics?.worker_wall_sec, queue_wait_sec: metrics?.queue_wait_sec,
          stage_timings: metrics?.stage_timings, io_timings: metrics?.io_timings,
          frames_inpainted: metrics?.frames_inpainted, sec_per_generated_frame: metrics?.sec_per_generated_frame,
          quality_acceptance: metrics?.quality_acceptance }, null, 2)}</pre>
      </details>}
      <div className="progress-track" role="progressbar" aria-label="Task progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}>
        <span style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
      </div>
      <div className="stage-list">
        {activeStages.map((item, index) => {
          const active = item.id === stage;
          const complete = progress >= item.threshold || (done && index <= activeStages.length - 1);
          return (
            <div className={`stage ${active ? "active" : ""} ${complete ? "complete" : ""}`} key={item.id}>
              <span className="stage-number">
                {complete ? <CheckCircle2 size={15} /> : String(index + 1).padStart(2, "0")}
              </span>
              <span>{item.label}</span>
              {active && <span className="stage-live">LIVE</span>}
            </div>
          );
        })}
      </div>
      <div className={`status-message ${failed ? "error" : ""}`}>
        {failed ? <AlertTriangle size={17} /> : <Clock3 size={17} />}
        <span>{message || (done ? "Your scene is ready to explore." : "Processing is running in the background.")}</span>
      </div>
      <div className="telemetry">
        {telemetry?.operations_total !== undefined && <p>
          Inference passes: {telemetry.operations_completed ?? 0}/{telemetry.operations_total}
          {telemetry.frames_to_inpaint !== undefined && <> · {telemetry.frames_to_inpaint} views need diffusion</>}
          {telemetry.operations_resumed !== undefined && <> · {telemetry.operations_resumed} passes resumed</>}
        </p>}
        {!done && !failed && typeof telemetry?.eta_sec === "number" && <p>Estimated inference remaining: {formatRuntime(telemetry.eta_sec)} · updates from measured steps; upload and rebuild follow.</p>}
        <h3>Runtime telemetry</h3>
        <table>
          <tbody>
            <tr>
              <th>VRAM</th>
              <td>
                {telemetry?.vram_used_mb !== undefined
                  ? `${telemetry.vram_used_mb} / ${telemetry.vram_total_mb ?? "—"} MB`
                  : "—"}
              </td>
            </tr>
            <tr>
              <th>Elapsed</th>
              <td>{formatRuntime(telemetry?.runtime_seconds)}</td>
            </tr>
            <tr>
              <th>Worker status</th>
              <td>{status}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {failed && onRetry && (
        <button className="button secondary retry" onClick={onRetry}>
          <RotateCcw size={16} /> Try again
        </button>
      )}
    </section>
  );
}
