import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProgressPanel } from "@/components/ProgressPanel";
import { RoomApi, validateVideo } from "@/lib/api";

describe("video upload validation", () => {
  it("rejects invalid MIME types and oversized videos", () => {
    expect(validateVideo(new File(["x"], "room.txt", { type: "text/plain" }))).toContain("MP4");
    expect(validateVideo(new File([new Uint8Array(6)], "room.mp4", { type: "video/mp4" }), 5)).toContain("500 MB");
    expect(validateVideo(new File(["x"], "room.mp4", { type: "video/mp4" }))).toBeNull();
  });
});

describe("task polling", () => {
  it("transitions from running to completed", async () => {
    const api = new RoomApi();
    const first = { id: "p1", name: "Room", status: "running", active_version_id: null, created_at: "now", versions: [], tasks: [{ id: "t1", type: "reconstruct_week1", status: "running" as const, progress_pct: 42 }], edit_sessions: [] };
    const firstTask = first.tasks[0];
    if (!firstTask) throw new Error("test task missing");
    const second = { ...first, tasks: [{ ...firstTask, status: "completed" as const, progress_pct: 100 }] };
    vi.spyOn(api, "getProject").mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    const updates: number[] = [];
    const result = await api.pollProject("p1", (project) => updates.push(project.tasks[0]?.progress_pct ?? 0), 0);
    expect(updates).toEqual([42, 100]);
    expect(result.tasks[0]?.status).toBe("completed");
  });
});

describe("progress panel", () => {
  it("updates live diffusion and rebuild task percentages", () => {
    const { rerender } = render(<ProgressPanel taskType="diffusion_apply" progress={10} stage="06_diffusion" status="running" message="Loading edit assets" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "10");
    rerender(<ProgressPanel taskType="diffusion_apply" progress={64} stage="06_diffusion" status="running" message="Editing frame 10/16" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "64");
    expect(screen.getByText("Editing frame 10/16")).toBeInTheDocument();
    rerender(<ProgressPanel taskType="rebuild_week2" progress={95} stage="07_rebuild" status="running" message="Rebuilding 3DGS step 5000/5000" />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "95");
    expect(screen.getByText("Rebuilding edited scene")).toBeInTheDocument();
  });

  it("renders the current percentage and stage", () => {
    render(<ProgressPanel progress={45} stage="02_poses" status="running" message="Aligning cameras" />);
    expect(screen.getByText("45%")).toBeInTheDocument();
    expect(screen.getByText("Aligning cameras")).toBeInTheDocument();
    expect(screen.getByText("Align camera poses")).toBeInTheDocument();
  });
});
