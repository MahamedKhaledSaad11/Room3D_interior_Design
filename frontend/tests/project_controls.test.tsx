import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import ProjectControls from "@/components/ProjectControls";
import { roomApi, type ProjectDetail } from "@/lib/api";

const project: ProjectDetail = {
  id: "room", name: "Playroom", status: "completed", active_version_id: "v1", created_at: "2026-11-19",
  processing_status: "running", tasks: [{ id: "task", type: "diffusion_apply", status: "running", progress_pct: 30 }],
  versions: [{ id: "v1", label: "Original" }], edit_sessions: [],
};

afterEach(() => vi.restoreAllMocks());

it("pauses and resumes processing through the server", async () => {
  const control = vi.spyOn(roomApi, "controlProject").mockResolvedValue(project);
  const refresh = vi.fn().mockResolvedValue(undefined);
  const { rerender } = render(<ProjectControls project={project} onRefresh={refresh} onDeleted={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Pause processing" }));
  await waitFor(() => expect(control).toHaveBeenCalledWith("room", "pause"));
  await waitFor(() => expect(refresh).toHaveBeenCalled());
  rerender(<ProjectControls project={{ ...project, processing_status: "paused", tasks: [] }} onRefresh={refresh} onDeleted={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "Resume processing" }));
  await waitFor(() => expect(control).toHaveBeenCalledWith("room", "resume"));
});

it("requires deletion confirmation and worker acknowledgement", async () => {
  const remove = vi.spyOn(roomApi, "deleteProject").mockResolvedValue({ deleted: true });
  const deleted = vi.fn();
  const refresh = vi.fn().mockResolvedValue(undefined);
  const { rerender } = render(<ProjectControls project={project} onRefresh={refresh} onDeleted={deleted} />);
  fireEvent.click(screen.getByRole("button", { name: "Delete project" }));
  expect(remove).not.toHaveBeenCalled();
  expect(screen.getByRole("button", { name: "Permanently delete project" })).toBeDisabled();
  rerender(<ProjectControls project={{ ...project, processing_status: "stopped", tasks: [] }} onRefresh={refresh} onDeleted={deleted} />);
  fireEvent.click(screen.getByRole("button", { name: "Permanently delete project" }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith("room"));
  expect(deleted).toHaveBeenCalled();
});
