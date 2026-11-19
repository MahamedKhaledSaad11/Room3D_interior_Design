import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import EditStudio from "@/components/EditStudio";
import { roomApi, type EditSession, type ProjectDetail } from "@/lib/api";

afterEach(() => vi.restoreAllMocks());

it("reviews diverse recolor previews and keeps a candidate separate from active-room acceptance", async () => {
  let latest: EditSession = { id: "paint", status: "tracked", base_version_id: "v1", ref_frame_idx: 0, mask_revision: 2 };
  const project: ProjectDetail = { id: "room", name: "Room", status: "completed", active_version_id: "v1", created_at: "now",
    versions: [{ id: "v1", label: "Original" }], tasks: [], edit_sessions: [latest] };
  vi.spyOn(roomApi, "getEdit").mockImplementation(async () => latest);
  vi.spyOn(roomApi, "editFrames").mockResolvedValue({ edit_id: "paint", page: 1, page_size: 24, total: 2,
    coverage: { total: 2, masked: 2, skipped: 0, missing: 0, ready: true }, frames: [0, 1].map((frame_idx) => ({
      frame_idx, image_relpath: `images/${frame_idx}.png`, mask_key: `room/edits/paint/masks/${frame_idx}.png`,
      mask_status: "masked", area_px: 100, skip_frame: false,
    })) });
  const preview = vi.spyOn(roomApi, "previewSeeds").mockImplementation(async () => {
    latest = { ...latest, status: "preview_ready", preview_seeds: { recolor: "room/preview0.png" },
      preview_gallery: { "0": "room/preview0.png", "1": "room/preview1.png" } };
    return { id: "preview", type: "diffusion_preview", status: "queued", progress_pct: 0 };
  });
  const apply = vi.spyOn(roomApi, "applyEdit").mockImplementation(async () => {
    latest = { ...latest, status: "candidate", target_version_id: "v2" };
    return { id: "build", type: "rebuild_week2", status: "queued", progress_pct: 0,
      edit_id: "paint", target_version_id: "v2", diffusion_task_id: null };
  });
  const accept = vi.spyOn(roomApi, "acceptVersion").mockResolvedValue({ active_version_id: "v2" });
  const select = vi.fn();
  render(<EditStudio projectId="room" project={project} manifest={{ width: 64, height: 64 }} controller={null}
    assetBaseUrl="/api/storage/room/v1" activeVersionId="v1" onSelectVersion={select} onRefreshProject={vi.fn().mockResolvedValue(undefined)} />);
  fireEvent.change(screen.getByLabelText("Continue previous edit"), { target: { value: "paint" } });
  await waitFor(() => expect(screen.getByRole("button", { name: /2 Review views/ })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: /2 Review views/ }));
  await waitFor(() => expect(screen.getByRole("checkbox", { name: /I checked the selection/ })).toBeEnabled());
  fireEvent.click(screen.getByRole("checkbox", { name: /I checked the selection/ }));
  fireEvent.click(screen.getByRole("button", { name: "Continue to color preview" }));
  expect(screen.getByLabelText("Edit operation")).toHaveValue("recolor");
  fireEvent.click(screen.getByRole("button", { name: "Preview paint in diverse views" }));
  await waitFor(() => expect(preview).toHaveBeenCalledWith("paint", "Paint selected surfaces #2563eb", [101, 202, 303, 404],
    "flux2_klein_4b", undefined, expect.objectContaining({ operation: "recolor" })));
  await waitFor(() => expect(screen.getAllByRole("img", { name: /Recolor preview frame/ })).toHaveLength(2));
  expect(screen.getByRole("button", { name: "Build recolor candidate" })).toBeDisabled();
  fireEvent.click(screen.getByRole("checkbox", { name: /intended color in all previews/ }));
  fireEvent.click(screen.getByRole("button", { name: "Build recolor candidate" }));
  await waitFor(() => expect(apply).toHaveBeenCalledWith("paint", "recolor", "masked_finetune", "Paint selected surfaces #2563eb",
    expect.objectContaining({ operation: "recolor", previews_reviewed: true })));
  await waitFor(() => expect(screen.getByRole("button", { name: /accept this room/ })).toBeVisible());
  expect(select).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: /accept this room/ }));
  await waitFor(() => expect(accept).toHaveBeenCalledWith("room", "v2"));
  await waitFor(() => expect(select).toHaveBeenCalledWith("v2"));
});
