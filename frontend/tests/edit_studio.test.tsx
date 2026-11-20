import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import EditStudio from "@/components/EditStudio";
import { roomApi, type EditSession, type ProjectDetail, type ViewerClickEvent, type ViewerController } from "@/lib/api";

describe("EditStudio interactive Week 2 workflow", () => {
  let clickCallback: ((ev: ViewerClickEvent) => void) | null = null;

  const project: ProjectDetail = {
    id: "proj-w2",
    name: "Living Room",
    status: "completed",
    active_version_id: "v1",
    created_at: "2026-11-12T00:00:00Z",
    versions: [
      { id: "v1", label: "V1 Base" },
      { id: "v2", label: "V2 Velvet Chair" },
    ],
    tasks: [],
    edit_sessions: [],
  };

  const controller: ViewerController = {
    onUserClick: (handler) => {
      clickCallback = handler;
      return () => {
        clickCallback = null;
      };
    },
    setCameraPose: vi.fn(),
    loadCameraPath: vi.fn(),
    setActiveVersion: vi.fn(),
  };

  beforeEach(() => {
    clickCallback = null;
    vi.restoreAllMocks();
  });

  it("restores the saved generative preset and seed when continuing a failed edit", async () => {
    const previous: EditSession = { id: "failed-edit", status: "failed", base_version_id: "v1", ref_frame_idx: 2,
      prompt: "blue leather", chosen_seed_id: "202", preview_seeds: { "202": "preview.png" },
      preview_settings: { operation: "generative", quality_preset: "quality", color_hex: "#2563eb", color_strength: 1,
        model_variant: "flux2_klein_4b" } };
    vi.spyOn(roomApi, "getEdit").mockResolvedValue(previous);
    vi.spyOn(roomApi, "editFrames").mockResolvedValue({ edit_id: previous.id, page: 1, page_size: 24, total: 1,
      coverage: { total: 1, masked: 1, skipped: 0, missing: 0, ready: true },
      frames: [{ frame_idx: 2, image_relpath: "images/2.png", mask_status: "manual", mask_key: "mask.png", skip_frame: false }] });
    render(<EditStudio projectId="proj-w2" project={{ ...project, edit_sessions: [previous] }}
      manifest={{ width: 640, height: 480, frames: [{ frame_idx: 2, image_relpath: "images/2.png" }] }}
      controller={controller} assetBaseUrl="/api/storage/proj-w2/v1" activeVersionId="v1"
      onSelectVersion={vi.fn()} onRefreshProject={async () => undefined} />);
    fireEvent.change(screen.getByLabelText("Continue previous edit"), { target: { value: previous.id } });
    await waitFor(() => expect(roomApi.editFrames).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("button", { name: /Review views/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: "I checked the selection in different views" }));
    fireEvent.click(screen.getByRole("button", { name: "Continue to color preview" }));
    expect(screen.getByLabelText("Edit operation")).toHaveValue("generative");
    expect(screen.getByLabelText("Sampling preset")).toHaveValue("quality");
    expect(screen.getByLabelText("Chosen Diffusion Seed")).toHaveValue("202");
    expect(screen.getByLabelText("Edit instruction prompt")).toHaveValue("blue leather");
  });

  it("drives click resolve -> refine -> track -> skip/mask upload -> seed preview -> apply -> version switch", async () => {
    let latest: EditSession = { id: "edit-1234", status: "segmented", base_version_id: "v1", ref_frame_idx: 2 };
    vi.spyOn(roomApi, "getEdit").mockImplementation(async () => latest);
    vi.spyOn(roomApi, "resolveClick").mockResolvedValue({
      edit_id: "edit-1234",
      ref_frame_idx: 2,
      u: 120,
      v: 90,
      score: 2.5,
      method: "projection",
      task_id: "task-init",
    });
    const refineSpy = vi.spyOn(roomApi, "refine").mockResolvedValue({ id: "t-ref", type: "segment_refine", status: "queued", progress_pct: 0 });
    const trackSpy = vi.spyOn(roomApi, "track").mockResolvedValue({ id: "t-trk", type: "track_propagate", status: "queued", progress_pct: 0 });
    vi.spyOn(roomApi, "editFrames").mockResolvedValue({
      edit_id: "edit-1234",
      page: 1,
      page_size: 24,
      total: 2,
      coverage: { total: 2, masked: 2, skipped: 0, missing: 0, ready: true },
      frames: [
        { frame_idx: 0, image_relpath: "images/frame_000000.jpg", mask_key: null, mask_status: "auto", skip_frame: false },
        { frame_idx: 2, image_relpath: "images/frame_000002.jpg", mask_key: null, mask_status: "auto", skip_frame: false },
      ],
    });
    const updateMaskSpy = vi.spyOn(roomApi, "updateMask").mockResolvedValue({
      edit_id: "edit-1234",
      frame_idx: 0,
      status: "skipped",
      skip_frame: true,
    });
    const previewSpy = vi.spyOn(roomApi, "previewSeeds").mockImplementation(async () => {
      latest = { ...latest, status: "preview_ready", preview_seeds: Object.fromEntries(
        [101, 202, 303, 404].map((seed) => [String(seed), `proj-w2/edits/edit-1234/seed_previews/${seed}.png`]),
      ) };
      return { id: "t-prev", type: "diffusion_preview", status: "queued", progress_pct: 0 };
    });
    const applySpy = vi.spyOn(roomApi, "applyEdit").mockImplementation(async () => {
      latest = { ...latest, status: "rebuilding", target_version_id: "v2" };
      return {
      id: "t-reb",
      type: "rebuild_week2",
      status: "queued",
      progress_pct: 0,
      edit_id: "edit-1234",
      target_version_id: "v2",
      diffusion_task_id: "t-diff",
      };
    });

    const onSelectVersion = vi.fn();
    const onRefreshProject = vi.fn().mockResolvedValue(undefined);

    render(
      <EditStudio
        projectId="proj-w2"
        project={project}
        manifest={{ width: 640, height: 480, frames: [{ frame_idx: 2, image_relpath: "images/frame_000002.jpg" }] }}
        controller={controller}
        assetBaseUrl="/api/storage/proj-w2/v1"
        activeVersionId="v1"
        onSelectVersion={onSelectVersion}
        onRefreshProject={onRefreshProject}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Select in 3D" }));
    expect(clickCallback).not.toBeNull();
    await act(async () => {
      clickCallback?.({
        screenX: 320,
        screenY: 240,
        worldHitColmap: [0, 0, 3],
        rayOriginColmap: [0, 0, 0],
        rayDirColmap: [0, 0, 1],
        nearestFrameIdx: 2,
      });
    });

    await waitFor(() => expect(screen.getByText(/Selected frame #2/i)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "Remove from selection" }));
    fireEvent.keyDown(screen.getByTestId("ref-frame-canvas"), { key: "Enter" });
    await waitFor(() => expect(refineSpy).toHaveBeenCalledWith("edit-1234", [{ x: 320, y: 240, label: 0 }]));

    await waitFor(() => expect(screen.getByRole("button", { name: /Track this object in all views/i })).toBeEnabled(), { timeout: 4000 });
    fireEvent.click(screen.getByRole("button", { name: /Track this object in all views/i }));
    await waitFor(() => expect(trackSpy).toHaveBeenCalledWith("edit-1234"));

    const skipButtons = screen.getAllByRole("button", { name: "Confirm object absent / occluded" });
    expect(skipButtons[0]).toBeDefined();
    await waitFor(() => expect(skipButtons[0]).toBeEnabled(), { timeout: 4000 });
    fireEvent.click(skipButtons[0] as HTMLElement);
    await waitFor(() => expect(updateMaskSpy).toHaveBeenCalledWith("edit-1234", 0, { skip_frame: true }));

    const fileInput = screen.getByLabelText("Upload PNG mask for frame 2");
    await waitFor(() => expect(fileInput).toBeEnabled());
    const pngFile = new File([new Uint8Array([137, 80, 78, 71])], "mask.png", { type: "image/png" });
    fireEvent.change(fileInput, { target: { files: [pngFile] } });
    await waitFor(() => expect(updateMaskSpy).toHaveBeenCalledTimes(2));

    await waitFor(() => expect(screen.getByRole("checkbox", { name: /I checked the selection/i })).toBeEnabled());
    fireEvent.click(screen.getByRole("checkbox", { name: /I checked the selection/i }));
    fireEvent.click(screen.getByRole("button", { name: "Continue to color preview" }));
    fireEvent.change(screen.getByLabelText("Edit operation"), { target: { value: "generative" } });
    fireEvent.change(screen.getByLabelText("Edit instruction prompt"), { target: { value: "emerald armchair" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate reference edit previews" }));
    await waitFor(() => expect(previewSpy).toHaveBeenCalledWith("edit-1234", "emerald armchair", [101, 202, 303, 404], "flux2_klein_4b", undefined, expect.objectContaining({ operation: "generative" })));
    await waitFor(() => expect(screen.getAllByRole("img", { name: /Preview for/ })).toHaveLength(4));
    fireEvent.click(screen.getByRole("button", { name: "Choose look 2" }));

    fireEvent.click(screen.getByRole("button", { name: "Apply to all views & rebuild room" }));
    await waitFor(() => expect(applySpy).toHaveBeenCalledWith("edit-1234", "202", "masked_finetune", "emerald armchair", expect.objectContaining({ operation: "generative", rebuild_safeguards: true })));
    await waitFor(() => expect(screen.getByText(/Generating targets and building a reviewable candidate/i)).toBeInTheDocument());
    expect(onRefreshProject).toHaveBeenCalled();

    fireEvent.change(screen.getByRole("combobox", { name: "Room version" }), { target: { value: "v2" } });
    expect(onSelectVersion).toHaveBeenCalledWith("v2");
    expect(controller.setActiveVersion).toHaveBeenCalledWith("v2");
    latest = { ...latest, status: "completed" };
    await waitFor(() => expect(screen.getByRole("group", { name: "Before / After" })).toBeInTheDocument(), { timeout: 4000 });
    fireEvent.click(screen.getByRole("button", { name: /^Before$/ }));
    expect(onSelectVersion).toHaveBeenLastCalledWith("v1");
    fireEvent.click(screen.getByRole("button", { name: /^After$/ }));
    expect(onSelectVersion).toHaveBeenLastCalledWith("v2");
  });
});
