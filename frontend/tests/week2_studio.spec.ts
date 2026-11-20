import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

test("live edit masks, manual fallback, four seeds, and rebuilt Before / After", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let status = "segmented";
  let skipped = false;
  let tracked = false;
  let manualUploads = 0;
  let refinedFrame: number | null = null;
  let chosenSeed: string | null = null;
  let previews: Record<string, string> = {};
  const masks = { "0": { mask_key: "live/edits/edit/masks/000000.png" } };
  const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const frames = [0, 1].map((index) => ({ frame_idx: index, image_relpath: `images/${index}.png`, cam_from_world: identity }));
  const versions = [{ id: "v1", label: "Original" }, { id: "v2", label: "Edited" }];

  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path.startsWith("/api/storage/")) {
      if (path.endsWith(".json")) return route.fulfill({ json: [{ position: [0, 0, 2], color: [200, 180, 160] }] });
      if (!path.endsWith(".png")) return route.fulfill({ status: 404, body: "fallback" });
      const seed = path.match(/seed_(\d+)\.png$/)?.[1];
      const file = seed ? `seed_${seed}.png` : path.includes("/masks/") ? "mask.png" : "room.png";
      return route.fulfill({ contentType: "image/png", body: readFileSync(join(process.cwd(), "tests", "fixtures", file)) });
    }
    if (path === "/api/projects/live") return route.fulfill({ json: {
      id: "live", name: "Live Room", status: "completed", active_version_id: status === "completed" ? "v2" : "v1",
      created_at: "2026-11-17T00:00:00Z", versions: status === "completed" ? versions : versions.slice(0, 1),
      tasks: [], edit_sessions: [],
    } });
    if (path.endsWith("/manifest")) return route.fulfill({ json: {
      version_id: path.includes("/v2/") ? "v2" : "v1", width: 64, height: 64, frames,
      T_viewer_from_colmap: identity, artifacts: { spz_relpath: "scene.spz", sparse_points_relpath: "sparse_points.json" },
    } });
    if (path.endsWith("/resolve-click")) return route.fulfill({ status: 201, json: {
      edit_id: "edit", ref_frame_idx: 0, u: 32, v: 32, method: "frame", score: 1, task_id: "segment",
    } });
    if (path.endsWith("/frames")) return route.fulfill({ json: {
      edit_id: "edit", page: 1, page_size: 24, total: 2,
      coverage: { total: 2, masked: tracked ? (skipped ? 1 : 2) : 1, skipped: skipped ? 1 : 0, missing: tracked ? 0 : 1, ready: tracked },
      frames: frames.map((frame) => ({ ...frame, mask_status: "auto", skip_frame: frame.frame_idx === 1 && skipped,
        mask_key: tracked || frame.frame_idx === 0 ? `live/edits/edit/masks/${String(frame.frame_idx).padStart(6, "0")}.png` : null })),
    } });
    if (path.endsWith("/refine")) { refinedFrame = request.postDataJSON().frame_idx; status = "segmented"; }
    else if (path.endsWith("/track")) { tracked = true; status = "tracked"; }
    else if (path.includes("/masks/")) {
      if (request.headers()["content-type"]?.includes("multipart/form-data")) manualUploads++;
      else skipped = Boolean(request.postDataJSON().skip_frame);
      return route.fulfill({ json: { edit_id: "edit", frame_idx: 1, status: "manual", skip_frame: skipped } });
    } else if (path.endsWith("/preview-seeds")) {
      expect(request.postDataJSON().seeds).toEqual([101, 202, 303, 404]);
      previews = Object.fromEntries([101, 202, 303, 404].map((seed) => [String(seed), `live/edits/edit/seed_previews/seed_${seed}.png`]));
      status = "preview_ready";
    } else if (path.endsWith("/apply")) {
      chosenSeed = request.postDataJSON().chosen_seed_id;
      status = "rebuilding";
      return route.fulfill({ json: { id: "rebuild", type: "rebuild_week2", status: "queued", progress_pct: 0,
        edit_id: "edit", target_version_id: "v2", diffusion_task_id: "diffusion" } });
    } else if (path.endsWith("/edits/edit")) {
      if (status === "rebuilding") status = "completed";
      return route.fulfill({ json: { id: "edit", status, base_version_id: "v1", target_version_id: status === "completed" ? "v2" : null,
        ref_frame_idx: 0, masks_summary: masks, preview_seeds: previews } });
    } else return route.fulfill({ status: 404, json: { detail: `Unexpected route ${path}` } });
    return route.fulfill({ json: { id: "task", status: "queued", progress_pct: 0 } });
  });

  await page.goto("/projects/live");
  await page.getByTestId("frame-picker-canvas").click();
  await expect(page.getByTestId("mask-overlay")).toBeVisible();
  await expect.poll(() => page.getByTestId("mask-overlay").evaluate((image: HTMLImageElement) => image.naturalWidth)).toBe(64);
  await page.getByRole("button", { name: "Track this object in all views" }).click();
  await expect(page.getByTestId("mask-thumb-1")).toBeVisible();
  await page.getByRole("button", { name: "Mark object in frame 1" }).click();
  await page.getByTestId("ref-frame-canvas").press("Enter");
  await expect.poll(() => refinedFrame).toBe(1);
  await page.getByRole("button", { name: /Review views/ }).click();
  const frame = page.locator("article").filter({ hasText: "Frame #1" });
  await frame.getByRole("button", { name: "Confirm object absent / occluded", exact: true }).click();
  await expect(frame.getByRole("button", { name: "Object is visible — correct mask" })).toBeVisible();
  await frame.getByRole("button", { name: "Object is visible — correct mask" }).click();
  await page.getByLabel("Upload PNG mask for frame 1").setInputFiles("tests/fixtures/mask.png");
  await expect.poll(() => manualUploads).toBe(1);
  await page.screenshot({ path: "../.local_data/edit-studio-review.png", fullPage: true });
  await page.getByRole("checkbox", { name: "I checked the selection in different views" }).check();
  await page.getByRole("button", { name: "Continue to color preview" }).click();
  await page.getByLabel("Edit operation").selectOption("generative");
  await page.getByLabel("Edit instruction prompt").fill("emerald velvet chair");
  await page.getByRole("button", { name: "Generate reference edit previews" }).click();
  for (const seed of [101, 202, 303, 404]) {
    await expect.poll(() => page.getByTestId(`seed-preview-img-${seed}`).evaluate((image: HTMLImageElement) => image.naturalWidth)).toBe(64);
  }
  await page.getByRole("button", { name: "Choose look 2", exact: true }).click();
  await page.getByRole("button", { name: "Apply to all views & rebuild room" }).click();
  await expect.poll(() => chosenSeed).toBe("202");
  await page.getByRole("button", { name: "After", exact: true }).click();
  await expect(page.locator(".viewer3d")).toHaveAttribute("data-version-id", "v2");
  await page.getByRole("button", { name: "Before", exact: true }).click();
  await expect(page.locator(".viewer3d")).toHaveAttribute("data-version-id", "v1");
  await page.getByRole("button", { name: "After", exact: true }).click();
  await expect(page.locator(".viewer3d")).toHaveAttribute("data-version-id", "v2");
  expect(errors).toEqual([]);
});

test("project controls pause, resume, stop, and confirm permanent deletion", async ({ page }) => {
  let processing = "running";
  let taskStatus = "running";
  let pendingReads = 0;
  let removed = false;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/processing")) {
      const action = route.request().postDataJSON().action;
      processing = action === "pause" ? "paused" : action === "stop" ? "stopped" : "running";
      taskStatus = action === "pause" ? "pausing" : action === "stop" ? "cancelled" : "running";
      return route.fulfill({ json: {} });
    }
    if (path === "/api/projects/controls") {
      if (route.request().method() === "DELETE") {
        removed = true;
        return route.fulfill({ json: { deleted: true } });
      }
      if (taskStatus === "pausing" && ++pendingReads > 1) taskStatus = "paused";
      return route.fulfill({ json: {
        id: "controls", name: "Controls test", status: "running", processing_status: processing,
        active_version_id: null, created_at: "2026-11-19", versions: [], edit_sessions: [],
        tasks: [{ id: "task", type: "diffusion_apply", stage: "06_diffusion", status: taskStatus, progress_pct: 50 }],
      } });
    }
    return route.fulfill({ json: [] });
  });
  await page.goto("/projects/controls");
  await page.getByRole("button", { name: "Pause processing", exact: true }).click();
  await expect(page.getByRole("button", { name: "Resume processing" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Resume processing" })).toBeEnabled();
  await page.getByRole("button", { name: "Resume processing" }).click();
  await expect(page.getByRole("button", { name: "Pause processing", exact: true })).toBeEnabled();
  await page.getByRole("button", { name: "Stop processing", exact: true }).click();
  await page.getByRole("button", { name: "Stop current work" }).click();
  await expect(page.getByRole("button", { name: "Enable new edits" })).toBeEnabled();
  await page.getByRole("button", { name: "Delete project", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Delete project permanently" })).toBeVisible();
  expect(removed).toBe(false);
  await page.getByRole("button", { name: "Permanently delete project" }).click();
  await expect(page).toHaveURL(/\/$/);
  expect(removed).toBe(true);
});
