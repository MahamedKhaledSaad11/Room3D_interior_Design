import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { getAuthToken, sceneAssetUrl, setAuthToken, storageKeyUrl } from "@/lib/api";
import {
  applyDirection4RowMajor,
  applyMatrix4RowMajor,
  findNearestTrainingFrame,
  interpolateCameraPath,
  invertRigid4x4,
} from "@/lib/cameraMath";
import { createSceneRoot } from "@/lib/splatLoader";

describe("cameraMath", () => {
  it("round trips points between colmap and viewer coordinates", () => {
    const transform = [0, -1, 0, 2, 1, 0, 0, 3, 0, 0, 1, 4, 0, 0, 0, 1];
    const point = [1, 2, 3];
    expect(applyMatrix4RowMajor(invertRigid4x4(transform), applyMatrix4RowMajor(transform, point))).toEqual(point);
  });

  it("rotates direction vectors without adding translation", () => {
    const transform = [0, -1, 0, 25, 1, 0, 0, -40, 0, 0, 1, 12, 0, 0, 0, 1];
    expect(applyDirection4RowMajor(transform, [0, 1, 0])).toEqual([-1, 0, 0]);
  });

  it("ranks the training frame aligned with the click ray", () => {
    const frames = [
      { frame_idx: 2, cam_from_world: [[1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] },
      { frame_idx: 8, cam_from_world: [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] },
    ];
    expect(findNearestTrainingFrame([1, 0, 0], [0, 0, 0], [1, 0, 0], frames)).toBe(8);
  });

  it("interpolates camera position and orientation", () => {
    const result = interpolateCameraPath(
      [
        { positionColmap: [0, 0, 0], targetColmap: [0, 0, -1], upColmap: [0, 1, 0] },
        { positionColmap: [10, 0, 0], targetColmap: [0, 0, -2], upColmap: [0, 1, 0], fovDeg: 80 },
      ],
      0.5,
    );
    expect(result.positionColmap[0]).toBe(5);
    expect(result.targetColmap[2]).toBe(-1.5);
    expect(result.fovDeg).toBe(70);
  });

  it("applies row-major T_viewer_from_colmap once to sceneRoot", () => {
    const tViewerFromColmap = [
      [1, 0, 0, 2],
      [0, -1, 0, 3],
      [0, 0, -1, 4],
      [0, 0, 0, 1],
    ];
    const root = createSceneRoot(tViewerFromColmap);
    expect(root.name).toBe("sceneRoot");
    expect(root.matrixAutoUpdate).toBe(false);

    const pColmap = new THREE.Vector3(1, 2, 5);
    const pViewer = pColmap.clone().applyMatrix4(root.matrix);
    expect(pViewer.toArray()).toEqual([3, 1, -1]);

    const expected = applyMatrix4RowMajor(tViewerFromColmap.flat(), [1, 2, 5]);
    expect(pViewer.toArray()).toEqual(expected);
    expect(applyMatrix4RowMajor(invertRigid4x4(tViewerFromColmap.flat()), pViewer.toArray())).toEqual([1, 2, 5]);
  });

  it("keeps identity sceneRoot when T_viewer_from_colmap is missing or invalid", () => {
    const root = createSceneRoot(undefined);
    expect(root.matrix.equals(new THREE.Matrix4())).toBe(true);
  });

  it("builds storageKeyUrl without double-prefixing", () => {
    expect(storageKeyUrl("/proj-1/edits/e1/masks/000000.png")).toBe("/api/storage/proj-1/edits/e1/masks/000000.png");
    expect(storageKeyUrl("proj-1/edits/e1/previews/seed_101.png")).toBe("/api/storage/proj-1/edits/e1/previews/seed_101.png");
  });

  it("persists JWT auth token and appends ?token= to storage/scene asset URLs when signed in", () => {
    if (typeof window === "undefined") return;
    setAuthToken("jwt-test-token");
    expect(getAuthToken()).toBe("jwt-test-token");
    expect(storageKeyUrl("proj-1/edits/e1/masks/000000.png")).toBe(
      "/api/storage/proj-1/edits/e1/masks/000000.png?token=jwt-test-token",
    );
    expect(sceneAssetUrl("proj-1", "v1", "images/frame_000000.jpg")).toBe(
      "/api/storage/proj-1/v1/images/frame_000000.jpg?token=jwt-test-token",
    );
    setAuthToken(null);
    expect(getAuthToken()).toBeNull();
  });
});
