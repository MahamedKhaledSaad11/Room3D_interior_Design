import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { getAuthToken, sceneAssetUrl, setAuthToken, storageKeyUrl } from "@/lib/api";
import { applyMatrix4RowMajor, invertRigid4x4 } from "@/lib/cameraMath";
import { createSceneRoot } from "@/lib/splatLoader";

describe("scene root & storage URL helpers", () => {
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
