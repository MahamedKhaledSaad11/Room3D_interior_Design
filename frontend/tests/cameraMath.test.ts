import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { appendAuthTokenQuery, getAuthToken, RoomApi, sceneAssetUrl, setAuthToken, storageKeyUrl } from "@/lib/api";
import {
  applyDirection4RowMajor,
  applyMatrix4RowMajor,
  cameraFrustum,
  findNearestTrainingFrame,
  interpolateCameraPath,
  invertRigid4x4,
} from "@/lib/cameraMath";
import { applyViewerTransform, assetFailureMessage, createSceneRoot } from "@/lib/splatLoader";

describe("cameraMath", () => {
  it("omits scoped credentials from renderer failure messages", () => {
    const message = assetFailureMessage(new Error('Failed to fetch "https://backend/api/storage/p/v/scene.spz?token=secret": 404'));
    expect(message).not.toContain("secret");
    expect(message).toContain("404");
  });
  it("matches off-center OpenCV intrinsics in the viewer projection", () => {
    const intr = { fx: 730, fy: 820, cx: 460, cy: 290, width: 960, height: 624 };
    const near = 0.05;
    const { left, right, top, bottom } = cameraFrustum(intr, near);
    const projection = new THREE.Matrix4().makePerspective(left, right, top, bottom, near, 100);
    const point = new THREE.Vector3(0.3, -0.2, -2).applyMatrix4(projection);
    expect((point.x + 1) * intr.width / 2).toBeCloseTo(intr.fx * 0.3 / 2 + intr.cx);
    expect((1 - point.y) * intr.height / 2).toBeCloseTo(intr.fy * 0.2 / 2 + intr.cy);
  });
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

  it("applies T_viewer_from_colmap directly via applyViewerTransform and updates child matrixWorld", () => {
    const tViewerFromColmap = [
      [1, 0, 0, 0],
      [0, -1, 0, 2],
      [0, 0, -1, 5],
      [0, 0, 0, 1],
    ];
    const root = new THREE.Group();
    const child = new THREE.Object3D();
    root.add(child);
    applyViewerTransform(root, tViewerFromColmap);
    root.updateMatrixWorld(true);

    expect(root.matrixAutoUpdate).toBe(false);
    const transformed = new THREE.Vector3(1, 3, 4).applyMatrix4(child.matrixWorld);
    expect(transformed.toArray()).toEqual([1, -1, 1]);
  });

  it("builds storageKeyUrl without double-prefixing", () => {
    expect(storageKeyUrl("/proj-1/edits/e1/masks/000000.png")).toBe("/api/storage/proj-1/edits/e1/masks/000000.png");
    expect(storageKeyUrl("proj-1/edits/e1/previews/seed_101.png")).toBe("/api/storage/proj-1/edits/e1/previews/seed_101.png");
  });

  it("uses scoped asset credentials without exposing the session token in URLs", async () => {
    if (typeof window === "undefined") return;
    setAuthToken("jwt-test-token");
    expect(getAuthToken()).toBe("jwt-test-token");
    expect(storageKeyUrl("proj-1/edits/e1/masks/000000.png")).not.toContain("jwt-test-token");
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ id: "proj-1", asset_access: "asset-read-token" })));
    await new RoomApi().getProject("proj-1");
    expect(storageKeyUrl("proj-1/edits/e1/masks/000000.png")).toBe(
      "/api/storage/proj-1/edits/e1/masks/000000.png?token=asset-read-token",
    );
    expect(sceneAssetUrl("proj-1", "v1", "images/frame_000000.jpg")).toBe(
      "/api/storage/proj-1/v1/images/frame_000000.jpg?token=asset-read-token",
    );
    setAuthToken(null);
    expect(getAuthToken()).toBeNull();
    expect(appendAuthTokenQuery("https://external.example/api/storage/proj-1/v1/image.png")).not.toContain("token=");
    fetchSpy.mockRestore();
  });
});
