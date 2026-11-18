"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ObjectSelector, { type PromptPoint } from "@/components/ObjectSelector";
import SeedPreviewGallery from "@/components/SeedPreviewGallery";
import TrackingGrid from "@/components/TrackingGrid";
import {
  API_BASE,
  ApiError,
  appendAuthTokenQuery,
  roomApi,
  sceneAssetUrl,
  type DiffusionModelVariant,
  type EditFramesResponse,
  type EditSession,
  type Manifest,
  type MaskSummaryEntry,
  type ProjectDetail,
  type RebuildMode,
  type ViewerCameraPose,
  type ViewerClickEvent,
  type ViewerController,
} from "@/lib/api";
import { applyDirection4RowMajor, applyMatrix4RowMajor, invertRigid4x4 } from "@/lib/cameraMath";

export interface EditStudioProps {
  projectId: string;
  project: ProjectDetail;
  manifest?: Manifest;
  controller: ViewerController | null;
  assetBaseUrl: string;
  activeVersionId: string | null;
  onSelectVersion: (versionId: string) => void;
  onRefreshProject: () => Promise<void>;
  onTasksChanged?: () => void;
  onJumpToFrame?: (frameIdx: number) => void;
}

interface ManifestFrame {
  frame_idx: number;
  image_relpath?: string;
  cam_from_world?: number[][] | null;
  excluded?: boolean;
}

const DEFAULT_SEEDS = [101, 202, 303, 404];
const ACTIVE_EDIT_STATUSES = new Set(["segmenting", "tracking", "previewing", "rebuilding", "Edited"]);

function resolveSeedKey(previewSeeds: Record<string, string>, chosenSeed: string): string | null {
  if (chosenSeed in previewSeeds) return chosenSeed;
  const bare = chosenSeed.replace(/^seed_/, "");
  if (bare in previewSeeds) return bare;
  const prefixed = `seed_${bare}`;
  if (prefixed in previewSeeds) return prefixed;
  return null;
}

function manifestImageSize(manifest?: Manifest): { width: number; height: number } {
  const width = typeof manifest?.width === "number" ? manifest.width : 640;
  const height = typeof manifest?.height === "number" ? manifest.height : 480;
  return { width, height };
}

function unexcludedFramesFrom(manifest?: Manifest): ManifestFrame[] {
  const rawFrames = Array.isArray(manifest?.frames) ? manifest.frames : [];
  return rawFrames.filter(
    (item): item is ManifestFrame =>
      typeof item === "object" && item !== null && typeof (item as ManifestFrame).frame_idx === "number" && !(item as ManifestFrame).excluded,
  );
}

function registeredFramesFrom(manifest?: Manifest): ManifestFrame[] {
  const unexcluded = unexcludedFramesFrom(manifest);
  const hasAnyExtrinsic = unexcluded.some((f) => Array.isArray(f.cam_from_world));
  return unexcluded.filter((item) => {
    if (item.cam_from_world === null) return false;
    return hasAnyExtrinsic ? Array.isArray(item.cam_from_world) : true;
  });
}

function frameRelpathFor(frames: readonly ManifestFrame[], frameIdx: number): string | null {
  const hit = frames.find((item) => item.frame_idx === frameIdx);
  return typeof hit?.image_relpath === "string" ? hit.image_relpath : null;
}

function poseFromManifestFrame(frame: ManifestFrame): ViewerCameraPose | null {
  const m = frame.cam_from_world;
  if (!Array.isArray(m) || m.length < 3) return null;
  const flat = [
    m[0]?.[0] ?? 1, m[0]?.[1] ?? 0, m[0]?.[2] ?? 0, m[0]?.[3] ?? 0,
    m[1]?.[0] ?? 0, m[1]?.[1] ?? 1, m[1]?.[2] ?? 0, m[1]?.[3] ?? 0,
    m[2]?.[0] ?? 0, m[2]?.[1] ?? 0, m[2]?.[2] ?? 1, m[2]?.[3] ?? 0,
    0, 0, 0, 1,
  ];
  const c2w = invertRigid4x4(flat);
  return {
    positionColmap: applyMatrix4RowMajor(c2w, [0, 0, 0]),
    targetColmap: applyMatrix4RowMajor(c2w, [0, 0, 2]),
    upColmap: applyDirection4RowMajor(c2w, [0, -1, 0]),
  };
}

function maskKeyFromSession(session: EditSession | null, framesData: EditFramesResponse | null, frameIdx: number): string | null {
  const fromFrames = framesData?.frames.find((f) => f.frame_idx === frameIdx)?.mask_key;
  if (fromFrames) return fromFrames;
  const summary = session?.masks_summary;
  if (!summary) return null;
  if (Array.isArray(summary)) {
    return summary.find((entry) => entry.frame_idx === frameIdx)?.mask_key ?? null;
  }
  const keyed = (summary as Record<string, MaskSummaryEntry | undefined>)[String(frameIdx)];
  return typeof keyed?.mask_key === "string" ? keyed.mask_key : null;
}

export default function EditStudio({
  projectId,
  project,
  manifest,
  controller,
  assetBaseUrl,
  activeVersionId,
  onSelectVersion,
  onRefreshProject,
  onTasksChanged,
  onJumpToFrame,
}: EditStudioProps) {
  const [editMode, setEditMode] = useState(false);
  const [preferredSelectionMode, setSelectionMode] = useState<"3d" | "2d">("3d");
  const [requestedPickerFrameIdx, setPickerFrameIdx] = useState<number>(0);
  const [editId, setEditId] = useState<string | null>(null);
  const [editSession, setEditSession] = useState<EditSession | null>(null);
  const [baseManifest, setBaseManifest] = useState<{ versionId: string; data: Manifest } | null>(null);
  const [pollingEdit, setPollingEdit] = useState(false);
  const [initialRefFrameIdx, setInitialRefFrameIdx] = useState<number>(0);
  const [refFrameIdx, setRefFrameIdx] = useState<number>(0);
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [framesData, setFramesData] = useState<EditFramesResponse | null>(null);
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [modelVariant, setModelVariant] = useState<DiffusionModelVariant>("flux2_klein_4b");
  const [previewSignature, setPreviewSignature] = useState<string | null>(null);
  const pendingPreviewSigRef = useRef<string | null>(null);
  const [chosenSeed, setChosenSeed] = useState("seed_101");
  const [rebuildMode, setRebuildMode] = useState<RebuildMode>("masked_finetune");
  const [busy, setBusy] = useState(false);
  const [maskRevision, setMaskRevision] = useState(0);
  const framesRevisionRef = useRef("");
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState("");

  const editManifest = useMemo(
    () => (editSession?.base_version_id && editSession.base_version_id !== activeVersionId
      && baseManifest?.versionId === editSession.base_version_id ? baseManifest.data : undefined) ?? manifest,
    [activeVersionId, baseManifest, editSession?.base_version_id, manifest],
  );
  const viewerUnexcludedFrames = useMemo(() => unexcludedFramesFrom(manifest), [manifest]);
  const allFrames = useMemo(() => unexcludedFramesFrom(editManifest), [editManifest]);
  const frames = useMemo(() => registeredFramesFrom(manifest), [manifest]);
  const pickerFrames = useMemo(() => (frames.length > 0 ? frames : viewerUnexcludedFrames), [frames, viewerUnexcludedFrames]);

  const selectionMode = frames.length === 0 && viewerUnexcludedFrames.length > 0 ? "2d" : preferredSelectionMode;
  const pickerFrameIdx = pickerFrames.some((frame) => frame.frame_idx === requestedPickerFrameIdx)
    ? requestedPickerFrameIdx : pickerFrames[0]?.frame_idx ?? 0;

  useEffect(() => {
    const baseVersionId = editSession?.base_version_id;
    if (!baseVersionId || baseVersionId === activeVersionId) {
      return;
    }
    let cancelled = false;
    void roomApi
      .manifest(projectId, baseVersionId)
      .then((data) => {
        if (!cancelled) setBaseManifest({ versionId: baseVersionId, data });
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to load the original scene.");
      });
    return () => {
      cancelled = true;
    };
  }, [activeVersionId, editSession?.base_version_id, projectId]);

  const loadFrames = useCallback(async (targetEditId: string, page = 1) => {
    const data = await roomApi.editFrames(targetEditId, page);
    setFramesData(data);
  }, []);

  const notifyTasks = useCallback(() => {
    onTasksChanged?.();
  }, [onTasksChanged]);

  const runAction = useCallback(async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Operation failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  const startEditFromResolve = useCallback(
    async (payload: Parameters<typeof roomApi.resolveClick>[1]) => {
      const res = await roomApi.resolveClick(projectId, payload);
      setEditId(res.edit_id);
      setEditSession({ id: res.edit_id, status: "segmenting", base_version_id: activeVersionId, ref_frame_idx: res.ref_frame_idx });
      setInitialRefFrameIdx(res.ref_frame_idx);
      setRefFrameIdx(res.ref_frame_idx);
      setPreviewSignature(null);
      pendingPreviewSigRef.current = null;
      setPoints([{ x: Math.round(res.u), y: Math.round(res.v), label: 1 }]);
      setStatusMessage(`Selected reference frame #${res.ref_frame_idx}`);
      setPollingEdit(true);
      notifyTasks();
      await loadFrames(res.edit_id, 1);
    },
    [activeVersionId, loadFrames, notifyTasks, projectId],
  );

  const handleViewerClick = useCallback(
    async (click: ViewerClickEvent) => {
      await runAction(async () => {
        await startEditFromResolve({
          ...click,
          ...(activeVersionId ? { base_version_id: activeVersionId } : {}),
        });
      });
    },
    [activeVersionId, runAction, startEditFromResolve],
  );

  useEffect(() => {
    if (!controller || !editMode || selectionMode !== "3d") return;
    return controller.onUserClick((click) => {
      void handleViewerClick(click);
    });
  }, [controller, editMode, handleViewerClick, selectionMode]);

  useEffect(() => {
    if (!editId || !pollingEdit) return;
    let cancelled = false;
    let refreshing = false;
    const pollOnce = async () => {
      if (refreshing || cancelled) return;
      refreshing = true;
      try {
        const latest = await roomApi.getEdit(projectId, editId);
        if (cancelled || !latest) return;
        setEditSession(latest);
        setMaskRevision(latest.mask_revision ?? 0);
        const previewMap = latest.preview_seeds ?? {};
        const seeds = Object.keys(previewMap);
        if (latest.status === "preview_ready" || (seeds.length > 0 && latest.status !== "previewing")) {
          if (pendingPreviewSigRef.current) {
            setPreviewSignature(pendingPreviewSigRef.current);
            pendingPreviewSigRef.current = null;
          }
        }
        if (seeds.length > 0) {
          const matched = resolveSeedKey(previewMap, chosenSeed);
          setChosenSeed(matched ?? seeds[0]!);
        }
        const frameRevision = `${editId}:${framesData?.page ?? 1}:${latest.mask_revision ?? latest.status ?? "legacy"}`;
        if (framesRevisionRef.current !== frameRevision) {
          await loadFrames(editId, framesData?.page ?? 1);
          framesRevisionRef.current = frameRevision;
        }
        if (latest.status && !ACTIVE_EDIT_STATUSES.has(latest.status)) {
          setPollingEdit(false);
          if (latest.status === "completed") {
            await onRefreshProject();
            if (latest.target_version_id) {
              onSelectVersion(latest.target_version_id);
              setStatusMessage("Rebuild complete. Compare Before and After below.");
            }
          }
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : "Unable to refresh edit progress.");
      } finally {
        refreshing = false;
      }
    };
    void pollOnce();
    const timer = window.setInterval(() => {
      void pollOnce();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [chosenSeed, editId, framesData?.page, loadFrames, pollingEdit, projectId, onRefreshProject, onSelectVersion]);

  const handlePickerImageClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (busy) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const u_norm = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const v_norm = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
    void runAction(async () => {
      await startEditFromResolve({
        frame_idx: pickerFrameIdx,
        nearestFrameIdx: pickerFrameIdx,
        u_norm,
        v_norm,
        ...(activeVersionId ? { base_version_id: activeVersionId } : {}),
      });
    });
  };

  const handleJumpToPickerFrame = () => {
    onJumpToFrame?.(pickerFrameIdx);
    const frame = pickerFrames.find((f) => f.frame_idx === pickerFrameIdx);
    const pose = frame ? poseFromManifestFrame(frame) : null;
    if (pose) controller?.setCameraPose(pose);
  };

  const handleAddPoint = async (pt: PromptPoint) => {
    if (!editId) return;
    await runAction(async () => {
      await (refFrameIdx !== initialRefFrameIdx
        ? roomApi.refine(editId, [...points, pt], refFrameIdx)
        : roomApi.refine(editId, [pt]));
      setPoints((prev) => [...prev, pt]);
      setEditSession((prev) => (prev ? { ...prev, preview_seeds: {} } : prev));
      setPreviewSignature(null);
      setStatusMessage("Refined segmentation mask.");
      setPollingEdit(true);
      notifyTasks();
    });
  };

  const handleTrack = async () => {
    if (!editId) return;
    await runAction(async () => {
      await roomApi.track(editId);
      setEditSession((prev) => (prev ? { ...prev, preview_seeds: {} } : prev));
      setPreviewSignature(null);
      setStatusMessage("Queued cross-frame tracking.");
      setPollingEdit(true);
      notifyTasks();
      await loadFrames(editId, 1);
    });
  };

  const invalidateMasksAndSync = async (targetEditId: string) => {
    setPreviewSignature(null);
    pendingPreviewSigRef.current = null;
    setMaskRevision((v) => v + 1);
    setEditSession((prev) => (prev ? { ...prev, preview_seeds: {} } : prev));
    await loadFrames(targetEditId, framesData?.page ?? 1);
    const latest = await roomApi.getEdit(projectId, targetEditId);
    if (latest) {
      setEditSession(latest);
      setMaskRevision(latest.mask_revision ?? 0);
    }
  };

  const handleToggleSkip = async (frameIdx: number, skip: boolean) => {
    if (!editId) return;
    await runAction(async () => {
      await roomApi.updateMask(editId, frameIdx, { skip_frame: skip });
      await invalidateMasksAndSync(editId);
    });
  };

  const handleUploadMask = async (frameIdx: number, file: File) => {
    if (!editId) return;
    const isPng = file.name.toLowerCase().endsWith(".png") && (!file.type || file.type === "image/png");
    if (!isPng) {
      setError("Custom mask must be a 1-channel uint8 PNG file (.png).");
      return;
    }
    if (file.size === 0 || file.size > 16 * 1024 * 1024) {
      setError(file.size === 0 ? "Custom mask file cannot be empty." : "Custom mask exceeds maximum allowed size of 16 MB.");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    await runAction(async () => {
      await roomApi.updateMask(editId, frameIdx, form);
      await invalidateMasksAndSync(editId);
    });
  };

  const currentSignature = `${prompt.trim()}|${negativePrompt.trim()}|${modelVariant}`;

  const handlePreviewSeeds = async () => {
    if (!editId || !prompt.trim()) {
      setError("Enter an edit prompt before generating previews.");
      return;
    }
    await runAction(async () => {
      setPreviewSignature(null);
      pendingPreviewSigRef.current = currentSignature;
      await roomApi.previewSeeds(editId, prompt.trim(), DEFAULT_SEEDS, modelVariant, negativePrompt.trim() || undefined);
      setEditSession((prev) => (prev ? { ...prev, status: "previewing" } : prev));
      setStatusMessage("Seed preview generation queued.");
      setPollingEdit(true);
      notifyTasks();
    });
  };

  const handleApplyRebuild = async () => {
    if (!editId || !prompt.trim()) {
      setError("Enter a prompt and choose a seed before applying rebuild.");
      return;
    }
    await runAction(async () => {
      const extraOpts =
        modelVariant !== "flux2_klein_4b" || negativePrompt.trim().length > 0
          ? {
              model_variant: modelVariant,
              ...(negativePrompt.trim() ? { negative_prompt: negativePrompt.trim() } : {}),
            }
          : undefined;
      const res = extraOpts
        ? await roomApi.applyEdit(editId, selectedSeedValue, rebuildMode, prompt.trim(), extraOpts)
        : await roomApi.applyEdit(editId, selectedSeedValue, rebuildMode, prompt.trim());
      setStatusMessage(`Rebuild queued for version ${res.target_version_id}`);
      setPollingEdit(true);
      notifyTasks();
      await onRefreshProject();
    });
  };

  const { width, height } = manifestImageSize(editManifest);
  const { width: pickerWidth, height: pickerHeight } = manifestImageSize(manifest);
  const editBaseVersionId = editSession?.base_version_id ?? null;
  const editAssetBaseUrl = editBaseVersionId
    ? `${API_BASE}/api/storage/${encodeURIComponent(projectId)}/${encodeURIComponent(editBaseVersionId)}`
    : assetBaseUrl;
  const relpath = frameRelpathFor(allFrames, refFrameIdx);
  const refImageUrl = relpath
    ? editBaseVersionId
      ? sceneAssetUrl(projectId, editBaseVersionId, relpath)
      : assetBaseUrl
        ? appendAuthTokenQuery(`${assetBaseUrl.replace(/\/+$/, "")}/${relpath.replace(/^\/+/, "")}`)
        : null
    : null;
  const pickerRelpath = frameRelpathFor(pickerFrames, pickerFrameIdx);
  const pickerImageUrl = pickerRelpath
    ? activeVersionId
      ? sceneAssetUrl(projectId, activeVersionId, pickerRelpath)
      : assetBaseUrl
        ? appendAuthTokenQuery(`${assetBaseUrl.replace(/\/+$/, "")}/${pickerRelpath.replace(/^\/+/, "")}`)
        : null
    : null;
  const refMaskKey = maskKeyFromSession(editSession, framesData, refFrameIdx);
  const previewSeeds = editSession?.preview_seeds ?? {};
  const previewEntries = Object.entries(previewSeeds);
  const resolvedSeedKey = resolveSeedKey(previewSeeds, chosenSeed);
  const selectedSeedValue = resolvedSeedKey ?? chosenSeed;
  const seedOptions = previewEntries.length > 0
    ? Object.keys(previewSeeds)
    : DEFAULT_SEEDS.map((s) => `seed_${s}`);
  const stalePreviews = previewSignature !== null && previewSignature !== currentSignature;
  const applyDisabled =
    busy
    || stalePreviews
    || previewEntries.length === 0
    || !resolvedSeedKey
    || editSession?.status === "previewing";

  return (
    <section className="glass-card" aria-label="Week 2 Edit Studio" style={{ padding: "16px", display: "grid", gap: "14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
        <div>
          <p className="eyebrow">WEEK 2 EDIT STUDIO</p>
          <strong>Interactive Object Editing</strong>
        </div>
        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
          <button
            type="button"
            aria-pressed={editMode && selectionMode === "3d"}
            onClick={() => {
              setSelectionMode("3d");
              setEditMode((v) => !v);
            }}
            style={{
              borderColor: editMode ? "#2dd4bf" : "#3b82f6",
              background: editMode ? "#064e3b" : "#1e293b",
              color: editMode ? "#6ee7b7" : "#f8fafc",
              fontWeight: 700,
              fontSize: "12px",
              padding: "8px 14px",
              borderRadius: "8px",
              borderWidth: "2px",
            }}
          >
            {editMode ? "🎯 Edit Object Mode: ON (Selection Active)" : "✏️ Edit Object Mode: OFF (Click to Enable)"}
          </button>
          {pickerFrames.length > 0 ? (
            <button
              type="button"
              aria-pressed={selectionMode === "2d"}
              onClick={() => setSelectionMode((m) => (m === "2d" ? "3d" : "2d"))}
              style={{ fontSize: "12px", padding: "8px 10px" }}
            >
              {selectionMode === "2d" ? "2D Frame Picker: ON" : "2D Frame Picker"}
            </button>
          ) : null}
        </div>
      </div>

      {selectionMode === "2d" && pickerFrames.length > 0 ? (
        <div aria-label="2D Frame Picker" style={{ display: "grid", gap: "8px", border: "1px solid #334155", borderRadius: "8px", padding: "10px" }}>
          <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
            <label style={{ display: "flex", gap: "6px", alignItems: "center" }}>
              <span>Frame:</span>
              <select
                aria-label="Select reference frame"
                value={pickerFrameIdx}
                onChange={(e) => setPickerFrameIdx(Number(e.target.value))}
              >
                {pickerFrames.map((frame) => (
                  <option key={frame.frame_idx} value={frame.frame_idx}>{`Frame #${frame.frame_idx}`}</option>
                ))}
              </select>
            </label>
            <button type="button" onClick={handleJumpToPickerFrame}>
              Snap 3D Camera to Frame
            </button>
          </div>
          <div
            role="button"
            tabIndex={0}
            aria-label="2D frame picker canvas"
            data-testid="frame-picker-canvas"
            onClick={handlePickerImageClick}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && !busy) {
                e.preventDefault();
                void runAction(() =>
                  startEditFromResolve({
                    frame_idx: pickerFrameIdx,
                    nearestFrameIdx: pickerFrameIdx,
                    u_norm: 0.5,
                    v_norm: 0.5,
                    ...(activeVersionId ? { base_version_id: activeVersionId } : {}),
                  }),
                );
              }
            }}
            style={{ position: "relative", width: "100%", aspectRatio: `${pickerWidth}/${pickerHeight}`, background: "#0b1220", borderRadius: "6px", overflow: "hidden", cursor: "crosshair" }}
          >
            {pickerImageUrl ? (
              <img src={pickerImageUrl} alt={`Frame ${pickerFrameIdx}`} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
            ) : (
              <span style={{ padding: "12px", display: "block", fontSize: "12px" }}>Click to select Frame #{pickerFrameIdx}</span>
            )}
          </div>
        </div>
      ) : null}

      {editMode && selectionMode === "3d" && !editId ? (
        <div style={{ background: "#064e3b33", border: "1px solid #10b981", borderRadius: "8px", padding: "10px", color: "#6ee7b7", fontSize: "13px" }}>
          👉 <strong>Object Selection is Active:</strong> Click anywhere on the sofa or table in the 3D room to select it and start editing!
        </div>
      ) : null}
      {!editMode && selectionMode === "3d" && !editId ? (
        <p style={{ color: "var(--muted)", fontSize: "12px", margin: 0 }}>
          💡 To select an object for editing: Click <strong>&quot;Edit Object Mode: OFF&quot;</strong> above to turn it ON, then click any object in the 3D room.
        </p>
      ) : null}

      {project.versions.length > 0 ? (
        <div role="group" aria-label="Scene Version Switcher" style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
          <span>A/B Versions:</span>
          {project.versions.map((ver) => {
            const active = ver.id === activeVersionId;
            return (
              <button
                key={ver.id}
                type="button"
                aria-pressed={active}
                onClick={() => {
                  onSelectVersion(ver.id);
                  controller?.setActiveVersion(ver.id);
                }}
                style={{ borderColor: active ? "#2dd4bf" : undefined }}
              >
                {ver.label || ver.id}
              </button>
            );
          })}
        </div>
      ) : null}

      {editSession?.status === "completed" && editSession.base_version_id && editSession.target_version_id ? (
        <div role="group" aria-label="Before / After">
          {[["Before", editSession.base_version_id], ["After", editSession.target_version_id]].map(([label, version]) => (
            <button key={label} type="button" aria-pressed={activeVersionId === version}
              onClick={() => onSelectVersion(version!)}>{label}</button>
          ))}
        </div>
      ) : null}

      {editMode && selectionMode === "3d" && !editId ? <p className="hint-text">Click any object inside the 3D scene to start masked segmentation.</p> : null}
      {statusMessage ? <p className="hint-text">{statusMessage}</p> : null}
      {error ? <p role="alert" className="error-text">{error}</p> : null}

      {editId ? (
        <div style={{ display: "grid", gap: "14px" }}>
          <ObjectSelector
            editId={editId}
            refFrameIdx={refFrameIdx}
            points={points}
            imageUrl={refImageUrl}
            maskKey={refMaskKey}
            cacheKey={maskRevision}
            imageWidth={width}
            imageHeight={height}
            busy={busy}
            onAddPoint={handleAddPoint}
            onConfirmTrack={handleTrack}
          />

          {framesData ? (
            <TrackingGrid
              assetBaseUrl={editAssetBaseUrl}
              framesData={framesData}
              busy={busy}
              cacheKey={maskRevision}
              onPageChange={(page) => runAction(() => loadFrames(editId, page))}
              onToggleSkip={handleToggleSkip}
              onUploadMask={handleUploadMask}
              onSelectFrameForRefine={(frameIdx) => {
                setRefFrameIdx(frameIdx);
                setPoints([]);
                setStatusMessage(`Switched reference canvas to frame #${frameIdx} — click object to refine.`);
              }}
            />
          ) : null}

          <div style={{ display: "grid", gap: "10px", borderTop: "1px solid #334155", paddingTop: "10px" }}>
            <label style={{ display: "grid", gap: "4px" }}>
              <span>Edit Prompt</span>
              <input
                type="text"
                aria-label="Edit instruction prompt"
                maxLength={1000}
                value={prompt}
                placeholder="e.g. modern emerald velvet armchair"
                onChange={(e) => setPrompt(e.target.value)}
              />
            </label>

            <label style={{ display: "grid", gap: "4px" }}>
              <span>Negative Prompt (optional)</span>
              <input
                type="text"
                aria-label="Negative edit prompt"
                maxLength={500}
                value={negativePrompt}
                placeholder="e.g. blurry, distorted geometry"
                onChange={(e) => setNegativePrompt(e.target.value)}
              />
            </label>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
              <label>
                Model:
                <select
                  aria-label="Diffusion Model Variant"
                  value={modelVariant}
                  onChange={(e) => setModelVariant(e.target.value as DiffusionModelVariant)}
                >
                  <option value="flux2_klein_4b">FLUX.2 [klein] 4B (Apache-2.0)</option>
                  <option value="sdxl_inpaint_t4">SDXL Inpaint (T4 Fallback)</option>
                </select>
              </label>

              <label>
                Seed:
                <select aria-label="Chosen Diffusion Seed" value={selectedSeedValue} onChange={(e) => setChosenSeed(e.target.value)}>
                  {seedOptions.map((seed) => (
                    <option key={seed} value={seed}>{seed}</option>
                  ))}
                </select>
              </label>

              <label>
                Rebuild Mode:
                <select
                  aria-label="Rebuild Mode"
                  value={rebuildMode}
                  onChange={(e) => setRebuildMode(e.target.value as RebuildMode)}
                >
                  <option value="masked_finetune">Masked Fine-Tune (5k steps)</option>
                  <option value="from_scratch">From Scratch Ablation (15k steps)</option>
                </select>
              </label>
            </div>

            {Object.keys(previewSeeds).length > 0 ? (
              <SeedPreviewGallery
                previewSeeds={previewSeeds}
                selectedSeed={selectedSeedValue}
                referenceImageUrl={refImageUrl}
                stale={stalePreviews}
                cacheKey={maskRevision}
                onSelectSeed={(seed) => setChosenSeed(seed)}
              />
            ) : null}

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <button type="button" disabled={busy} onClick={() => void handlePreviewSeeds()}>
                Generate Seed Previews
              </button>
              <button type="button" disabled={applyDisabled} onClick={() => void handleApplyRebuild()}>
                Apply &amp; Rebuild 3D Room
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
