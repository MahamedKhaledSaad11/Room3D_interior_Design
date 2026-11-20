export type TaskStatus = "queued" | "claimed" | "leased" | "running" | "completed" | "failed" | "cancelled" | "paused" | "pausing" | "stopping";
export type StageId = "01_extract" | "02_poses" | "03_train" | "04_export" | "05_track" | "06_diffusion" | "07_rebuild";
export type RebuildMode = "masked_finetune" | "from_scratch";
export type DiffusionModelVariant = "flux2_klein_4b" | "sdxl_inpaint_t4";
export interface EditOperationOptions { operation: "recolor" | "generative"; color_hex: string; color_strength: number; quality_preset?: "fast" | "quality"; }

export interface ResolveClickResponse { edit_id: string; ref_frame_idx: number; u: number; v: number; u_norm?: number; v_norm?: number; score: number; method: string; task_id: string; }
export interface EditFrameItem { frame_idx: number; image_relpath: string | null; mask_key: string | null; mask_status: string; skip_frame: boolean; area_px?: number | null; visibility?: string; rejection_reason?: string | null; reviewed?: boolean; confidence?: number | null; }
export interface EditCoverage { total: number; masked: number; skipped: number; missing: number; ready: boolean; }
export interface EditFramesResponse { edit_id: string; page: number; page_size: number; total: number; frames: EditFrameItem[]; coverage?: EditCoverage | null; }
export interface MaskUpdateResponse { edit_id: string; frame_idx: number; status: string; skip_frame: boolean; mask_key?: string | null; }
export interface EditSummaryItem { id: string; project_id: string; base_version_id: string | null; target_version_id: string | null; ref_frame_idx: number | null; prompt: string | null; chosen_seed_id: string | null; status: string; }

export interface Project { processing_status?: "running" | "paused" | "stopped" | "deleting"; id: string; name: string; status: string; active_version_id: string | null; created_at: string; }
export interface TaskSummary { id: string; type: string; status: TaskStatus; progress_pct: number; stage?: StageId; message?: string | null; telemetry?: Telemetry | null; metrics?: Record<string, unknown>; upload_metrics?: Record<string, unknown>; }
export interface Telemetry { vram_used_mb?: number; vram_total_mb?: number; runtime_seconds?: number; [key: string]: number | string | undefined; }
export interface SceneVersion { id: string; label: string; parent_version_id: string | null; created_at: string; }
export interface ProjectDetail extends Project { versions: Array<{ id: string; label: string }>; tasks: TaskSummary[]; edit_sessions: EditSession[]; }
export interface MaskSummaryEntry { frame_idx: number; mask_key?: string | null; status?: string; skip_frame?: boolean; }
export interface EditSession { prompt?: string | null; chosen_seed_id?: string | null; id: string; status?: string; base_version_id?: string | null; target_version_id?: string | null; ref_frame_idx?: number | null; preview_seeds?: Record<string, string>; preview_gallery?: Record<string, string>; preview_settings?: EditOperationOptions & { model_variant?: DiffusionModelVariant; negative_prompt?: string | null }; masks_summary?: MaskSummaryEntry[] | Record<string, unknown>; mask_revision?: number; }
export interface ResolveClickBody {
  screenX?: number;
  screenY?: number;
  worldHitColmap?: [number, number, number] | null;
  rayOriginColmap?: [number, number, number];
  rayDirColmap?: [number, number, number];
  nearestFrameIdx?: number | null;
  point_3d?: number[];
  ray_origin?: number[];
  ray_dir?: number[];
  u_norm?: number;
  v_norm?: number;
  frame_idx?: number;
  base_version_id?: string;
}
export interface Manifest { [key: string]: unknown; }
export interface ViewerClickEvent { screenX: number; screenY: number; worldHitColmap: [number, number, number] | null; rayOriginColmap: [number, number, number]; rayDirColmap: [number, number, number]; nearestFrameIdx: number | null; }
export interface CameraCalibration { width: number; height: number; fx: number; fy: number; cx: number; cy: number; }
export interface ViewerCameraPose { positionColmap: [number, number, number]; targetColmap: [number, number, number]; upColmap: [number, number, number]; fovDeg?: number; calibration?: CameraCalibration; }
export interface ViewerController { onUserClick: (handler: (ev: ViewerClickEvent) => void) => () => void; setCameraPose: (pose: ViewerCameraPose) => void; loadCameraPath: (poses: ViewerCameraPose[], durationMs?: number) => void; setActiveVersion: (versionId: string) => void; }

export class ApiError extends Error { constructor(public readonly status: number, message: string) { super(message); this.name = "ApiError"; } }

export const API_BASE: string = (typeof process !== "undefined" ? process.env.NEXT_PUBLIC_API_BASE : undefined)?.replace(/\/+$/, "") ?? "";
const TOKEN_STORAGE_KEY = "room3d.auth_token";
const assetTokens = new Map<string, string>();
export const AUTH_STATE_EVENT = "room3d:auth-state";
export const AUTH_RESTORED_EVENT = "room3d:auth-restored";
let sessionExpired = false;
export const isSessionExpired = (): boolean => sessionExpired;

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setAuthToken(token: string | null): void {
  assetTokens.clear();
  sessionExpired = false;
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_STORAGE_KEY, token);
  else window.localStorage.removeItem(TOKEN_STORAGE_KEY);
  window.dispatchEvent(new Event(AUTH_STATE_EVENT));
}

function expireSession(bearer: string | null): void {
  if (bearer !== getAuthToken() || sessionExpired) return;
  setAuthToken(null);
  sessionExpired = true;
  window.dispatchEvent(new Event(AUTH_STATE_EVENT));
}

export function appendAuthTokenQuery(url: string): string {
  const parsed = new URL(url, typeof window !== "undefined" ? window.location.origin : "http://localhost");
  const apiOrigin = new URL(API_BASE || "/", parsed.origin).origin;
  if (parsed.origin !== apiOrigin) return url;
  const project = /^\/api\/storage\/([^/]+)\//.exec(parsed.pathname)?.[1];
  const token = project ? assetTokens.get(decodeURIComponent(project)) : undefined;
  if (!token) return url;
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}

/** Full storage keys (mask_key, preview_seeds values) are already project-scoped; never prefix them with assetBaseUrl. */
export function storageKeyUrl(key: string): string {
  return appendAuthTokenQuery(`${API_BASE}/api/storage/${key.replace(/^\/+/, "")}`);
}

export function sceneAssetUrl(projectId: string, versionId: string, relpath: string): string {
  return appendAuthTokenQuery(
    `${API_BASE}/api/storage/${encodeURIComponent(projectId)}/${encodeURIComponent(versionId)}/${relpath.replace(/^\/+/, "")}`,
  );
}

const jsonHeaders = { "Content-Type": "application/json" };
const asErrorMessage = (body: unknown): string => {
  if (typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string") return body.detail;
  return "The request could not be completed.";
};

async function request<T>(path: string, token: string | null, init?: RequestInit): Promise<T> {
  const authentication = /\/api\/auth\/(login|register)$/.test(path);
  if (sessionExpired && !authentication) throw new ApiError(401, "Session expired. Sign in to resume viewing progress; processing continues.");
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const bearer = token ?? getAuthToken();
  if (bearer) headers.set("Authorization", `Bearer ${bearer}`);
  const response = await fetch(path, { ...init, headers });
  const body: unknown = await response.json().catch(() => undefined);
  if (response.status === 401 && !authentication && typeof window !== "undefined") expireSession(bearer);
  if (!response.ok) throw new ApiError(response.status, asErrorMessage(body));
  if (typeof body === "object" && body !== null && "id" in body && typeof body.id === "string"
      && "asset_access" in body && typeof body.asset_access === "string") assetTokens.set(body.id, body.asset_access);
  return body as T;
}

export function validateVideo(file: File, maxBytes = 500 * 1024 * 1024): string | null {
  const allowed = new Set(["video/mp4", "video/quicktime", "video/webm"]);
  if (!allowed.has(file.type)) return "Choose an MP4, QuickTime, or WebM video.";
  if (file.size > maxBytes) return "Video must be 500 MB or smaller.";
  return null;
}

export class RoomApi {
  private get token(): string | null { return getAuthToken(); }
  constructor(private readonly baseUrl = API_BASE) {}
  setToken(token: string | null): void { setAuthToken(token); }
  private path(pathname: string): string { return `${this.baseUrl}${pathname}`; }
  async register(email: string, password: string): Promise<{ access_token: string }> {
    const res = await request<{ access_token: string }>(this.path("/api/auth/register"), this.token, { method: "POST", body: JSON.stringify({ email, password }) });
    if (res.access_token) this.setToken(res.access_token);
    return res;
  }
  async login(email: string, password: string): Promise<{ access_token: string }> {
    const res = await request<{ access_token: string }>(this.path("/api/auth/login"), this.token, { method: "POST", body: JSON.stringify({ email, password }) });
    if (res.access_token) this.setToken(res.access_token);
    return res;
  }
  listProjects(): Promise<Project[]> { return request(this.path("/api/projects"), this.token); }
  createProject(name: string): Promise<Project> { return request(this.path("/api/projects"), this.token, { method: "POST", body: JSON.stringify({ name }) }); }
  getProject(id: string): Promise<ProjectDetail> { return request(this.path(`/api/projects/${encodeURIComponent(id)}`), this.token); }
  controlProject(id: string, action: "pause" | "resume" | "stop"): Promise<Project> {
    return request(this.path(`/api/projects/${encodeURIComponent(id)}/processing`), this.token, { method: "POST", body: JSON.stringify({ action }) });
  }
  deleteProject(id: string): Promise<{ deleted: boolean }> {
    return request(this.path(`/api/projects/${encodeURIComponent(id)}`), this.token, { method: "DELETE" });
  }
  uploadVideo(projectId: string, file: File, onProgress?: (percent: number) => void): Promise<{ project_id: string; task_id: string; bytes: number }> {
    const validationError = validateVideo(file);
    if (validationError) return Promise.reject(new TypeError(validationError));
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const started = performance.now();
      let transferSeconds = 0;
      xhr.upload.onload = () => { transferSeconds = (performance.now() - started) / 1000; };
      xhr.open("POST", this.path(`/api/projects/${encodeURIComponent(projectId)}/upload`));
      const bearer = this.token ?? getAuthToken();
      if (bearer) xhr.setRequestHeader("Authorization", `Bearer ${bearer}`);
      xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(Math.round((event.loaded / event.total) * 100)); };
      xhr.onerror = () => reject(new ApiError(0, "Upload failed. Check your connection and try again."));
      xhr.onload = () => {
        let body: unknown;
        try { body = JSON.parse(xhr.responseText) as unknown; } catch { body = undefined; }
        if (xhr.status < 200 || xhr.status >= 300) { reject(new ApiError(xhr.status, asErrorMessage(body))); return; }
        const uploaded = body as { project_id: string; task_id: string; bytes: number };
        const timing = { browser_transfer_sec: transferSeconds, browser_roundtrip_sec: (performance.now() - started) / 1000 };
        void request(this.path(`/api/projects/${encodeURIComponent(projectId)}/tasks/${encodeURIComponent(uploaded.task_id)}/upload-timing`), this.token,
          { method: "POST", body: JSON.stringify(timing) }).catch((error: unknown) => console.warn("Upload timing could not be recorded", error));
        resolve(uploaded);
      };
      const form = new FormData(); form.append("file", file); xhr.send(form);
    });
  }
  async pollProject(id: string, onUpdate: (project: ProjectDetail) => void, intervalMs = 2000, signal?: AbortSignal): Promise<ProjectDetail> {
    while (!signal?.aborted) {
      const project = await this.getProject(id);
      onUpdate(project);
      const hasActiveTask = project.tasks.some(
        (task) => task.status === "queued" || task.status === "leased" || task.status === "claimed" || task.status === "running",
      );
      if (project.tasks.length > 0 && !hasActiveTask) return project;
      await new Promise<void>((resolve) => {
        const timer = window.setTimeout(resolve, intervalMs);
        signal?.addEventListener("abort", () => { window.clearTimeout(timer); resolve(); }, { once: true });
      });
    }
    throw new DOMException("Polling stopped", "AbortError");
  }
  versions(projectId: string): Promise<SceneVersion[]> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/versions`), this.token); }
  manifest(projectId: string, versionId: string): Promise<Manifest> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}/manifest`), this.token); }
  resolveClick(projectId: string, body: ResolveClickBody): Promise<ResolveClickResponse> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/edits/resolve-click`), this.token, { method: "POST", body: JSON.stringify(body) }); }
  refine(editId: string, points: Array<{ x: number; y: number; label: 1 | 0 }>, frameIdx?: number): Promise<TaskSummary> {
    const body = frameIdx !== undefined ? { points, frame_idx: frameIdx } : { points };
    return request(this.path(`/api/edits/${encodeURIComponent(editId)}/refine`), this.token, { method: "POST", body: JSON.stringify(body) });
  }
  track(editId: string): Promise<TaskSummary> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/track`), this.token, { method: "POST" }); }
  editFrames(editId: string, page = 1, pageSize = 24): Promise<EditFramesResponse> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/frames?page=${encodeURIComponent(page)}&page_size=${encodeURIComponent(pageSize)}`), this.token); }
  updateMask(editId: string, frameIdx: number, body: FormData | { skip_frame: boolean }): Promise<MaskUpdateResponse> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/masks/${encodeURIComponent(frameIdx)}`), this.token, { method: "PUT", headers: body instanceof FormData ? {} : jsonHeaders, body: body instanceof FormData ? body : JSON.stringify(body) }); }
  previewSeeds(editId: string, prompt: string, seeds: number[], model_variant: DiffusionModelVariant, negative_prompt?: string, operation?: EditOperationOptions): Promise<TaskSummary> {
    const payload = { prompt, seeds, model_variant, ...(negative_prompt ? { negative_prompt } : {}), ...operation };
    return request(this.path(`/api/edits/${encodeURIComponent(editId)}/preview-seeds`), this.token, { method: "POST", body: JSON.stringify(payload) });
  }
  applyEdit(
    editId: string,
    chosen_seed_id: string,
    rebuild_mode: RebuildMode,
    prompt?: string,
    options?: Partial<EditOperationOptions> & { model_variant?: DiffusionModelVariant; negative_prompt?: string; rebuild_safeguards?: boolean; unmasked_weight?: number; previews_reviewed?: boolean },
  ): Promise<TaskSummary & { edit_id: string; target_version_id: string; diffusion_task_id: string | null }> {
    const body: {
      chosen_seed_id: string;
      rebuild_mode: RebuildMode;
      prompt?: string;
      model_variant?: DiffusionModelVariant;
      negative_prompt?: string;
      rebuild_safeguards?: boolean;
      unmasked_weight?: number;
    } = { chosen_seed_id, rebuild_mode, prompt };
    if (options?.model_variant) body.model_variant = options.model_variant;
    if (options?.negative_prompt) body.negative_prompt = options.negative_prompt;
    if (options?.rebuild_safeguards !== undefined) body.rebuild_safeguards = options.rebuild_safeguards;
    if (options?.unmasked_weight !== undefined) body.unmasked_weight = options.unmasked_weight;
    return request(this.path(`/api/edits/${encodeURIComponent(editId)}/apply`), this.token, { method: "POST", body: JSON.stringify({ ...body, ...options }) });
  }
  acceptVersion(projectId: string, versionId: string): Promise<{ active_version_id: string }> {
    return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}/accept`), this.token,
      { method: "POST", body: JSON.stringify({ reviewed: true }) });
  }
  restoreVersion(projectId: string, versionId: string): Promise<{ active_version_id: string }> {
    return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}/restore`), this.token,
      { method: "POST", body: JSON.stringify({ reviewed: true }) });
  }
  edits(projectId: string): Promise<EditSummaryItem[]> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/edits`), this.token); }
  async getEdit(projectId: string, editId: string): Promise<EditSession | null> {
    try {
      return await request<EditSession>(
        this.path(`/api/projects/${encodeURIComponent(projectId)}/edits/${encodeURIComponent(editId)}`),
        this.token,
      );
    } catch (err) {
      if (!(err instanceof ApiError) || (err.status !== 404 && err.status !== 405)) {
        throw err;
      }
      const project = await this.getProject(projectId);
      return project.edit_sessions.find((item) => item.id === editId) ?? null;
    }
  }
}

export const roomApi = new RoomApi();
