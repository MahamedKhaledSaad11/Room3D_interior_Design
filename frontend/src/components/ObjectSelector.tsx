"use client";

import React, { useState } from "react";
import { storageKeyUrl } from "@/lib/api";

export interface PromptPoint {
  x: number;
  y: number;
  label: 1 | 0;
}

export interface ObjectSelectorProps {
  editId: string;
  refFrameIdx: number;
  points: PromptPoint[];
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  busy: boolean;
  onAddPoint: (point: PromptPoint) => Promise<void>;
  onConfirmTrack: () => Promise<void>;
  maskKey?: string | null;
  cacheKey?: number | string;
}

export function maskOverlayUrl(maskKey: string, version: number | string): string {
  const url = storageKeyUrl(maskKey);
  return `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(String(version))}`;
}

export default function ObjectSelector({
  editId,
  refFrameIdx,
  points,
  imageUrl,
  imageWidth,
  imageHeight,
  busy,
  onAddPoint,
  onConfirmTrack,
  maskKey,
  cacheKey = 0,
}: ObjectSelectorProps) {
  const [label, setLabel] = useState<1 | 0>(1);
  const [showMask, setShowMask] = useState(true);

  const handleImageClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (busy) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = Math.max(0, Math.min(imageWidth - 1, Math.round(((event.clientX - rect.left) / rect.width) * imageWidth)));
    const y = Math.max(0, Math.min(imageHeight - 1, Math.round(((event.clientY - rect.top) / rect.height) * imageHeight)));
    void onAddPoint({ x, y, label });
  };

  return (
    <section aria-label="Object Mask Selector" style={{ display: "grid", gap: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
        <strong>Ref Frame #{refFrameIdx} ({editId.slice(0, 8)})</strong>
        <div style={{ display: "flex", gap: "6px" }}>
          <button type="button" aria-pressed={label === 1} onClick={() => setLabel(1)} style={{ borderColor: label === 1 ? "#2dd4bf" : undefined }}>
            + Foreground
          </button>
          <button type="button" aria-pressed={label === 0} onClick={() => setLabel(0)} style={{ borderColor: label === 0 ? "#fb7185" : undefined }}>
            - Background
          </button>
          {maskKey ? (
            <button type="button" aria-pressed={showMask} onClick={() => setShowMask((v) => !v)}>
              {showMask ? "Hide Mask" : "Show Mask"}
            </button>
          ) : null}
        </div>
      </div>

      <div
        role="button"
        tabIndex={0}
        aria-label="Reference frame segmentation canvas"
        aria-disabled={busy}
        data-testid="ref-frame-canvas"
        onClick={handleImageClick}
        onKeyDown={(e) => {
          if (busy) return;
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            void onAddPoint({ x: Math.round(imageWidth / 2), y: Math.round(imageHeight / 2), label });
          }
        }}
        style={{ position: "relative", width: "100%", aspectRatio: `${imageWidth}/${imageHeight}`, background: "#0b1220", border: "1px solid #334155", borderRadius: "8px", overflow: "hidden", cursor: "crosshair" }}
      >
        {imageUrl ? <img src={imageUrl} alt={`Reference frame ${refFrameIdx}`} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} /> : null}
        {maskKey && showMask ? (
          <img
            src={maskOverlayUrl(maskKey, `${points.length}-${cacheKey}`)}
            alt={`Segmentation mask for frame ${refFrameIdx}`}
            data-testid="mask-overlay"
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              pointerEvents: "none",
              mixBlendMode: "screen",
              opacity: 0.55,
              filter: "sepia(1) hue-rotate(95deg) saturate(6)",
            }}
          />
        ) : null}
        {points.map((pt, idx) => (
          <span
            key={`${pt.x}-${pt.y}-${idx}`}
            title={`${pt.label === 1 ? "+" : "-"} (${pt.x}, ${pt.y})`}
            style={{
              position: "absolute",
              left: `${(pt.x / Math.max(1, imageWidth)) * 100}%`,
              top: `${(pt.y / Math.max(1, imageHeight)) * 100}%`,
              transform: "translate(-50%, -50%)",
              width: "12px",
              height: "12px",
              borderRadius: "999px",
              background: pt.label === 1 ? "#2dd4bf" : "#fb7185",
              border: "2px solid #090d16",
            }}
          />
        ))}
      </div>

      <button type="button" disabled={busy} onClick={() => void onConfirmTrack()}>
        Confirm Mask &amp; Track Across Frames
      </button>
    </section>
  );
}
