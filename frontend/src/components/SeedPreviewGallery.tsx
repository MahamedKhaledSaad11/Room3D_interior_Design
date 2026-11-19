"use client";

import React, { useState } from "react";
import { storageKeyUrl } from "@/lib/api";

export interface SeedPreviewGalleryProps {
  previewSeeds: Record<string, string>;
  selectedSeed: string | null;
  referenceImageUrl: string | null;
  stale?: boolean;
  cacheKey?: number | string;
  onSelectSeed: (seed: string) => void;
}

const thumbStyle: React.CSSProperties = { width: "100%", aspectRatio: "4/3", objectFit: "cover", borderRadius: "6px", display: "block" };

function previewUrlWithVersion(key: string, cacheKey: number | string): string {
  const url = storageKeyUrl(key);
  return `${url}${url.includes("?") ? "&" : "?"}v=${encodeURIComponent(String(cacheKey))}`;
}

export default function SeedPreviewGallery({
  previewSeeds,
  selectedSeed,
  referenceImageUrl,
  stale = false,
  cacheKey = 0,
  onSelectSeed,
}: SeedPreviewGalleryProps) {
  const [showOriginal, setShowOriginal] = useState(false);
  const entries = Object.entries(previewSeeds);
  if (entries.length === 0) return null;

  return (
    <section aria-label="Seed Preview Gallery" style={{ display: "grid", gap: "8px", opacity: stale ? 0.55 : 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
        <strong>Choose a look ({entries.length})</strong>
        <button type="button" aria-pressed={showOriginal} onClick={() => setShowOriginal((v) => !v)}>
          {showOriginal ? "Showing Original" : "Compare with Original"}
        </button>
      </div>
      {stale ? (
        <p className="hint-text" role="status">
          Prompt or model changed since these previews were generated. Regenerate before applying.
        </p>
      ) : null}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: "8px" }}>
        {entries.map(([seed, key], index) => {
          const selected = seed === selectedSeed;
          const src = showOriginal && referenceImageUrl ? referenceImageUrl : previewUrlWithVersion(key, cacheKey);
          return (
            <button
              key={seed}
              type="button"
              aria-pressed={selected}
              aria-label={`Choose look ${index + 1}`}
              onClick={() => onSelectSeed(seed)}
              style={{ padding: "4px", display: "grid", gap: "4px", borderColor: selected ? "#2dd4bf" : undefined, borderWidth: selected ? 2 : 1 }}
            >
              <img src={src} alt={`Preview for ${seed}`} data-testid={`seed-preview-img-${seed}`} style={thumbStyle} />
              <span style={{ fontSize: "11px" }}>Look {index + 1}{selected ? " · Selected" : ""}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
