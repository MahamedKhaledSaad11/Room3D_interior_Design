from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel


class EditCoverage(BaseModel):
    total: int
    masked: int
    skipped: int
    missing: int
    ready: bool


def mask_coverage(frame_ids: Sequence[int], summary: Mapping[str, object]) -> EditCoverage:
    masked = skipped = 0
    for frame_idx in frame_ids:
        entry = summary.get(str(frame_idx))
        if not isinstance(entry, dict):
            continue
        if entry.get("skip_frame"):
            skipped += 1
        elif entry.get("mask_key"):
            masked += 1
    missing = len(frame_ids) - masked - skipped
    return EditCoverage(total=len(frame_ids), masked=masked, skipped=skipped,
                        missing=missing, ready=masked > 0 and missing == 0)
