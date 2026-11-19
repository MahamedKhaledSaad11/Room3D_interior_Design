from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel


class EditCoverage(BaseModel):
    total: int
    masked: int
    skipped: int
    missing: int
    ready: bool


def mask_state(entry: object) -> str:
    if not isinstance(entry, dict):
        return "unresolved"
    confirmed_absence = (entry.get("reviewed") is True or
                         entry.get("absence_evidence") == "trusted_depth_and_empty_raw_sam")
    if entry.get("skip_frame"):
        return "absent" if confirmed_absence else "unresolved"
    if entry.get("visibility") in {"absent", "occluded", "out_of_frustum"}:
        return "absent" if confirmed_absence else "unresolved"
    area = entry.get("area_px")
    if (isinstance(area, int) and not isinstance(area, bool) and area > 0
            and entry.get("mask_key") and entry.get("content_sha256")
            and entry.get("visibility") == "visible" and not entry.get("rejection_reason")):
        return "masked"
    return "unresolved"


def mask_coverage(frame_ids: Sequence[int], summary: Mapping[str, object]) -> EditCoverage:
    masked = skipped = 0
    for frame_idx in frame_ids:
        entry = summary.get(str(frame_idx))
        state = mask_state(entry)
        if state == "absent":
            skipped += 1
        elif state == "masked":
            masked += 1
    missing = len(frame_ids) - masked - skipped
    return EditCoverage(total=len(frame_ids), masked=masked, skipped=skipped,
                        missing=missing, ready=masked > 0 and missing == 0)
