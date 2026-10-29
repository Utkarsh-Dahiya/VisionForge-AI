"""Tracking diagnostics and metric definitions for Video Intelligence.

Pure Python + pandas: no Streamlit, OpenCV or YOLO imports, so everything here
is unit-testable and reusable by the dashboard and the exports.

IMPORTANT - what these numbers are and are not
----------------------------------------------
* A *detection observation* is one bounding box in one processed frame.
* A *track* is one tracker-assigned ID. IDs are tracker-local estimates, not
  guaranteed real-world identities: occlusion, missed detections and low
  confidence can split one object into several IDs (fragmentation), and
  crowded scenes can occasionally merge or swap IDs.
* "Possible fragmentation" below is a heuristic that flags pairs of tracks
  which *might* be one object. It is a hint for review, never ground truth.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

# One definition per metric so the dashboard, exports and documentation never
# disagree about what a number means.
METRIC_DEFINITIONS: dict[str, str] = {
    "detection_observations": (
        "Bounding boxes summed over every processed frame. One object seen in 100 "
        "frames is 100 observations. NOT a count of objects."),
    "tracked_observations": "Observations that carry a tracker ID.",
    "untracked_observations": (
        "Observations without a tracker ID (tracking off, or the tracker confirmed "
        "no track for that frame)."),
    "frames_with_detections": "Processed frames that contained at least one observation.",
    "frames_with_untracked_boxes": "Processed frames containing at least one observation without an ID.",
    "distinct_track_ids": (
        "Different tracker IDs seen in this run. Includes short-lived and spurious IDs, "
        "and counts one object several times if its ID changed."),
    "estimated_unique_objects": (
        "Distinct tracker IDs seen in at least `min_track_observations` processed frames. "
        "An estimate: ID changes push it up, merged or swapped IDs push it down."),
    "short_track_count": "Tracks seen in fewer than `min_track_observations` processed frames (often flicker).",
    "short_track_fraction": "short_track_count / distinct_track_ids.",
    "tracks_with_gaps": (
        "Tracks that disappear for at least one expected processed frame and then reappear "
        "with the same ID (missed detections or occlusion)."),
    "missing_observation_steps": "Total processed frames in which such tracks were missing.",
    "peak_concurrent_tracks": "Most distinct track IDs visible together in a single processed frame.",
    "class_switching_tracks": "Tracks whose boxes were labelled with more than one class.",
    "possible_fragmentation_links": (
        "Heuristic: pairs of same-class tracks where one ends and another starts nearby "
        "shortly after. Each pair MAY be one object whose ID changed. Not ground truth."),
    "oversized_box_observations": (
        "Observations whose box covers at least `oversized_box_fraction` of the frame - "
        "worth reviewing as possible false positives or very close objects."),
    "degenerate_box_observations": "Observations with zero or negative box width or height.",
}


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Thresholds for the diagnostics. All are heuristics, not model facts."""

    min_track_observations: int = 3
    fragment_max_gap_seconds: float = 1.5
    fragment_max_distance_factor: float = 1.5  # x the larger side of the ended track's last box
    oversized_box_fraction: float = 0.6
    short_track_warning_fraction: float = 0.5
    short_track_warning_min_tracks: int = 5

    def __post_init__(self) -> None:
        if self.min_track_observations < 1:
            raise ValueError("min_track_observations must be at least 1.")
        if self.fragment_max_gap_seconds <= 0:
            raise ValueError("fragment_max_gap_seconds must be positive.")
        if self.fragment_max_distance_factor <= 0:
            raise ValueError("fragment_max_distance_factor must be positive.")
        if not 0 < self.oversized_box_fraction <= 1:
            raise ValueError("oversized_box_fraction must be in (0, 1].")


_NUMERIC = ("frame", "timestamp_seconds", "track_id", "x1", "y1", "x2", "y2")


def _prepare(records: Iterable[dict[str, Any]] | pd.DataFrame | None) -> pd.DataFrame:
    if records is None:
        df = pd.DataFrame()
    elif isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(list(records))
    for column in ("frame", "timestamp_seconds", "track_id", "label", "x1", "y1", "x2", "y2"):
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")
    for column in _NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["label"] = df["label"].fillna("unknown").astype(str)
    return df.dropna(subset=["frame"]).reset_index(drop=True)


def _mode_label(labels: pd.Series) -> str:
    modes = labels.mode()
    return str(modes.iloc[0]) if not modes.empty else str(labels.iloc[0])


def _settings(cfg: DiagnosticsConfig) -> dict[str, Any]:
    return {
        "min_track_observations": cfg.min_track_observations,
        "fragment_max_gap_seconds": cfg.fragment_max_gap_seconds,
        "fragment_max_distance_factor": cfg.fragment_max_distance_factor,
        "oversized_box_fraction": cfg.oversized_box_fraction,
    }


def _empty_result(tracking_enabled: bool, frame_stride: int, fps_assumed: bool,
                  cfg: DiagnosticsConfig) -> dict[str, Any]:
    return {
        "tracking_enabled": bool(tracking_enabled),
        "detection_observations": 0, "tracked_observations": 0, "untracked_observations": 0,
        "frames_with_detections": 0, "frames_with_untracked_boxes": 0,
        "distinct_track_ids": 0, "estimated_unique_objects": 0,
        "estimated_unique_objects_by_class": {},
        "short_track_count": 0, "short_track_fraction": None,
        "track_length_observations": None,
        "tracks_with_gaps": 0, "missing_observation_steps": 0,
        "peak_concurrent_tracks": 0,
        "class_switching_tracks": 0, "class_switch_examples": [],
        "possible_fragmentation_links": 0, "possible_fragmentation_examples": [],
        "oversized_box_observations": 0, "degenerate_box_observations": 0,
        "frame_stride": int(frame_stride), "fps_assumed": bool(fps_assumed),
        "settings": _settings(cfg),
        "warnings": [],
    }


def _warn(out: dict[str, Any], code: str, message: str) -> None:
    out["warnings"].append({"code": code, "message": message})


def _track_table(tracked: pd.DataFrame, frame_stride: int) -> pd.DataFrame:
    """One row per track ID with the facts the diagnostics need."""
    rows = []
    step = max(int(frame_stride), 1)
    for track_id, group in tracked.groupby("track_id", sort=True):
        group = group.sort_values("frame", kind="stable")
        frames = sorted(set(int(f) for f in group["frame"]))
        gap_events = missing = 0
        for previous, current in zip(frames, frames[1:]):
            diff = current - previous
            if diff > step:  # processed frames are `step` apart, so a larger jump means missed updates
                gap_events += 1
                missing += diff // step - 1
        first, last = group.iloc[0], group.iloc[-1]
        rows.append({
            "track_id": int(track_id),
            "label": _mode_label(group["label"]),
            "n_labels": int(group["label"].nunique()),
            "labels": sorted(group["label"].unique().tolist()),
            "observations": int(len(group)),
            "first_frame": frames[0], "last_frame": frames[-1],
            "gap_events": gap_events, "missing_steps": int(missing),
            "first_cx": (float(first["x1"]) + float(first["x2"])) / 2,
            "first_cy": (float(first["y1"]) + float(first["y2"])) / 2,
            "last_cx": (float(last["x1"]) + float(last["x2"])) / 2,
            "last_cy": (float(last["y1"]) + float(last["y2"])) / 2,
            "last_size": max(float(last["x2"]) - float(last["x1"]),
                             float(last["y2"]) - float(last["y1"])),
        })
    return pd.DataFrame(rows)


def _fragmentation_links(tracks: pd.DataFrame, fps: float, cfg: DiagnosticsConfig) -> list[dict[str, Any]]:
    """Greedy one-to-one matching of 'ended' tracks to 'started' tracks.

    Track B is a candidate continuation of track A when: same class, B starts
    after A ended (no overlap in time), the gap is at most
    ``fragment_max_gap_seconds``, and B's first box centre is close to A's last
    box centre. Each track is used at most once so one busy scene cannot
    inflate the count.
    """
    if tracks.empty or fps <= 0:
        return []
    max_gap_frames = cfg.fragment_max_gap_seconds * fps
    ordered = tracks.sort_values("first_frame", kind="stable").reset_index(drop=True)
    starts = ordered["first_frame"].tolist()
    candidates: list[tuple[int, float, int, int]] = []
    for _, a in ordered.iterrows():
        lo = bisect_right(starts, a["last_frame"])  # only tracks starting strictly after A ended
        for b_idx in range(lo, len(ordered)):
            b = ordered.iloc[b_idx]
            gap = float(b["first_frame"] - a["last_frame"])
            if gap > max_gap_frames:
                break  # sorted by start, so every later track is even further away
            if b["label"] != a["label"]:
                continue
            distance = ((b["first_cx"] - a["last_cx"]) ** 2 + (b["first_cy"] - a["last_cy"]) ** 2) ** 0.5
            if distance <= cfg.fragment_max_distance_factor * max(a["last_size"], 1.0):
                candidates.append((int(gap), float(distance), int(a["track_id"]), int(b["track_id"])))
    candidates.sort()
    used_from: set[int] = set()
    used_to: set[int] = set()
    labels = dict(zip(ordered["track_id"].astype(int), ordered["label"]))
    links = []
    for gap, distance, from_id, to_id in candidates:
        if from_id in used_from or to_id in used_to:
            continue
        used_from.add(from_id)
        used_to.add(to_id)
        links.append({"from_track": from_id, "to_track": to_id, "label": str(labels[from_id]),
                      "gap_seconds": round(gap / fps, 3), "distance_px": round(distance, 1)})
    return links


def compute_tracking_diagnostics(
    records: Iterable[dict[str, Any]] | pd.DataFrame | None,
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    frame_stride: int = 1,
    tracking_enabled: bool = True,
    fps_assumed: bool = False,
    config: DiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Summarise how trustworthy the tracking data of one run looks.

    Never raises on empty input or missing IDs; those are valid outcomes.
    """
    cfg = config or DiagnosticsConfig()
    out = _empty_result(tracking_enabled, frame_stride, fps_assumed, cfg)
    df = _prepare(records)
    if fps_assumed:
        _warn(out, "fps_assumed",
              "The video did not report a frame rate, so 25 fps was assumed. All timestamps and "
              "time-based diagnostics are approximate.")
    if frame_stride > 1 and tracking_enabled:
        _warn(out, "stride_tracking",
              f"Frame stride is {int(frame_stride)}: the tracker only sees every {int(frame_stride)}th frame, "
              "so objects move farther between updates and IDs fragment more easily. Use stride 1 "
              "when tracking quality matters.")
    if df.empty:
        return out

    has_id = df["track_id"].notna() & (df["track_id"] >= 0)
    tracked, untracked = df[has_id], df[~has_id]
    out["detection_observations"] = int(len(df))
    out["tracked_observations"] = int(len(tracked))
    out["untracked_observations"] = int(len(untracked))
    out["frames_with_detections"] = int(df["frame"].nunique())
    out["frames_with_untracked_boxes"] = int(untracked["frame"].nunique())

    # Box sanity checks work with or without tracking.
    widths, heights = df["x2"] - df["x1"], df["y2"] - df["y1"]
    out["degenerate_box_observations"] = int(((widths <= 0) | (heights <= 0)).sum())
    if frame_width > 0 and frame_height > 0:
        coverage = (widths.clip(lower=0) * heights.clip(lower=0)) / float(frame_width * frame_height)
        out["oversized_box_observations"] = int((coverage >= cfg.oversized_box_fraction).sum())
    if out["oversized_box_observations"]:
        _warn(out, "oversized_boxes",
              f"{out['oversized_box_observations']} observation(s) cover at least "
              f"{cfg.oversized_box_fraction:.0%} of the frame. Review them: they may be false positives "
              "or objects very close to the camera.")
    if out["degenerate_box_observations"]:
        _warn(out, "degenerate_boxes",
              f"{out['degenerate_box_observations']} observation(s) have a zero or negative box size.")

    if tracked.empty:
        if tracking_enabled:
            _warn(out, "tracking_no_ids",
                  "Tracking was enabled but no track IDs were produced. Detections are still valid, "
                  "but unique-object estimates, line crossings and region events need track IDs. "
                  "Try a lower confidence threshold or a clearer video.")
        return out

    if len(untracked):
        _warn(out, "untracked_observations",
              f"{len(untracked)} observation(s) in {out['frames_with_untracked_boxes']} frame(s) have no track ID. "
              "With Ultralytics tracking, a frame normally returns only boxes that belong to a track, but "
              "when the tracker confirms none for a frame, raw detector boxes can pass through without IDs. "
              "Detection counts can therefore differ slightly from a detector-only run.")

    tracks = _track_table(tracked, frame_stride)
    out["distinct_track_ids"] = int(len(tracks))
    stable = tracks[tracks["observations"] >= cfg.min_track_observations]
    short = tracks[tracks["observations"] < cfg.min_track_observations]
    out["estimated_unique_objects"] = int(len(stable))
    out["estimated_unique_objects_by_class"] = {
        str(k): int(v) for k, v in stable["label"].value_counts().sort_index().items()}
    out["short_track_count"] = int(len(short))
    out["short_track_fraction"] = round(len(short) / len(tracks), 4)
    lengths = tracks["observations"]
    out["track_length_observations"] = {
        "min": int(lengths.min()), "median": float(lengths.median()),
        "mean": round(float(lengths.mean()), 2), "max": int(lengths.max())}
    out["tracks_with_gaps"] = int((tracks["gap_events"] > 0).sum())
    out["missing_observation_steps"] = int(tracks["missing_steps"].sum())
    out["peak_concurrent_tracks"] = int(tracked.groupby("frame")["track_id"].nunique().max())

    switching = tracks[tracks["n_labels"] > 1]
    out["class_switching_tracks"] = int(len(switching))
    out["class_switch_examples"] = [
        {"track_id": int(r.track_id), "labels": list(r.labels)} for r in switching.head(10).itertuples()]
    if len(switching):
        _warn(out, "class_switching",
              f"{len(switching)} track(s) were labelled with more than one class. The most frequent class "
              "is used for summaries; review these tracks if class-wise counts matter.")

    links = _fragmentation_links(tracks, fps, cfg)
    out["possible_fragmentation_links"] = len(links)
    out["possible_fragmentation_examples"] = links[:10]
    if links:
        _warn(out, "possible_fragmentation",
              f"{len(links)} pair(s) of tracks look like one object whose ID may have changed "
              f"(same class, close in space, within {cfg.fragment_max_gap_seconds:g}s). "
              "The unique-object estimate may be too high.")
    if (len(tracks) >= cfg.short_track_warning_min_tracks
            and out["short_track_fraction"] > cfg.short_track_warning_fraction):
        _warn(out, "many_short_tracks",
              f"{out['short_track_fraction']:.0%} of tracks lasted fewer than {cfg.min_track_observations} "
              "processed frames. This usually means flickering detections or ID churn; consider raising "
              "the confidence threshold or lowering the frame stride.")
    return out


def diagnostics_table(diagnostics: dict[str, Any] | None) -> pd.DataFrame:
    """Metric / Value / Meaning table for display and export. Empty if no diagnostics."""
    columns = ["Metric", "Value", "Meaning"]
    if not diagnostics:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, meaning in METRIC_DEFINITIONS.items():
        if key not in diagnostics:
            continue
        value = diagnostics[key]
        if value is None:
            shown = "n/a"
        elif isinstance(value, float) and key.endswith("fraction"):
            shown = f"{value:.1%}"
        else:
            shown = str(value)
        rows.append({"Metric": key.replace("_", " ").capitalize(), "Value": shown, "Meaning": meaning})
    return pd.DataFrame(rows, columns=columns)


def diagnostics_warnings(diagnostics: dict[str, Any] | None) -> list[str]:
    """Human-readable warning messages (empty list when there is nothing to flag)."""
    return [w["message"] for w in (diagnostics or {}).get("warnings", [])]
