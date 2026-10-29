"""Rule-based event detection for tracked video detections.

Events are derived only from observed track IDs and bounding-box coordinates.
Line crossing uses centroid movement between consecutive observations; region
entry/exit uses the centroid's position relative to a normalized rectangle.
"""
from __future__ import annotations
from typing import Any, Iterable
import pandas as pd

REQUIRED = {"frame", "timestamp_seconds", "track_id", "label", "x1", "y1", "x2", "y2"}
EVENT_COLUMNS = ["timestamp_seconds", "frame", "event_type", "track_id", "label", "details"]


def detect_video_events(
    records: Iterable[dict[str, Any]] | pd.DataFrame,
    *,
    frame_width: int,
    frame_height: int,
    line_y_ratio: float = 0.5,
    region: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75),
) -> pd.DataFrame:
    """Return line-crossing and region entry/exit events for tracked objects.

    line_y_ratio is the horizontal line's vertical position in [0, 1].
    region is normalized (left, top, right, bottom), each in [0, 1].
    Rows without valid track IDs are ignored rather than inventing identities.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    if not 0 <= line_y_ratio <= 1:
        raise ValueError("line_y_ratio must be between 0 and 1")
    if len(region) != 4 or not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1):
        raise ValueError("region must be (left, top, right, bottom) normalized to [0, 1]")
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if df.empty or not REQUIRED.issubset(df.columns):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = df.copy()
    df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
    for col in ("frame", "timestamp_seconds", "x1", "y1", "x2", "y2"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(REQUIRED - {"label"})).sort_values(["track_id", "frame"], kind="stable")
    df = df[df["track_id"] >= 0]
    if df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    left, top, right, bottom = region
    line_y = line_y_ratio * frame_height
    events: list[dict[str, Any]] = []
    previous: dict[int, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        tid = int(row["track_id"])
        cx = (float(row["x1"]) + float(row["x2"])) / 2
        cy = (float(row["y1"]) + float(row["y2"])) / 2
        inside = left * frame_width <= cx <= right * frame_width and top * frame_height <= cy <= bottom * frame_height
        prev = previous.get(tid)
        base = {"timestamp_seconds": round(float(row["timestamp_seconds"]), 3), "frame": int(row["frame"]),
                "track_id": tid, "label": str(row.get("label", "unknown"))}
        if prev is not None:
            if (prev["cy"] < line_y <= cy) or (prev["cy"] > line_y >= cy):
                direction = "down" if cy > prev["cy"] else "up"
                events.append({**base, "event_type": "line_crossing", "details": f"Crossed horizontal line {direction}"})
            if not prev["inside"] and inside:
                events.append({**base, "event_type": "region_entry", "details": "Entered configured region"})
            elif prev["inside"] and not inside:
                events.append({**base, "event_type": "region_exit", "details": "Exited configured region"})
        previous[tid] = {"cy": cy, "inside": inside}
    return pd.DataFrame(events, columns=EVENT_COLUMNS).sort_values(["frame", "track_id"], kind="stable").reset_index(drop=True) if events else pd.DataFrame(columns=EVENT_COLUMNS)
