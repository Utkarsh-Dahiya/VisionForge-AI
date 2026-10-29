"""
VisionForge AI - Video Intelligence Engine.

Processes videos with YOLO detection/tracking, annotated output, and CSV reports.
"""

import logging
from collections import defaultdict, deque
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import pandas as pd

from core.detection.detection_engine import DetectionEngine
from core.video.tracking_diagnostics import DiagnosticsConfig, compute_tracking_diagnostics

log = logging.getLogger("visionforge.video")

CSV_COLUMNS = [
    "frame", "timestamp_seconds", "track_id", "class_id", "label",
    "confidence", "x1", "y1", "x2", "y2",
]


class VideoProcessingError(RuntimeError):
    """Raised when video processing cannot be completed."""


def _class_label(names: Any, class_id: int) -> str:
    """Model-provided class name, with a safe fallback for unexpected class IDs."""
    try:
        return str(names[class_id])
    except (KeyError, IndexError, TypeError):
        return f"class_{class_id}"


class VideoProcessor:
    """Process videos using VisionForge's detection engine and optional tracking."""

    def __init__(
        self,
        detection_engine: DetectionEngine,
        output_dir: str | Path = "outputs/videos",
    ) -> None:
        self.detection_engine = detection_engine
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_video(
        self,
        video_path: str | Path,
        confidence: float = 0.25,
        frame_stride: int = 1,
        max_frames: int | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
        enable_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
        trail_length: int = 30,
        iou: float | None = None,
        min_track_observations: int = 3,
    ) -> dict[str, Any]:
        """Create an annotated MP4 and CSV detection/tracking report.

        Tracking is enabled by default when the model exposes ``track``.
        Lightweight/test models that only implement ``predict`` remain supported.
        IDs are tracker-local and are not guaranteed to identify an object forever.

        ``iou`` (optional) is forwarded to the model as its NMS IoU threshold; when
        None the model default is used. ``tracker`` may be a built-in name such as
        ``"bytetrack.yaml"`` or a path to a custom tracker YAML. ``min_track_observations``
        is the minimum number of processed frames a track needs before it counts
        toward ``estimated_unique_objects``.

        Count semantics (see core.video.tracking_diagnostics.METRIC_DEFINITIONS):
        ``total_detections`` = detection observations (boxes summed over processed
        frames); ``unique_tracks`` = distinct tracker IDs; ``estimated_unique_objects``
        = tracks seen in at least ``min_track_observations`` frames.
        """
        source_path = Path(video_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Video file not found: {source_path}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0.")
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1.")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1 when specified.")
        if trail_length < 1:
            raise ValueError("trail_length must be at least 1.")
        if not isinstance(tracker, str) or not tracker.strip():
            raise ValueError("tracker must be a non-empty tracker configuration.")
        if iou is not None and not 0.0 < iou <= 1.0:
            raise ValueError("iou must be in (0.0, 1.0] when specified.")
        if min_track_observations < 1:
            raise ValueError("min_track_observations must be at least 1.")

        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            capture.release()
            raise VideoProcessingError(f"Could not open video: {source_path}")

        output_path = self.output_dir / f"{source_path.stem}_annotated.mp4"
        csv_path = self.output_dir / f"{source_path.stem}_detections.csv"
        writer = None
        frames_read = frames_processed = 0
        records: list[dict[str, Any]] = []
        # Keep recent center points for visual trails; IDs are scoped to this run.
        trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))
        started_at = perf_counter()

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise VideoProcessingError("The video has invalid frame dimensions.")
            fps_assumed = not fps > 0  # also true for NaN
            if fps_assumed:
                fps = 25.0  # timestamps become approximate; reported in the result and diagnostics

            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise VideoProcessingError(
                    "Could not initialize the MP4 video writer. Check your OpenCV codec support."
                )

            model = self.detection_engine.model
            names = model.names  # class names always come from the model itself
            extra_args: dict[str, Any] = {"iou": iou} if iou is not None else {}
            tracking_used = bool(enable_tracking and callable(getattr(model, "track", None)))
            # If this model has already tracked another video, reset existing
            # trackers so IDs/trajectories do not leak across separate runs.
            if tracking_used:
                predictor = getattr(model, "predictor", None)
                existing_trackers = getattr(predictor, "trackers", None)
                if existing_trackers:
                    for existing_tracker in existing_trackers:
                        reset = getattr(existing_tracker, "reset", None)
                        if callable(reset):
                            reset()

            while max_frames is None or frames_read < max_frames:
                success, frame = capture.read()
                if not success:
                    break

                frame_number = frames_read
                frames_read += 1
                if progress_callback is not None:
                    total_for_progress = (
                        max_frames if max_frames is not None
                        else reported_frame_count if reported_frame_count > 0 else None
                    )
                    if total_for_progress is not None and max_frames is not None and reported_frame_count > 0:
                        total_for_progress = min(total_for_progress, reported_frame_count)
                    progress_callback(frames_read, total_for_progress)

                if frame_number % frame_stride != 0:
                    writer.write(frame)
                    continue

                if tracking_used:
                    predictions = model.track(
                        source=frame,
                        conf=confidence,
                        persist=True,
                        tracker=tracker,
                        verbose=False,
                        **extra_args,
                    )
                else:
                    predictions = model.predict(
                        source=frame, conf=confidence, verbose=False, **extra_args
                    )

                if not predictions:
                    raise VideoProcessingError("The detection model returned no prediction result.")
                prediction = predictions[0]
                annotated_frame = prediction.plot()
                if annotated_frame is None or annotated_frame.shape[:2] != (height, width):
                    raise VideoProcessingError("The detection model returned an invalid annotated frame.")

                boxes = prediction.boxes
                if boxes is not None:
                    ids = getattr(boxes, "id", None)
                    id_values = ids.tolist() if ids is not None else None
                    for index, box in enumerate(boxes):
                        track_id = None
                        if id_values is not None and index < len(id_values):
                            raw_id = id_values[index]
                            if raw_id is not None:
                                track_id = int(raw_id)

                        class_id = int(box.cls.item())
                        label = _class_label(names, class_id)
                        score = float(box.conf.item())
                        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())

                        if track_id is not None:
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            trails[track_id].append(center)
                            points = list(trails[track_id])
                            for point_index in range(1, len(points)):
                                cv2.line(
                                    annotated_frame, points[point_index - 1], points[point_index],
                                    (0, 255, 255), 2,
                                )
                            cv2.putText(
                                annotated_frame, f"ID {track_id}",
                                (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2, cv2.LINE_AA,
                            )

                        records.append({
                            "frame": frame_number,
                            "timestamp_seconds": round(frame_number / fps, 3),
                            "track_id": track_id,
                            "class_id": class_id,
                            "label": label,
                            "confidence": round(score, 4),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        })

                writer.write(annotated_frame)
                frames_processed += 1

            if frames_read == 0:
                raise VideoProcessingError("No readable frames were found in the video.")
            if frames_processed == 0:
                raise VideoProcessingError("No video frames were processed.")

            writer.release()
            writer = None
            dataframe = pd.DataFrame(records, columns=CSV_COLUMNS)
            dataframe.to_csv(csv_path, index=False)

            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise VideoProcessingError("The output video was not generated correctly.")
            verification = cv2.VideoCapture(str(output_path))
            try:
                if not verification.isOpened():
                    raise VideoProcessingError("The generated output video cannot be reopened.")
                success, first_frame = verification.read()
                if not success or first_frame is None:
                    raise VideoProcessingError("The generated output video contains no readable frames.")
            finally:
                verification.release()

            summaries = []
            tracked = dataframe[dataframe["track_id"].notna()] if not dataframe.empty else dataframe
            if not tracked.empty:
                for track_id, group in tracked.groupby("track_id", sort=True):
                    summaries.append({
                        "track_id": int(track_id),
                        "label": str(group["label"].mode().iloc[0]),
                        "first_frame": int(group["frame"].min()),
                        "last_frame": int(group["frame"].max()),
                        "observed_frames": int(group["frame"].nunique()),
                        "duration_seconds": round(
                            (int(group["frame"].max()) - int(group["frame"].min())) / fps, 3
                        ),
                    })

            # Detection observations count every box in every processed frame.
            # Unique-track counts are estimates and depend on tracker ID stability.
            detection_counts_by_class = (
                dataframe.groupby("label").size().sort_values(ascending=False).to_dict()
                if not dataframe.empty else {}
            )
            per_frame_counts = (
                dataframe.groupby("frame").size() if not dataframe.empty
                else pd.Series(dtype="int64")
            )
            max_detections_in_frame = int(per_frame_counts.max()) if not per_frame_counts.empty else 0

            unique_tracks_by_class: dict[str, int] = {}
            if not tracked.empty:
                track_labels = tracked.groupby("track_id")["label"].agg(
                    lambda labels: str(labels.mode().iloc[0]) if not labels.mode().empty else str(labels.iloc[0])
                )
                unique_tracks_by_class = {
                    str(label): int(count)
                    for label, count in track_labels.value_counts().sort_index().items()
                }

            tracked_observations = int(len(tracked))
            diagnostics = None
            try:
                diagnostics = compute_tracking_diagnostics(
                    dataframe,
                    fps=fps, frame_width=width, frame_height=height,
                    frame_stride=frame_stride, tracking_enabled=tracking_used,
                    fps_assumed=fps_assumed,
                    config=DiagnosticsConfig(min_track_observations=min_track_observations),
                )
            except Exception:  # advisory data: never throw away a finished analysis
                log.exception("Tracking diagnostics failed; continuing without them.")

            elapsed_seconds = perf_counter() - started_at
            return {
                "source_path": str(source_path),
                "output_video_path": str(output_path),
                "output_csv_path": str(csv_path),
                "video_fps": fps,
                "video_width": width,
                "video_height": height,
                "reported_frame_count": reported_frame_count,
                "frames_read": frames_read,
                "frames_processed": frames_processed,
                "total_detections": len(records),
                "detection_counts_by_class": {str(k): int(v) for k, v in detection_counts_by_class.items()},
                "max_detections_in_frame": max_detections_in_frame,
                "unique_classes": sorted(dataframe["label"].dropna().unique().tolist()),
                "tracking_enabled": tracking_used,
                "tracker": tracker if tracking_used else None,
                "iou": iou,
                "frame_stride": frame_stride,
                "fps_assumed": fps_assumed,
                "tracked_observations": tracked_observations,
                "untracked_observations": len(records) - tracked_observations,
                "estimated_unique_objects": (
                    diagnostics["estimated_unique_objects"] if diagnostics else None),
                "estimated_unique_objects_by_class": (
                    diagnostics["estimated_unique_objects_by_class"] if diagnostics else None),
                "tracking_diagnostics": diagnostics,
                "unique_tracks": len(summaries),
                "unique_tracks_by_class": unique_tracks_by_class,
                "track_summaries": summaries,
                "processing_time_seconds": round(elapsed_seconds, 3),
                "detections": records,
            }
        except Exception:
            if writer is not None:
                writer.release()
            output_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
