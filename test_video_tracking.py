"""VideoProcessor tracking-path tests using a scripted fake model (no YOLO weights needed)."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

import core.video.video_processor as vp_module
from core.video.video_processor import CSV_COLUMNS, VideoProcessingError, VideoProcessor

PERSON = (50, 60, 150, 160)
PERSON_2 = (200, 60, 280, 170)
CAR = (30, 150, 130, 220)


# --------------------------------------------------------------------------- fakes
class IdArray(list):
    """Stands in for the tensor Ultralytics puts in ``boxes.id`` (only .tolist() is used)."""

    def tolist(self):
        return list(self)


class ScriptedBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = np.array([cls], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class ScriptedBoxes:
    """rows: (class_id, confidence, xyxy, track_id or None). ``id`` is None when no row has an ID."""

    def __init__(self, rows):
        self._boxes = [ScriptedBox(c, s, xy) for c, s, xy, _ in rows]
        ids = [row[3] for row in rows]
        self.id = None if all(i is None for i in ids) else IdArray(ids)

    def __iter__(self):
        return iter(self._boxes)


class ScriptedPrediction:
    def __init__(self, frame, boxes):
        self.frame, self.boxes = frame, boxes

    def plot(self):
        return self.frame.copy()


class PredictOnlyModel:
    """A model without ``track`` (like the original test model)."""

    names = {0: "person", 2: "car"}

    def __init__(self, script):
        self.script, self.calls = script, 0
        self.track_calls, self.predict_calls = [], []
        self.predictor = None

    def _next(self, source, none_boxes=False, strip_ids=False):
        index, self.calls = self.calls, self.calls + 1
        rows = [] if none_boxes or index >= len(self.script) else self.script[index]
        if strip_ids:  # plain predict() never returns tracker IDs
            rows = [(c, s, xy, None) for c, s, xy, _ in rows]
        return [ScriptedPrediction(source, None if none_boxes else ScriptedBoxes(rows))]

    def predict(self, source, conf, verbose=False, **extra):
        self.predict_calls.append({"conf": conf, "extra": extra})
        return self._next(source, none_boxes=self.script == "none", strip_ids=True)


class TrackingModel(PredictOnlyModel):
    def track(self, source, conf, persist, tracker, verbose=False, **extra):
        self.track_calls.append({"conf": conf, "persist": persist, "tracker": tracker, "extra": extra})
        return self._next(source, none_boxes=self.script == "none")


class Engine:
    def __init__(self, model):
        self.model = model


def make_video(path, frames=5, fps=10, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("MP4 encoding is unavailable in this environment.")
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "clip.mp4", frames=5)


def process(tmp_path, video, model, **kwargs):
    processor = VideoProcessor(Engine(model), output_dir=tmp_path / "out")
    return processor.process_video(video, **kwargs)


def codes(result):
    return {w["code"] for w in result["tracking_diagnostics"]["warnings"]}


# --------------------------------------------------------------------------- tracking path
def test_tracking_is_used_with_persist_and_configured_tracker(tmp_path, video):
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 5)
    result = process(tmp_path, video, model)

    assert len(model.track_calls) == 5 and not model.predict_calls
    assert all(c["persist"] is True and c["tracker"] == "bytetrack.yaml" and c["conf"] == 0.25
               for c in model.track_calls)
    assert result["tracking_enabled"] is True and result["tracker"] == "bytetrack.yaml"


def test_detection_observations_are_not_unique_tracks(tmp_path, video):
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 5)
    result = process(tmp_path, video, model)

    assert result["total_detections"] == 5            # observations
    assert result["unique_tracks"] == 1               # distinct tracker IDs
    assert result["estimated_unique_objects"] == 1
    assert result["estimated_unique_objects_by_class"] == {"person": 1}
    assert result["unique_tracks_by_class"] == {"person": 1}
    assert result["tracked_observations"] == 5 and result["untracked_observations"] == 0
    assert result["track_summaries"][0]["observed_frames"] == 5
    csv = pd.read_csv(result["output_csv_path"])
    assert list(csv.columns) == CSV_COLUMNS and set(csv["track_id"]) == {1}


def test_class_wise_counts_observations_and_tracks(tmp_path):
    video = make_video(tmp_path / "four.mp4", frames=4)
    frame_rows = [(0, 0.9, PERSON, 1), (0, 0.8, PERSON_2, 2), (2, 0.7, CAR, 3)]
    result = process(tmp_path, video, TrackingModel([frame_rows] * 4))

    assert result["detection_counts_by_class"] == {"person": 8, "car": 4}
    assert result["unique_tracks_by_class"] == {"car": 1, "person": 2}
    assert result["estimated_unique_objects_by_class"] == {"car": 1, "person": 2}
    assert result["max_detections_in_frame"] == 3
    assert result["unique_classes"] == ["car", "person"]


def test_track_ids_missing_on_every_box(tmp_path, video):
    result = process(tmp_path, video, TrackingModel([[(0, 0.9, PERSON, None)]] * 5))

    assert result["total_detections"] == 5
    assert result["unique_tracks"] == 0 and result["track_summaries"] == []
    assert result["tracked_observations"] == 0 and result["untracked_observations"] == 5
    assert result["estimated_unique_objects"] == 0
    assert "tracking_no_ids" in codes(result)
    assert pd.read_csv(result["output_csv_path"])["track_id"].isna().all()


def test_ids_that_appear_midway_are_counted_only_when_present(tmp_path, video):
    script = [[(0, 0.9, PERSON, None)]] * 2 + [[(0, 0.9, PERSON, 4)]] * 3
    result = process(tmp_path, video, TrackingModel(script))

    assert result["tracked_observations"] == 3 and result["untracked_observations"] == 2
    assert result["unique_tracks"] == 1 and result["estimated_unique_objects"] == 1
    assert "untracked_observations" in codes(result)


def test_empty_detections_produce_valid_empty_outputs(tmp_path, video):
    result = process(tmp_path, video, TrackingModel([[]] * 5))

    assert result["frames_processed"] == 5 and result["total_detections"] == 0
    assert result["detections"] == [] and result["unique_classes"] == []
    assert result["detection_counts_by_class"] == {} and result["max_detections_in_frame"] == 0
    assert result["unique_tracks"] == 0 and result["estimated_unique_objects"] == 0
    assert result["tracking_diagnostics"]["warnings"] == []
    csv = pd.read_csv(result["output_csv_path"])
    assert csv.empty and list(csv.columns) == CSV_COLUMNS
    assert Path(result["output_video_path"]).stat().st_size > 0


def test_prediction_without_boxes_object_is_treated_as_no_detections(tmp_path, video):
    result = process(tmp_path, video, TrackingModel("none"))
    assert result["total_detections"] == 0 and result["frames_processed"] == 5


# --------------------------------------------------------------------------- stride / max frames
def test_stride_with_tracking_keeps_source_timestamps(tmp_path):
    video = make_video(tmp_path / "six.mp4", frames=6)
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 6)
    result = process(tmp_path, video, model, frame_stride=2)

    assert len(model.track_calls) == 3
    assert [d["frame"] for d in result["detections"]] == [0, 2, 4]
    assert [d["timestamp_seconds"] for d in result["detections"]] == [0.0, 0.2, 0.4]
    assert result["frame_stride"] == 2 and "stride_tracking" in codes(result)
    assert result["tracking_diagnostics"]["tracks_with_gaps"] == 0  # a 2-frame step is expected


def test_max_frames_limits_tracking_calls(tmp_path):
    video = make_video(tmp_path / "six.mp4", frames=6)
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 6)
    result = process(tmp_path, video, model, max_frames=4)
    assert result["frames_read"] == 4 and len(model.track_calls) == 4


# --------------------------------------------------------------------------- tracker lifecycle / options
def test_existing_trackers_are_reset_before_a_new_run(tmp_path, video):
    resets = []
    spy = lambda name: SimpleNamespace(reset=lambda: resets.append(name))  # noqa: E731
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 5)
    model.predictor = SimpleNamespace(trackers=[spy("a"), spy("b")])
    process(tmp_path, video, model)
    assert resets == ["a", "b"]


def test_second_run_does_not_inherit_the_first_runs_records(tmp_path, video):
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 5 + [[(0, 0.9, PERSON, 2)]] * 5)
    processor = VideoProcessor(Engine(model), output_dir=tmp_path / "out")
    first = processor.process_video(video)
    second = processor.process_video(video)
    assert {d["track_id"] for d in first["detections"]} == {1}
    assert {d["track_id"] for d in second["detections"]} == {2}
    assert second["total_detections"] == 5


def test_enable_tracking_false_uses_predict_even_if_track_exists(tmp_path, video):
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 5)
    result = process(tmp_path, video, model, enable_tracking=False)
    assert not model.track_calls and len(model.predict_calls) == 5
    assert result["tracking_enabled"] is False and result["tracker"] is None
    assert result["unique_tracks"] == 0 and result["estimated_unique_objects"] == 0


def test_model_without_track_falls_back_to_predict(tmp_path, video):
    model = PredictOnlyModel([[(0, 0.9, PERSON, None)]] * 5)
    result = process(tmp_path, video, model)
    assert result["tracking_enabled"] is False and len(model.predict_calls) == 5
    assert "tracking_no_ids" not in codes(result)


def test_custom_tracker_and_iou_are_forwarded_only_when_set(tmp_path, video):
    model = TrackingModel([[]] * 5)
    process(tmp_path, video, model)
    assert model.track_calls[0]["extra"] == {}

    model = TrackingModel([[]] * 5)
    result = process(tmp_path, video, model, tracker="custom.yaml", iou=0.5)
    assert model.track_calls[0]["tracker"] == "custom.yaml"
    assert model.track_calls[0]["extra"] == {"iou": 0.5}
    assert result["iou"] == 0.5


@pytest.mark.parametrize("kwargs", [{"iou": 0.0}, {"iou": 1.5}, {"iou": -0.1},
                                    {"min_track_observations": 0}])
def test_invalid_new_parameters_are_rejected(tmp_path, video, kwargs):
    with pytest.raises(ValueError):
        process(tmp_path, video, TrackingModel([]), **kwargs)


def test_min_track_observations_changes_the_unique_object_estimate(tmp_path):
    video = make_video(tmp_path / "two.mp4", frames=2)
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 2)
    assert process(tmp_path, video, model)["estimated_unique_objects"] == 0
    model = TrackingModel([[(0, 0.9, PERSON, 1)]] * 2)
    assert process(tmp_path, video, model, min_track_observations=2)["estimated_unique_objects"] == 1


# --------------------------------------------------------------------------- class names
def test_unknown_class_id_gets_a_safe_label_instead_of_crashing(tmp_path, video):
    result = process(tmp_path, video, TrackingModel([[(9, 0.9, PERSON, 1)]] * 5))
    assert result["unique_classes"] == ["class_9"]


def test_list_style_class_names_are_supported(tmp_path, video):
    model = TrackingModel([[(1, 0.9, PERSON, 1)]] * 5)
    model.names = ["person", "bicycle"]
    assert process(tmp_path, video, model)["unique_classes"] == ["bicycle"]


# --------------------------------------------------------------------------- failure handling
def test_invalid_video_file_raises_and_leaves_no_outputs(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(VideoProcessingError):
        process(tmp_path, bad, TrackingModel([]))
    assert list((tmp_path / "out").iterdir()) == []


def test_zero_byte_video_raises(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(VideoProcessingError):
        process(tmp_path, empty, TrackingModel([]))


def test_model_returning_no_result_cleans_up_partial_outputs(tmp_path, video):
    model = TrackingModel([])
    model.track = lambda **kwargs: []
    with pytest.raises(VideoProcessingError):
        process(tmp_path, video, model)
    assert list((tmp_path / "out").iterdir()) == []


def test_diagnostics_failure_does_not_discard_a_finished_analysis(tmp_path, video, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("diagnostics exploded")

    monkeypatch.setattr(vp_module, "compute_tracking_diagnostics", boom)
    result = process(tmp_path, video, TrackingModel([[(0, 0.9, PERSON, 1)]] * 5))
    assert result["total_detections"] == 5 and result["unique_tracks"] == 1
    assert result["tracking_diagnostics"] is None and result["estimated_unique_objects"] is None


# --------------------------------------------------------------------------- compatibility
def test_original_result_keys_are_all_still_present(tmp_path, video):
    result = process(tmp_path, video, TrackingModel([[(0, 0.9, PERSON, 1)]] * 5))
    original = {
        "source_path", "output_video_path", "output_csv_path", "video_fps", "video_width",
        "video_height", "reported_frame_count", "frames_read", "frames_processed",
        "total_detections", "detection_counts_by_class", "max_detections_in_frame",
        "unique_classes", "tracking_enabled", "unique_tracks", "unique_tracks_by_class",
        "track_summaries", "processing_time_seconds", "detections",
    }
    assert original <= set(result)
    assert set(result["detections"][0]) == set(CSV_COLUMNS)
