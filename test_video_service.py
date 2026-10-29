"""video_service.analyze_video tests: backward compatibility and tracking-option forwarding.

Run from the project root (``python -m pytest``) so ``video_service`` is importable.
"""
import ast
import inspect
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

import video_service as vs
from core.video.video_processor import VideoProcessor

BOX = (50, 60, 150, 160)


class _Box:
    def __init__(self):
        self.cls = np.array([0], dtype=np.float32)
        self.conf = np.array([0.9], dtype=np.float32)
        self.xyxy = np.array([BOX], dtype=np.float32)


class _Boxes:
    def __init__(self, track_id):
        self._box = _Box()
        self.id = None if track_id is None else _Ids([track_id])

    def __iter__(self):
        yield self._box


class _Ids(list):
    def tolist(self):
        return list(self)


class _Prediction:
    def __init__(self, frame, boxes):
        self.frame, self.boxes = frame, boxes

    def plot(self):
        return self.frame.copy()


class Model:
    """Tracking-capable fake that records how it was called."""

    names = {0: "person"}

    def __init__(self, track_id=1):
        self.track_id = track_id
        self.track_calls, self.predict_calls = [], []
        self.predictor = None

    def track(self, source, conf, persist, tracker, verbose=False, **extra):
        self.track_calls.append({"persist": persist, "tracker": tracker, "extra": extra})
        return [_Prediction(source, _Boxes(self.track_id))]

    def predict(self, source, conf, verbose=False, **extra):
        self.predict_calls.append({"extra": extra})
        return [_Prediction(source, _Boxes(None))]


class Engine:
    def __init__(self, model):
        self.model = model


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    if not writer.isOpened():
        pytest.skip("MP4 encoding is unavailable in this environment.")
    for _ in range(5):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def test_call_style_used_by_the_streamlit_page_still_works(tmp_path, video):
    """mainn2.py passes six positional arguments plus progress_callback."""
    progress = []
    result = vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.25, 1, None,
                              progress_callback=lambda read, total: progress.append(read))

    assert progress and progress[-1] == 5
    assert Path(result["run_dir"]).is_dir()
    assert Path(result["output_video_path"]).parent == Path(result["run_dir"]).resolve()
    assert result["frames_read"] == 5 and result["total_detections"] == 5
    assert result["tracking_enabled"] is True and result["unique_tracks"] == 1
    assert result["playback_path"] is None or Path(result["playback_path"]).is_file()
    assert result["throughput_fps"] is None or result["throughput_fps"] > 0
    for key in ("wall_seconds", "transcode_seconds", "playback_note"):
        assert key in result


def test_tracking_options_are_forwarded_to_the_processor(tmp_path, video):
    model = Model()
    vs.analyze_video(Engine(model), video, tmp_path / "runs", 0.3, 1, None, transcode=False,
                     tracker="custom.yaml", iou=0.4, min_track_observations=2)
    assert model.track_calls[0]["tracker"] == "custom.yaml"
    assert model.track_calls[0]["extra"] == {"iou": 0.4}


def test_tracking_can_be_switched_off_through_the_service(tmp_path, video):
    model = Model()
    result = vs.analyze_video(Engine(model), video, tmp_path / "runs", 0.3, 1, None,
                              transcode=False, enable_tracking=False)
    assert not model.track_calls and model.predict_calls
    assert result["tracking_enabled"] is False


def test_each_run_gets_its_own_folder_and_its_own_records(tmp_path, video):
    root = tmp_path / "runs"
    first = vs.analyze_video(Engine(Model(track_id=1)), video, root, 0.3, 1, None, transcode=False)
    second = vs.analyze_video(Engine(Model(track_id=2)), video, root, 0.3, 1, None, transcode=False)
    assert first["run_dir"] != second["run_dir"]
    assert {d["track_id"] for d in first["detections"]} == {1}
    assert {d["track_id"] for d in second["detections"]} == {2}


def test_frame_estimate_matches_what_the_processor_actually_does(tmp_path, video):
    meta = vs.probe_video(video)
    for stride, max_frames in [(1, None), (2, None), (2, 4), (3, 5)]:
        expected = vs.estimate_frames(meta, stride, max_frames)
        result = vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.3, stride,
                                  max_frames, transcode=False)
        assert expected == (result["frames_read"], result["frames_processed"])


def test_failed_run_leaves_no_run_folder_behind(tmp_path):
    root = tmp_path / "runs"
    with pytest.raises(FileNotFoundError):
        vs.analyze_video(Engine(Model()), tmp_path / "missing.mp4", root, 0.3, 1, None, transcode=False)
    assert not root.exists() or list(root.iterdir()) == []


def test_invalid_tracking_option_fails_cleanly(tmp_path, video):
    root = tmp_path / "runs"
    with pytest.raises(ValueError):
        vs.analyze_video(Engine(Model()), video, root, 0.3, 1, None, transcode=False, iou=2.0)
    assert not root.exists() or list(root.iterdir()) == []


def test_service_still_wraps_the_single_existing_processor():
    """Guard against accidentally introducing a second pipeline."""
    import inspect

    assert "VideoProcessor" in inspect.getsource(vs.analyze_video)
    assert VideoProcessor.process_video.__name__ == "process_video"


# =========================================================================== interface contract
FORWARDED_OPTIONS = {"enable_tracking", "tracker", "iou", "trail_length", "min_track_observations"}


def test_every_option_the_service_forwards_is_accepted_by_the_real_processor():
    """Regression guard: service and processor must never drift apart again."""
    service = inspect.signature(vs.analyze_video).parameters
    keyword_only = {n for n, p in service.items() if p.kind is inspect.Parameter.KEYWORD_ONLY}
    assert keyword_only == FORWARDED_OPTIONS
    processor = inspect.signature(VideoProcessor.process_video).parameters
    assert FORWARDED_OPTIONS <= set(processor)


def test_streamlit_page_only_calls_the_service_with_parameters_it_accepts():
    page = Path(vs.__file__).with_name("mainn2.py")
    if not page.is_file():
        pytest.skip("mainn2.py is not next to video_service.py in this layout.")
    tree = ast.parse(page.read_text(encoding="utf-8-sig"))
    params = inspect.signature(vs.analyze_video).parameters
    positional = [n for n, p in params.items()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "analyze_video"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "vs"]
    assert calls, "mainn2.py no longer calls vs.analyze_video"
    for call in calls:
        assert len(call.args) <= len(positional)
        assert {k.arg for k in call.keywords if k.arg} <= set(params)


def test_all_five_options_reach_process_video_with_the_values_given(tmp_path, video, monkeypatch):
    seen = {}
    original = VideoProcessor.process_video

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VideoProcessor, "process_video", spy)
    vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.3, 1, None, transcode=False,
                     enable_tracking=False, tracker="custom.yaml", iou=0.6,
                     trail_length=7, min_track_observations=4)
    assert {k: seen[k] for k in FORWARDED_OPTIONS} == {
        "enable_tracking": False, "tracker": "custom.yaml", "iou": 0.6,
        "trail_length": 7, "min_track_observations": 4}


def test_defaults_reach_process_video_unchanged(tmp_path, video, monkeypatch):
    seen = {}
    original = VideoProcessor.process_video
    monkeypatch.setattr(VideoProcessor, "process_video",
                        lambda self, *a, **k: (seen.update(k), original(self, *a, **k))[1])
    vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.3, 1, None, transcode=False)
    assert {k: seen[k] for k in FORWARDED_OPTIONS} == {
        "enable_tracking": True, "tracker": "bytetrack.yaml", "iou": None,
        "trail_length": 30, "min_track_observations": 3}


def test_min_track_observations_changes_the_result_end_to_end(tmp_path, video):
    default = vs.analyze_video(Engine(Model()), video, tmp_path / "a", 0.3, 1, None, transcode=False)
    strict = vs.analyze_video(Engine(Model()), video, tmp_path / "b", 0.3, 1, None, transcode=False,
                              min_track_observations=6)  # the clip only has 5 frames
    assert default["estimated_unique_objects"] == 1
    assert strict["estimated_unique_objects"] == 0


def test_invalid_trail_length_is_rejected_not_ignored(tmp_path, video):
    with pytest.raises(ValueError):
        vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.3, 1, None, transcode=False,
                         trail_length=0)


# =========================================================================== version skew
class OldProcessor:
    """Mimics the pre-Phase-2 VideoProcessor: no ``iou`` and no ``min_track_observations``."""

    seen: list = []

    def __init__(self, detection_engine, output_dir="outputs/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_video(self, video_path, confidence=0.25, frame_stride=1, max_frames=None,
                      progress_callback=None, enable_tracking=True, tracker="bytetrack.yaml",
                      trail_length=30):
        OldProcessor.seen.append({"enable_tracking": enable_tracking, "tracker": tracker,
                                  "trail_length": trail_length})
        video = self.output_dir / "x_annotated.mp4"
        csv = self.output_dir / "x_detections.csv"
        video.write_bytes(b"video")
        csv.write_text("frame\n")
        return {"output_video_path": str(video), "output_csv_path": str(csv),
                "processing_time_seconds": 1.0, "frames_read": 5}


@pytest.fixture
def old_processor(monkeypatch):
    OldProcessor.seen = []
    monkeypatch.setattr("core.video.video_processor.VideoProcessor", OldProcessor)
    return OldProcessor


def test_old_processor_still_works_with_default_options_and_logs_the_omission(
        tmp_path, video, old_processor, caplog):
    with caplog.at_level(logging.WARNING, logger="visionforge.video"):
        result = vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.3, 1, None,
                                  transcode=False, tracker="custom.yaml")
    assert old_processor.seen == [{"enable_tracking": True, "tracker": "custom.yaml", "trail_length": 30}]
    assert result["frames_read"] == 5
    assert "iou" in caplog.text and "min_track_observations" in caplog.text


@pytest.mark.parametrize("option", [{"iou": 0.5}, {"min_track_observations": 5}])
def test_old_processor_with_a_requested_new_option_raises_instead_of_ignoring_it(
        tmp_path, video, old_processor, option):
    root = tmp_path / "runs"
    name = next(iter(option))
    with pytest.raises(vs.ProcessorVersionError, match=name):
        vs.analyze_video(Engine(Model()), video, root, 0.3, 1, None, transcode=False, **option)
    assert old_processor.seen == []                       # the processor was never called
    assert not root.exists() or list(root.iterdir()) == []


def test_error_message_names_the_file_to_replace(tmp_path, video, old_processor):
    with pytest.raises(vs.ProcessorVersionError) as info:
        vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.3, 1, None,
                         transcode=False, iou=0.5)
    assert "core/video/video_processor.py" in str(info.value) and "restart" in str(info.value).lower()


def test_processor_accepting_arbitrary_keywords_receives_every_option(tmp_path, video, monkeypatch):
    class Flexible(OldProcessor):
        def process_video(self, video_path, **kwargs):
            Flexible.kwargs = kwargs
            return OldProcessor.process_video(self, video_path)

    monkeypatch.setattr("core.video.video_processor.VideoProcessor", Flexible)
    vs.analyze_video(Engine(Model()), video, tmp_path / "runs", 0.3, 1, None, transcode=False, iou=0.5)
    assert FORWARDED_OPTIONS <= set(Flexible.kwargs)
