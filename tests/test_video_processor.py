import cv2
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from core.video.video_processor import (
    VideoProcessor,
    VideoProcessingError,
)


class FakeBox:
    def __init__(self):
        self.cls = np.array([0], dtype=np.float32)
        self.conf = np.array([0.95], dtype=np.float32)
        self.xyxy = np.array([[50, 60, 150, 160]], dtype=np.float32)

    def __iter__(self):
        yield self


class FakePrediction:
    def __init__(self, frame):
        self.boxes = FakeBox()
        self.frame = frame

    def plot(self):
        return self.frame.copy()


class FakeModel:
    names = {0: "person"}

    def predict(self, source, conf, verbose=False):
        return [FakePrediction(source)]


class FakeDetectionEngine:
    def __init__(self):
        self.model = FakeModel()


@pytest.fixture
def sample_video(tmp_path):
    video_path = tmp_path / "sample.mp4"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (320, 240),
    )

    if not writer.isOpened():
        pytest.skip("MP4 encoding is unavailable in this environment.")

    for index in range(5):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {index}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    writer.release()

    assert video_path.exists()
    assert video_path.stat().st_size > 0

    return video_path


def test_process_video_creates_valid_outputs(tmp_path, sample_video):
    processor = VideoProcessor(
        FakeDetectionEngine(),
        output_dir=tmp_path / "outputs",
    )

    result = processor.process_video(sample_video)

    assert result["frames_read"] == 5
    assert result["frames_processed"] == 5
    assert result["total_detections"] == 5
    assert result["unique_classes"] == ["person"]

    output_video = Path(result["output_video_path"])
    output_csv = Path(result["output_csv_path"])

    assert output_video.exists()
    assert output_video.stat().st_size > 0

    assert output_csv.exists()
    assert output_csv.stat().st_size > 0

    capture = cv2.VideoCapture(str(output_video))

    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5

        success, frame = capture.read()

        assert success
        assert frame is not None
    finally:
        capture.release()

    detections = pd.read_csv(output_csv)

    assert len(detections) == 5
    assert set(detections["label"]) == {"person"}
    assert detections["confidence"].between(0, 1).all()


def test_missing_video_raises_error(tmp_path):
    processor = VideoProcessor(
        FakeDetectionEngine(),
        output_dir=tmp_path / "outputs",
    )

    with pytest.raises(FileNotFoundError):
        processor.process_video(tmp_path / "missing.mp4")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"frame_stride": 0},
        {"frame_stride": -1},
        {"max_frames": 0},
        {"max_frames": -1},
    ],
)
def test_invalid_parameters_are_rejected(
    tmp_path,
    sample_video,
    kwargs,
):
    processor = VideoProcessor(
        FakeDetectionEngine(),
        output_dir=tmp_path / "outputs",
    )

    with pytest.raises(ValueError):
        processor.process_video(sample_video, **kwargs)


def test_frame_stride_processes_only_selected_frames(tmp_path, sample_video):
    processor = VideoProcessor(
        FakeDetectionEngine(),
        output_dir=tmp_path / "stride_outputs",
    )

    result = processor.process_video(
        sample_video,
        frame_stride=2,
    )

    # Five source frames: process frames 0, 2, and 4.
    assert result["frames_read"] == 5
    assert result["frames_processed"] == 3
    assert result["total_detections"] == 3


def test_max_frames_limits_source_frames(tmp_path, sample_video):
    processor = VideoProcessor(
        FakeDetectionEngine(),
        output_dir=tmp_path / "limited_outputs",
    )

    result = processor.process_video(
        sample_video,
        max_frames=3,
    )

    # Only the first three source frames should be read.
    assert result["frames_read"] == 3
    assert result["frames_processed"] == 3
    assert result["total_detections"] == 3


def test_max_frames_and_frame_stride_work_together(
    tmp_path, sample_video
):
    processor = VideoProcessor(
        FakeDetectionEngine(),
        output_dir=tmp_path / "combined_outputs",
    )

    result = processor.process_video(
        sample_video,
        frame_stride=2,
        max_frames=4,
    )

    # Read frames 0, 1, 2, and 3; process frames 0 and 2.
    assert result["frames_read"] == 4
    assert result["frames_processed"] == 2
    assert result["total_detections"] == 2