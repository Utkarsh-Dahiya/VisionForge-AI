from pathlib import Path

import cv2
import numpy as np
import pytest

from core.detection.detection_engine import DetectionEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE = PROJECT_ROOT / "assets" / "test_image.jpg"


@pytest.fixture
def engine():
    """Create a detection engine for tests that require YOLO."""
    return DetectionEngine(
        model_path=str(PROJECT_ROOT / "yolo11n.pt")
    )


def test_missing_image_raises_file_not_found():
    """A nonexistent image should raise FileNotFoundError."""
    engine = DetectionEngine.__new__(DetectionEngine)

    missing_path = PROJECT_ROOT / "does_not_exist.jpg"

    with pytest.raises(FileNotFoundError):
        engine.detect_image(str(missing_path))


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_invalid_confidence_raises_value_error(confidence):
    """Confidence values outside 0–1 should be rejected."""
    engine = DetectionEngine.__new__(DetectionEngine)

    with pytest.raises(ValueError, match="Confidence"):
        engine.detect_image(
            image_path=str(TEST_IMAGE),
            confidence=confidence,
        )


def test_corrupted_image_raises_value_error(tmp_path):
    """A file that is not a valid image should be rejected."""
    corrupted_image = tmp_path / "corrupted.jpg"
    corrupted_image.write_text(
        "This is not a valid image",
        encoding="utf-8",
    )

    engine = DetectionEngine.__new__(DetectionEngine)

    with pytest.raises(ValueError, match="Could not read image"):
        engine.detect_image(str(corrupted_image))


def test_valid_image_returns_expected_result_structure(engine):
    """A real image should produce the expected result fields."""
    result = engine.detect_image(
        image_path=str(TEST_IMAGE),
        confidence=0.25,
    )

    assert result["image_width"] > 0
    assert result["image_height"] > 0
    assert result["object_count"] == len(result["detections"])
    assert result["inference_time_ms"] >= 0
    assert isinstance(result["annotated_image"], np.ndarray)

    assert result["annotated_image"].shape[:2] == (
        result["image_height"],
        result["image_width"],
    )

    assert isinstance(result["detections"], list)


def test_detection_records_have_valid_fields(engine):
    """Every returned detection should contain valid structured data."""
    result = engine.detect_image(
        image_path=str(TEST_IMAGE),
        confidence=0.25,
    )

    for detection in result["detections"]:
        assert isinstance(detection["class_id"], int)
        assert isinstance(detection["label"], str)
        assert detection["label"]
        assert 0 <= detection["confidence"] <= 1

        bbox = detection["bbox"]

        assert set(bbox) == {"x1", "y1", "x2", "y2"}

        assert 0 <= bbox["x1"] < bbox["x2"] <= result["image_width"]
        assert 0 <= bbox["y1"] < bbox["y2"] <= result["image_height"]


def test_test_image_exists():
    """The project's sample image should be available."""
    assert TEST_IMAGE.is_file()


# ---------------------------------------------------------------------------
# Unknown class ID regression tests (no real YOLO weights needed)
# ---------------------------------------------------------------------------

class _FakeBox:
    """Mimics an Ultralytics detection box (cls/conf/xyxy are numpy arrays)."""

    def __init__(self, class_id, confidence, xyxy):
        self.cls = np.array([class_id], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class _FakeBoxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)


class _FakePrediction:
    def __init__(self, boxes, annotated_image):
        self.boxes = _FakeBoxes(boxes)
        self._annotated_image = annotated_image

    def plot(self):
        return self._annotated_image


class _FakeModel:
    """Stand-in for a YOLO model exposing ``names`` and ``predict``."""

    def __init__(self, names, boxes):
        self.names = names
        self._boxes = boxes
        self.predict_call_count = 0

    def predict(self, source, conf, verbose=False):
        self.predict_call_count += 1
        return [
            _FakePrediction(
                self._boxes,
                np.zeros((64, 64, 3), dtype=np.uint8),
            )
        ]


def _engine_with_model(model):
    """Build a DetectionEngine without loading real YOLO weights."""
    engine = DetectionEngine.__new__(DetectionEngine)
    engine.model = model
    return engine


def _write_image(path):
    cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))


def test_unknown_class_id_dict_names_uses_safe_fallback(tmp_path):
    """Unknown class id with dict names must not raise KeyError; label is class_<id>."""
    image = tmp_path / "image.jpg"
    _write_image(image)

    model = _FakeModel(
        names={0: "person", 1: "car"},
        boxes=[_FakeBox(class_id=99, confidence=0.9, xyxy=(10, 10, 50, 50))],
    )
    engine = _engine_with_model(model)
    result = engine.detect_image(image_path=str(image), confidence=0.25)

    assert len(result["detections"]) == 1
    assert result["detections"][0]["class_id"] == 99
    assert result["detections"][0]["label"] == "class_99"


def test_unknown_class_id_list_names_uses_safe_fallback(tmp_path):
    """Unknown class id with list names (out of range) must yield class_<id>."""
    image = tmp_path / "image.jpg"
    _write_image(image)

    model = _FakeModel(
        names=["person", "car"],
        boxes=[_FakeBox(class_id=7, confidence=0.8, xyxy=(5, 5, 30, 40))],
    )
    engine = _engine_with_model(model)
    result = engine.detect_image(image_path=str(image), confidence=0.25)

    assert len(result["detections"]) == 1
    assert result["detections"][0]["class_id"] == 7
    assert result["detections"][0]["label"] == "class_7"


def test_unknown_class_id_tuple_names_uses_safe_fallback(tmp_path):
    """Unknown class id with tuple names (out of range) must yield class_<id>."""
    image = tmp_path / "image.jpg"
    _write_image(image)

    model = _FakeModel(
        names=("person", "car"),
        boxes=[_FakeBox(class_id=3, confidence=0.7, xyxy=(0, 0, 20, 20))],
    )
    engine = _engine_with_model(model)
    result = engine.detect_image(image_path=str(image), confidence=0.25)

    assert len(result["detections"]) == 1
    assert result["detections"][0]["class_id"] == 3
    assert result["detections"][0]["label"] == "class_3"


def test_known_class_id_still_resolves_to_real_name(tmp_path):
    """Known class ids must still resolve to their configured names (no regression)."""
    image = tmp_path / "image.jpg"
    _write_image(image)

    model = _FakeModel(
        names={0: "person", 2: "car"},
        boxes=[
            _FakeBox(class_id=0, confidence=0.9, xyxy=(0, 0, 10, 10)),
            _FakeBox(class_id=2, confidence=0.8, xyxy=(5, 5, 15, 15)),
        ],
    )
    engine = _engine_with_model(model)
    result = engine.detect_image(image_path=str(image), confidence=0.25)

    labels = {d["label"] for d in result["detections"]}
    assert labels == {"person", "car"}
    assert model.predict_call_count == 1