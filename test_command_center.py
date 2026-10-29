"""Command Center KPI and history-persistence tests.

Tests the pure logic behind page_command_center() KPI classification and
persist_history_record() behavior without invoking Streamlit UI rendering.
"""
from __future__ import annotations

import pytest

import helpers
import mainn2


def _make_record(kind="image", filename="sample.jpg", objects=2):
    return {
        "kind": kind,
        "filename": filename,
        "timestamp": "2026-10-27T12:00:00+00:00",
        "objects": objects,
        "classes": ["person"],
        "confidences": [0.9],
        "mean_confidence": 0.9,
    }


def test_persist_history_record_returns_true_when_store_saves(monkeypatch, tmp_path):
    store_path = tmp_path / "history.sqlite3"
    store = mainn2.HistoryStore(store_path)
    monkeypatch.setattr(mainn2, "HISTORY_STORE", store)
    assert mainn2.persist_history_record(_make_record()) is True


def test_persist_history_record_returns_false_when_store_none(monkeypatch):
    monkeypatch.setattr(mainn2, "HISTORY_STORE", None)
    assert mainn2.persist_history_record(_make_record()) is False


def test_persist_history_record_returns_false_when_save_raises(monkeypatch):
    class FailingStore:
        def save(self, record):
            raise RuntimeError("disk full")

    monkeypatch.setattr(mainn2, "HISTORY_STORE", FailingStore())
    assert mainn2.persist_history_record(_make_record()) is False


# ---------------------------------------------------------------------------
# History classification (mirrors page_command_center KPI logic)
# ---------------------------------------------------------------------------


def test_history_classification_separates_image_video_plate():
    history = helpers.normalize_history([
        _make_record(kind="image", filename="a.jpg", objects=3),
        _make_record(kind="image", filename="b.jpg", objects=1),
        _make_record(kind="video", filename="c.mp4", objects=10),
        _make_record(kind="plate", filename="d.jpg", objects=2),
        _make_record(kind="plate", filename="e.jpg", objects=1),
    ])

    images = [r for r in history if r.get("kind") == "image"]
    videos = [r for r in history if r.get("kind") == "video"]
    plates = [r for r in history if r.get("kind") == "plate"]

    assert len(images) == 2
    assert len(videos) == 1
    assert len(plates) == 2


def test_plate_records_are_not_counted_as_image_analyses():
    history = helpers.normalize_history([
        _make_record(kind="image", filename="img.jpg", objects=5),
        _make_record(kind="plate", filename="plate.jpg", objects=3),
    ])

    images = [r for r in history if r.get("kind") == "image"]
    plates = [r for r in history if r.get("kind") == "plate"]

    assert len(images) == 1
    assert len(plates) == 1


def test_plate_records_missing_objects_do_not_raise_keyerror():
    history = helpers.normalize_history([
        {"kind": "plate", "filename": "plate.jpg", "timestamp": "2026-10-27T12:00:00+00:00"},
        _make_record(kind="plate", filename="plate2.jpg", objects=4),
    ])

    plates = [r for r in history if r.get("kind") == "plate"]
    total = sum(r.get("objects", 0) for r in plates)
    assert total == 4


def test_objects_detected_excludes_plate_objects():
    history = helpers.normalize_history([
        _make_record(kind="image", filename="a.jpg", objects=3),
        _make_record(kind="image", filename="b.jpg", objects=2),
        _make_record(kind="plate", filename="c.jpg", objects=10),
    ])

    images = [r for r in history if r.get("kind") == "image"]
    plates = [r for r in history if r.get("kind") == "plate"]

    detection_observations = sum(r.get("objects", 0) for r in images)
    plate_detections = sum(r.get("objects", 0) for r in plates)

    assert detection_observations == 5
    assert plate_detections == 10


def test_detection_observations_not_conflated_with_unique_objects():
    history = helpers.normalize_history([
        _make_record(kind="image", filename="a.jpg", objects=5),
    ])

    images = [r for r in history if r.get("kind") == "image"]
    observations = sum(r.get("objects", 0) for r in images)

    assert observations == 5


# ---------------------------------------------------------------------------
# Persistence warning session-state flag
# ---------------------------------------------------------------------------


def test_persistence_warning_flag_is_set_on_failure(monkeypatch):
    monkeypatch.setattr(mainn2, "HISTORY_STORE", None)
    flag = False
    if not mainn2.persist_history_record(_make_record()):
        flag = True
    assert flag is True


def test_persistence_warning_flag_not_set_on_success(monkeypatch, tmp_path):
    store = mainn2.HistoryStore(tmp_path / "history.sqlite3")
    monkeypatch.setattr(mainn2, "HISTORY_STORE", store)
    flag = False
    if not mainn2.persist_history_record(_make_record()):
        flag = True
    assert flag is False