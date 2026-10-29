import json
from pathlib import Path

import pytest

from core.storage.history_store import HistoryStore


def sample_record(kind="image", filename="sample.jpg", timestamp="2026-10-18T12:00:00+05:30"):
    return {
        "kind": kind,
        "filename": filename,
        "timestamp": timestamp,
        "objects": 2,
        "classes": ["person", "car"],
        "confidences": [0.91, 0.83],
        "mean_confidence": 0.87,
    }


def test_save_and_reload_persists_across_store_instances(tmp_path):
    path = tmp_path / "history.sqlite3"
    first = HistoryStore(path)
    first.save(sample_record())
    second = HistoryStore(path)
    assert second.load_all() == [sample_record()]
    assert second.count() == 1


def test_records_return_oldest_to_newest_and_limit(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.save(sample_record(filename="one.jpg", timestamp="2026-01-01T00:00:00+00:00"))
    store.save(sample_record(kind="video", filename="two.mp4", timestamp="2026-01-02T00:00:00+00:00"))
    assert [r["filename"] for r in store.load_all()] == ["one.jpg", "two.mp4"]
    assert [r["filename"] for r in store.load_all(limit=1)] == ["two.mp4"]
    assert store.count("video") == 1


def test_save_many_is_supported(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    assert store.save_many([sample_record(), sample_record(kind="plate")]) == 2
    assert store.count() == 2


def test_rejects_non_dict_record(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    with pytest.raises(TypeError):
        store.save(["not", "a", "record"])


def test_rejects_missing_timestamp(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    with pytest.raises(ValueError, match="timestamp"):
        store.save({"kind": "image", "filename": "x.jpg"})


def test_rejects_non_json_values(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    record = sample_record()
    record["bad"] = object()
    with pytest.raises(ValueError, match="JSON-serializable"):
        store.save(record)


def test_clear_removes_history(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.save(sample_record())
    store.clear()
    assert store.load_all() == []
    assert store.count() == 0


def test_invalid_limit_is_rejected(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    with pytest.raises(ValueError, match="limit"):
        store.load_all(limit=0)
