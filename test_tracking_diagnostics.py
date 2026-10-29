"""Unit tests for core.video.tracking_diagnostics (no video, model or Streamlit needed)."""
import pandas as pd
import pytest

from core.video.tracking_diagnostics import (
    METRIC_DEFINITIONS,
    DiagnosticsConfig,
    compute_tracking_diagnostics,
    diagnostics_table,
    diagnostics_warnings,
)

FPS = 10.0
W, H = 640, 480
BOX = (100, 100, 140, 180)  # 40 x 80 -> larger side 80


def rec(frame, tid, label="person", box=BOX):
    x1, y1, x2, y2 = box
    return {"frame": frame, "timestamp_seconds": frame / FPS, "track_id": tid, "class_id": 0,
            "label": label, "confidence": 0.9, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def run(records, **kwargs):
    kwargs.setdefault("fps", FPS)
    kwargs.setdefault("frame_width", W)
    kwargs.setdefault("frame_height", H)
    return compute_tracking_diagnostics(records, **kwargs)


def codes(diag):
    return {w["code"] for w in diag["warnings"]}


# ----------------------------------------------------------------- empty / missing IDs
@pytest.mark.parametrize("records", [None, [], pd.DataFrame()])
def test_empty_input_is_a_valid_outcome(records):
    diag = run(records)
    assert diag["detection_observations"] == 0
    assert diag["estimated_unique_objects"] == 0
    assert diag["distinct_track_ids"] == 0
    assert diag["warnings"] == []


def test_all_ids_missing_reports_untracked_not_tracks():
    diag = run([rec(f, None) for f in range(4)])
    assert diag["detection_observations"] == 4
    assert diag["tracked_observations"] == 0
    assert diag["untracked_observations"] == 4
    assert diag["distinct_track_ids"] == 0
    assert diag["estimated_unique_objects"] == 0
    assert "tracking_no_ids" in codes(diag)


def test_missing_ids_with_tracking_disabled_is_not_a_warning():
    diag = run([rec(f, None) for f in range(4)], tracking_enabled=False)
    assert diag["untracked_observations"] == 4
    assert "tracking_no_ids" not in codes(diag)


def test_mixed_ids_flags_untracked_frames():
    records = [rec(0, None), rec(1, None)] + [rec(f, 4) for f in (2, 3, 4)]
    diag = run(records)
    assert diag["tracked_observations"] == 3
    assert diag["untracked_observations"] == 2
    assert diag["frames_with_untracked_boxes"] == 2
    assert diag["distinct_track_ids"] == 1
    assert "untracked_observations" in codes(diag)


def test_missing_columns_do_not_crash():
    diag = run([{"frame": 0, "label": "person"}])
    assert diag["detection_observations"] == 1
    assert diag["tracked_observations"] == 0


# ----------------------------------------------------------------- observations vs tracks
def test_observations_are_not_unique_objects():
    diag = run([rec(f, 5) for f in range(10)])
    assert diag["detection_observations"] == 10
    assert diag["distinct_track_ids"] == 1
    assert diag["estimated_unique_objects"] == 1
    assert diag["track_length_observations"]["max"] == 10


def test_short_tracks_are_excluded_from_the_estimate():
    records = [rec(f, 1) for f in range(5)] + [rec(f, 2, box=(400, 300, 440, 380)) for f in (0, 1)]
    diag = run(records)
    assert diag["distinct_track_ids"] == 2
    assert diag["estimated_unique_objects"] == 1
    assert diag["short_track_count"] == 1
    assert diag["short_track_fraction"] == 0.5


def test_min_track_observations_is_configurable():
    records = [rec(f, 1) for f in range(2)]
    assert run(records)["estimated_unique_objects"] == 0
    cfg = DiagnosticsConfig(min_track_observations=2)
    assert run(records, config=cfg)["estimated_unique_objects"] == 1


def test_estimated_objects_by_class_and_peak_concurrency():
    records = []
    for f in range(4):
        records += [rec(f, 1, "person"), rec(f, 2, "person", (300, 100, 340, 180)),
                    rec(f, 3, "car", (400, 300, 480, 360))]
    diag = run(records)
    assert diag["estimated_unique_objects_by_class"] == {"car": 1, "person": 2}
    assert diag["peak_concurrent_tracks"] == 3
    assert diag["detection_observations"] == 12


def test_list_and_dataframe_inputs_agree():
    records = [rec(f, 1) for f in range(4)]
    assert run(records) == run(pd.DataFrame(records))


# ----------------------------------------------------------------- gaps and stride
def test_gap_detection_respects_frame_stride():
    # processed frames are 2 apart, so 0,2,4 is continuous but 4 -> 10 skips two updates
    assert run([rec(f, 1) for f in (0, 2, 4)], frame_stride=2)["tracks_with_gaps"] == 0
    diag = run([rec(f, 1) for f in (0, 2, 4, 10)], frame_stride=2)
    assert diag["tracks_with_gaps"] == 1
    assert diag["missing_observation_steps"] == 2


def test_stride_and_fps_warnings():
    diag = run([rec(0, 1)], frame_stride=3, fps_assumed=True)
    assert {"stride_tracking", "fps_assumed"} <= codes(diag)
    assert "stride_tracking" not in codes(run([rec(0, 1)], frame_stride=3, tracking_enabled=False))


# ----------------------------------------------------------------- suspicious boxes / classes
def test_class_switching_track_is_reported_with_majority_label():
    records = [rec(0, 1, "car"), rec(1, 1, "car"), rec(2, 1, "truck")]
    diag = run(records)
    assert diag["class_switching_tracks"] == 1
    assert diag["class_switch_examples"] == [{"track_id": 1, "labels": ["car", "truck"]}]
    assert diag["estimated_unique_objects_by_class"] == {"car": 1}
    assert "class_switching" in codes(diag)


def test_oversized_and_degenerate_boxes_are_flagged_not_removed():
    records = [rec(0, 1, box=(0, 0, 600, 450)), rec(1, 1, box=(10, 10, 10, 50)), rec(2, 1)]
    diag = run(records)
    assert diag["oversized_box_observations"] == 1
    assert diag["degenerate_box_observations"] == 1
    assert diag["detection_observations"] == 3  # nothing is suppressed
    assert {"oversized_boxes", "degenerate_boxes"} <= codes(diag)


# ----------------------------------------------------------------- fragmentation heuristic
def two_tracks(a_label="person", b_label="person", b_start=14, b_box=(110, 105, 150, 185)):
    return ([rec(f, 1, a_label) for f in range(0, 11)]
            + [rec(f, 2, b_label, b_box) for f in range(b_start, b_start + 7)])


def test_fragmentation_link_found_for_nearby_same_class_restart():
    diag = run(two_tracks())
    assert diag["possible_fragmentation_links"] == 1
    link = diag["possible_fragmentation_examples"][0]
    assert (link["from_track"], link["to_track"]) == (1, 2)
    assert link["gap_seconds"] == 0.4
    assert "possible_fragmentation" in codes(diag)


@pytest.mark.parametrize("kwargs", [
    {"b_label": "car"},                         # different class
    {"b_box": (560, 400, 600, 480)},            # far away
    {"b_start": 40},                            # gap of 3 s > 1.5 s
    {"b_start": 8},                             # overlaps in time with track 1
])
def test_fragmentation_not_reported_when_conditions_fail(kwargs):
    assert run(two_tracks(**kwargs))["possible_fragmentation_links"] == 0


def test_fragmentation_links_are_one_to_one():
    records = ([rec(f, 1) for f in range(0, 11)]
               + [rec(f, 2, box=(105, 100, 145, 180)) for f in range(0, 11)]
               + [rec(f, 3, box=(110, 105, 150, 185)) for f in range(14, 21)])
    assert run(records)["possible_fragmentation_links"] == 1


# ----------------------------------------------------------------- table / config
def test_diagnostics_table_defines_every_metric_it_shows():
    diag = run([rec(f, 1) for f in range(4)])
    table = diagnostics_table(diag)
    assert not table.empty
    assert (table["Meaning"].str.len() > 10).all()
    assert len(table) == len([k for k in METRIC_DEFINITIONS if k in diag])


def test_table_and_warnings_handle_missing_diagnostics():
    assert diagnostics_table(None).empty
    assert diagnostics_warnings(None) == []
    assert diagnostics_warnings(run([rec(0, None)])) != []


@pytest.mark.parametrize("kwargs", [
    {"min_track_observations": 0}, {"fragment_max_gap_seconds": 0},
    {"fragment_max_distance_factor": -1}, {"oversized_box_fraction": 0},
    {"oversized_box_fraction": 1.5},
])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DiagnosticsConfig(**kwargs)
