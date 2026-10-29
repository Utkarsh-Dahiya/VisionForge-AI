import pandas as pd
import pytest
from core.analytics.event_engine import detect_video_events


def rows(points):
    return [dict(frame=i, timestamp_seconds=i/10, track_id=7, label='car',
                 x1=x-5, y1=y-5, x2=x+5, y2=y+5) for i, (x,y) in enumerate(points)]


def test_detects_line_crossing_and_region_entry_exit():
    df = detect_video_events(rows([(50, 20), (50, 40), (50, 60), (50, 90)]),
                             frame_width=100, frame_height=100, line_y_ratio=.5,
                             region=(.2, .5, .8, .8))
    assert list(df.event_type) == ['line_crossing', 'region_entry', 'region_exit']
    assert set(df.track_id) == {7}


def test_ignores_missing_track_ids():
    data = rows([(10, 10), (10, 80)])
    data[0]['track_id'] = None
    data[1]['track_id'] = None
    assert detect_video_events(data, frame_width=100, frame_height=100).empty


def test_rejects_invalid_region():
    with pytest.raises(ValueError):
        detect_video_events([], frame_width=100, frame_height=100, region=(.8,.2,.2,.8))
