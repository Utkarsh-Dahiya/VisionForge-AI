VisionForge AI — Object Tracking Update

Contents:
- core/video/video_processor.py: updated processor with optional Ultralytics tracking, ByteTrack, track IDs, visual trails, track summaries, and tracking CSV columns.

Installation:
1. Back up your existing file:
   Copy-Item .\core\video\video_processor.py .\core\video\video_processor_backup.py
2. Extract this ZIP into the project root, allowing the included core/video/video_processor.py to replace the existing file.
3. Run:
   python -m py_compile core/video/video_processor.py
   python -m pytest -q tests/test_video_processor.py
   python -m pytest -q

Notes:
- Tracking is enabled by default if the model supports model.track().
- Fake/test models with only model.predict() continue to work; their track_id values are blank.
- CSV now includes track_id. New result fields: tracking_enabled, unique_tracks, track_summaries.
- This file was syntax-compiled, but the full test suite has not been run against your local project. Please run the commands above.
