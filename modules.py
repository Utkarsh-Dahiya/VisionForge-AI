"""Single source of truth for module status (Command Center + Platform Overview).
Statuses are conservative: only functionality present in the app is marked implemented.
"""

APP_VERSION = "1.0.1"
APP_PHASE = "Video Analytics Upgrade"

IMPLEMENTED, PARTIAL, PLANNED = "Implemented", "Partial", "Planned"

MODULES = [
    {"icon": "◈", "name": "Vision Studio", "status": IMPLEMENTED,
     "desc": "Upload an image and run YOLO object detection.",
     "detail": "Single-image inference, annotated output, table and CSV export.", "target": "Vision Studio"},
    {"icon": "⌗", "name": "License Plate Studio", "status": PARTIAL,
     "desc": "Plate-specific detection and OCR.",
     "detail": "Requires custom plate-trained YOLO weights; OCR uses EasyOCR.", "target": "License Plate Studio"},
    {"icon": "▥", "name": "Analytics", "status": PARTIAL,
     "desc": "Charts and summaries from analysis results.",
     "detail": "Charts use available session results; video summaries include tracked classes and events.", "target": "Analytics"},
    {"icon": "⌁", "name": "Video Intelligence", "status": IMPLEMENTED,
     "desc": "Frame-by-frame video analysis with optional tracking.",
     "detail": "Annotated video, detection/tracking CSV, track summaries, and rule-based line/region events.", "target": "Video Intelligence"},
    {"icon": "⬡", "name": "Safety / PPE", "status": PLANNED,
     "desc": "PPE checks and configurable safety rules.", "detail": "Requires a suitable trained PPE model and validation data.", "target": None},
    {"icon": "◎", "name": "Tracking & Events", "status": PARTIAL,
     "desc": "Track IDs, line crossings, region entries/exits, and class-aware summaries.",
     "detail": "Rule-based events rely on observed tracks; accuracy depends on detections and tracking quality.", "target": "Video Intelligence"},
    {"icon": "⌗", "name": "OCR Studio", "status": PLANNED,
     "desc": "General text extraction from images.", "detail": "Separate general-purpose OCR workspace is not yet implemented.", "target": None},
    {"icon": "▤", "name": "Persistent storage", "status": IMPLEMENTED,
     "desc": "SQLite-backed analysis history.",
     "detail": "Stores compact image, plate and video analysis summaries locally.", "target": "Analytics"},
    {"icon": "✦", "name": "AI reporting", "status": PLANNED,
     "desc": "Generated analysis reports.", "detail": "Not started.", "target": None},
]

PAGES = ["Command Center", "Vision Studio", "License Plate Studio", "Video Intelligence", "Analytics", "Platform Overview"]
