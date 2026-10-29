"""Pure, Streamlit-free helpers so they can be unit-tested without a server."""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd

ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode bytes into a BGR image, or None if the bytes are not an image."""
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode_png(image_bgr: np.ndarray) -> bytes | None:
    ok, buf = cv2.imencode(".png", image_bgr)
    return buf.tobytes() if ok else None


@contextmanager
def temporary_image_file(image_bgr: np.ndarray) -> Iterator[Path]:
    """Write an image to a unique temp file and always delete it afterwards.

    A unique name per call avoids collisions between concurrent sessions.
    """
    fd, name = tempfile.mkstemp(prefix="visionforge_", suffix=".png")
    os.close(fd)
    path = Path(name)
    try:
        if not cv2.imwrite(str(path), image_bgr):
            raise RuntimeError("Could not prepare the input image.")
        yield path
    finally:
        path.unlink(missing_ok=True)


def detections_to_frame(detections: list[dict[str, Any]] | None) -> pd.DataFrame:
    """Build the display table from the engine's `detections` list.

    Uses only the keys the existing app already relies on: `label`, `confidence`.
    """
    columns = ["ID", "Object", "Confidence"]
    if not detections:
        return pd.DataFrame(columns=columns)
    rows = [
        {
            "ID": i,
            "Object": d.get("label", "unknown"),
            "Confidence": float(d.get("confidence", 0.0)),
        }
        for i, d in enumerate(detections, start=1)
    ]
    return pd.DataFrame(rows, columns=columns)


def class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Object", "Count"])
    return (
        frame["Object"].value_counts().rename_axis("Object").reset_index(name="Count")
    )


def summarize_analysis(
    filename: str, result: dict[str, Any], threshold: float
) -> dict[str, Any]:
    """Compact record stored in session history (no image data)."""
    frame = detections_to_frame(result.get("detections"))
    return {
        "kind": "image",
        "filename": filename,
        "threshold": threshold,
        "objects": int(result.get("object_count", len(frame))),
        "inference_ms": float(result.get("inference_time_ms", 0.0)),
        "width": int(result.get("image_width", 0)),
        "height": int(result.get("image_height", 0)),
        "mean_confidence": float(frame["Confidence"].mean()) if len(frame) else None,
        "classes": frame["Object"].tolist(),
        "confidences": frame["Confidence"].tolist(),
    }


def history_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten every detection across the session into one frame."""
    rows = []
    for n, record in enumerate(history, start=1):
        for lbl, conf in zip(record["classes"], record["confidences"]):
            rows.append(
                {"Analysis": n, "Image": record["filename"],
                 "Object": lbl, "Confidence": conf}
            )
    return pd.DataFrame(rows, columns=["Analysis", "Image", "Object", "Confidence"])


def find_weights(project_root: Path, model_name: str = "yolo11n.pt") -> Path | None:
    """Look for local weights in common locations. Returns None if not found."""
    for candidate in (
        project_root / model_name,
        project_root / "models" / model_name,
        project_root / "weights" / model_name,
        Path.cwd() / model_name,
    ):
        if candidate.is_file():
            return candidate
    return None


# ------------------------------------------------------------------ OCR


INDIAN_REG_PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$"
)


def create_ocr_variants_legacy(crop_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """LEGACY variant builder, kept verbatim ONLY as the baseline for before/after evaluation.

    Known defects (see create_ocr_variants): the 'color_x2' variant is smaller than 'color' for small
    crops, and the adaptive threshold erases light-on-dark text. Do not use in production.

    Generate preprocessing variants for OCR augmentation.

    Variants (ordered best-first):
      1.  color upscale      — original BGR, INTER_CUBIC upsample
      2.  gray clahe         — genuine 8-bit grayscale with CLAHE (clipLimit=1.5)
      3.  gray adaptive_th   — actual 8-bit grayscale + adaptive Gaussian threshold
      4.  color upscale x2   — color upscaled more aggressively (2x stride)
    """
    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        return [("color", crop_bgr)]

    # --- Variant 1: upscale color (baseline) ---
    scale = max(1.0, 80.0 / max(h, 1))
    scale = min(scale, 1200.0 / max(w, 1)) if w * scale > 1200 else scale
    up_color = cv2.resize(
        crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    # Greyscale once (shared base for more variants)
    gray_base = cv2.cvtColor(up_color, cv2.COLOR_BGR2GRAY)

    # --- Variant 2: grayscale + CLAHE (gentler than original clipLimit=2.0) ---
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_base)

    # --- Variant 3: grayscale + adaptive thresholding ---
    gray_adaptive = cv2.adaptiveThreshold(
        gray_base,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=4,
    )

    # --- Variant 4: further upscale colour (readability boost) ---
    double_scale = min(scale * 2.0, 3.0)
    up_color_x2 = cv2.resize(
        crop_bgr, None, fx=double_scale, fy=double_scale,
        interpolation=cv2.INTER_CUBIC,
    )

    return [
        ("color", up_color),
        ("gray_clahe", gray_clahe),
        ("gray_adaptive", gray_adaptive),
        ("color_x2", up_color_x2),
    ]
def is_valid_crop(crop: Any) -> bool:
    """True for a non-empty 2-D/3-channel image array at least 2x2 pixels."""
    return (isinstance(crop, np.ndarray) and crop.size > 0 and crop.ndim in (2, 3)
            and (crop.ndim == 2 or crop.shape[2] in (1, 3)) and min(crop.shape[:2]) >= 2)


def _dark_text_on_light(gray: np.ndarray) -> bool:
    """Otsu split: the majority class is the background. True when text is darker than background."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((gray > t).mean()) >= 0.5


def create_ocr_variants(crop_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Preprocessing variants for OCR. The source crop is never modified (every output is a new array).

    Variants:
      color         upscale to ~80 px height (INTER_CUBIC), or downscale very wide crops (INTER_AREA)
      gray_clahe    grayscale + mild CLAHE (local contrast)
      gray_adaptive blur + adaptive threshold with a block size proportional to the crop height and
                    polarity normalisation (text is made dark-on-light first), so light-on-dark
                    plates are not erased
      color_hi      a genuinely larger colour version (1.5x the base scale), skipped when the size
                    cap prevents it from being larger

    Returns [] for an invalid or empty crop (callers must treat that as "nothing to read").
    """
    if not is_valid_crop(crop_bgr):
        return []
    crop = crop_bgr if crop_bgr.ndim == 3 else cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    h, w = crop.shape[:2]
    max_w = 1200.0
    scale = max(1.0, 80.0 / h)
    scale = min(scale, max_w / w)

    def resize(factor: float) -> np.ndarray:
        if abs(factor - 1.0) < 1e-6:
            return crop.copy()
        interp = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
        return cv2.resize(crop, None, fx=factor, fy=factor, interpolation=interp)

    base = resize(scale)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)

    ink = gray if _dark_text_on_light(gray) else 255 - gray
    block = max(15, int(0.35 * base.shape[0]))
    block += 1 - block % 2  # must be odd
    adaptive = cv2.adaptiveThreshold(cv2.GaussianBlur(ink, (3, 3), 0), 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)

    out = [("color", base), ("gray_clahe", clahe), ("gray_adaptive", adaptive)]
    hi = min(scale * 1.5, max_w / w)
    if hi > scale * 1.05 and base.shape[0] < 160:  # only helps genuinely small crops
        out.append(("color_hi", resize(hi)))
    return out


def processing_label(record: dict[str, Any]) -> str:
    """Human-readable processing time for any history record (used by the Command Center)."""
    if record.get("kind") == "video":
        secs = record.get("processing_seconds", record.get("processing_time_seconds"))
        return format_seconds(secs)
    return f"{float(record.get('inference_ms', 0.0) or 0.0):.0f} ms"


def validate_characters(text: str) -> tuple[bool, list[str]]:
    """Return (is_valid, problem_chars).

    Checks that all characters are in the allowed set for Indian plates.
    """
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    problems: list[str] = []
    for ch in text:
        if ch not in allowed:
            problems.append(ch)
    return (len(problems) == 0, problems)


def correct_plate_format(raw_text: str) -> tuple[str, list[str], str]:
    """Apply light format-aware correction to an OCR plate string.

    Steps (only applied when justified):
      1. Clean the OCR text (strip non-alphanumeric, uppercase)
      2. Validate against common Indian registration patterns
      3. If validation fails, try corrections for visually similar
         characters (O/0, I/1, B/8, S/5, Z/2, A/4)
      4. Only accept a correction if the resulting text passes pattern
         validation AND the number of corrections is reasonable

    Returns (corrected_text, corrections_applied, confidence_note).
    Raw OCR is always returned as a baseline; corrections are returned
    separately.
    """
    raw = raw_text.strip().upper()
    cleaned = re.sub(r"[^A-Z0-9]", "", raw)

    # Start with a correction budget: at most 2 character changes
    # to avoid over-correction.
    MAX_CORRECTIONS = 2

    # Check original text against common Indian registration patterns
    is_valid = _validate_plate_pattern(cleaned)[0]
    if is_valid:
        return cleaned, [], "raw OCR valid"

    # Try corrections in order of likelihood
    # The key insight: only characters that are in the allowed set
    # can be corrected, and only when the correction makes the text
    # pattern-valid.
    confusion_candidates = [
        # (original_char, corrected_char)
        ("0", "O"),  # zero read as letter O
        ("1", "I"),  # one read as letter I
        ("8", "B"),  # eight read as letter B
        ("5", "S"),  # five read as letter S
        ("2", "Z"),  # two read as letter Z
        ("4", "A"),  # four read as letter A
        ("O", "0"),  # letter O read as zero
        ("I", "1"),  # letter I read as one
        ("B", "8"),  # letter B read as eight
        ("S", "5"),  # letter S read as five
        ("Z", "2"),  # letter Z read as two
        ("A", "4"),  # letter A read as four
    ]

    corrections: list[str] = []
    for _ in range(MAX_CORRECTIONS):
        # Try each candidate correction
        improved = False
        for orig, corrected_char in confusion_candidates:
            if orig not in cleaned:
                continue
            new_text = cleaned.replace(orig, corrected_char, 1)
            is_valid, _ = _validate_plate_pattern(new_text)
            if is_valid:
                corrections.append(f"{orig}->{corrected_char}")
                cleaned = new_text
                improved = True
                break
        if not improved:
            break

    is_valid, notes = _validate_plate_pattern(cleaned)
    if is_valid:
        return cleaned, corrections, "; ".join(notes) if notes else "pattern valid"

    return cleaned, corrections, "raw OCR kept; no valid correction found"


def _validate_plate_pattern(text: str) -> tuple[bool, list[str]]:
    """Check text against common Indian registration patterns.

    Returns (is_valid, notes).
    """
    notes: list[str] = []
    if not text:
        return False, ["empty"]

    length = len(text)

    # Indian state-code format: XX (state) + DD (district) + LLL (authority) + NNN (sequence)
    # e.g., UP32AB1234, DL4CAF1234, MH12AB1234, TN56CD1234
    # Pattern: 2 letters + 1-2 digits + 1-3 letters + 3-4 digits
    # Total length: 7-11 characters
    if 7 <= length <= 11:
        match = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{3,4})$", text)
        if match:
            return True, ["state_code_format"]

    # Classic/older Indian plate format: 3 letters + 4 digits
    # e.g., ABC12345
    if length == 7:
        match = re.match(r"^([A-Z]{3})(\d{4})$", text)
        if match:
            return True, ["older_format"]

    # Provisional/temporary format: 2 digits + 1-2 letters + 3-4 digits
    # e.g., 32AB1234
    if length == 8:
        match = re.match(r"^(\d{2})([A-Z]{1,2})(\d{3,4})$", text)
        if match:
            return True, ["preliminary_format"]

    notes.append("length")
    return False, notes


def format_bytes(size_bytes: int) -> str:
    """Convert a file size in bytes into a human-readable format."""
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"
def format_seconds(seconds: float) -> str:
    """Convert seconds into a human-readable duration."""
    if seconds is None:
        return "N/A"

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes}:{secs:02d}"


def video_records_frame(result: dict[str, Any] | None) -> pd.DataFrame:
    """Return video detections and tracking IDs as a stable DataFrame."""
    columns = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "class_id",
        "label",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    records = (result or {}).get("detections") or []
    frame = pd.DataFrame.from_records(records)

    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")

    frame = frame[columns]

    if not frame.empty:
        frame["frame"] = (
            pd.to_numeric(frame["frame"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        frame["timestamp_seconds"] = pd.to_numeric(
            frame["timestamp_seconds"], errors="coerce"
        ).fillna(0.0)

        frame["track_id"] = pd.to_numeric(
            frame["track_id"], errors="coerce"
        ).astype("Int64")

        frame["class_id"] = pd.to_numeric(
            frame["class_id"], errors="coerce"
        )

        frame["confidence"] = pd.to_numeric(
            frame["confidence"], errors="coerce"
        ).fillna(0.0)

        for column in ("x1", "y1", "x2", "y2"):
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

        frame["label"] = frame["label"].fillna("unknown").astype(str)

        frame = frame.sort_values(
            ["frame", "timestamp_seconds"],
            kind="stable",
        ).reset_index(drop=True)

    return frame



def detections_per_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Count detected boxes for each processed source frame."""
    columns = ["frame", "timestamp_seconds", "Detections"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"frame", "timestamp_seconds"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    return (
        frame.groupby(["frame", "timestamp_seconds"], dropna=False)
        .size()
        .reset_index(name="Detections")
        .sort_values("frame", kind="stable")
        .reset_index(drop=True)[columns]
    )


def summarize_video(
    filename: str,
    result: dict[str, Any],
    threshold: float,
    stride: int,
    max_frames: int | None,
) -> dict[str, Any]:
    """Compact video-analysis record compatible with the session history view."""
    frame = video_records_frame(result)
    processing_seconds = float(result.get("processing_time_seconds", 0.0) or 0.0)
    return {
        "kind": "video",
        "filename": filename,
        "threshold": float(threshold),
        "stride": int(stride),
        "max_frames": max_frames,
        "objects": int(result.get("total_detections", len(frame))),
        "inference_ms": processing_seconds * 1000.0,
        "width": int(result.get("video_width", 0) or 0),
        "height": int(result.get("video_height", 0) or 0),
        "mean_confidence": float(frame["confidence"].mean()) if not frame.empty else None,
        "classes": frame["label"].tolist(),
        "confidences": frame["confidence"].astype(float).tolist(),
        "frames_read": int(result.get("frames_read", 0) or 0),
        "frames_processed": int(result.get("frames_processed", 0) or 0),
        "processing_time_seconds": processing_seconds,
    }

def stamp(value=None):
    """Format a timestamp, or add a timestamp field to a structured record.

    Design:
    * stamp()                       -> current ISO timestamp string
    * stamp(datetime_obj)           -> ISO timestamp string
    * stamp(dict)                   -> dict with a new "timestamp" key
    * stamp(other)                  -> str(other)  (kept for backward compat)
    """
    from datetime import datetime

    if value is None:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    if isinstance(value, datetime):
        return value.astimezone().isoformat(timespec="seconds")

    if isinstance(value, dict):
        rec = dict(value)  # shallow copy – original is never mutated
        if "timestamp" not in rec:
            rec["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
        return rec

    return str(value)


def normalize_history(raw: list[object] | None) -> list[dict[str, Any]]:
    """Validate and normalise the session history list.

    Guarantees:
    * Returns a NEW list; the caller's list (and its items) are never mutated.
    * Only dict records are kept.  Legacy scalar/JSON-string records are
      reported via log.warning and dropped.
    * Required fields are given safe defaults so downstream consumers never
      raise KeyError.
    * No fabricated values are added: objects, classes and confidences are
      preserved when present, otherwise they stay as the recorded data,
      defaulting only to empty structures.

    Args:
        raw: The raw session_state.history, possibly containing
             malformed or legacy entries.

    Returns:
        A list of normalised dict history records.
    """
    import logging
    from datetime import datetime

    log = logging.getLogger("visionforge.history")
    if raw is None:
        return []

    cleaned: list[dict[str, Any]] = []
    for idx, rec in enumerate(raw):
        if not isinstance(rec, dict):
            log.warning(
                "History entry %d is not a dict (%s) and will be skipped.",
                idx,
                type(rec).__name__,
            )
            continue

        norm = dict(rec)  # shallow copy – original not mutated

        # --- kind ---
        kind = norm.get("kind", "unknown")
        if not isinstance(kind, str) or not kind:
            norm["kind"] = "unknown"
            log.warning(
                "History entry %d has missing/invalid kind; defaulted to 'unknown'.", idx
            )
        else:
            norm["kind"] = kind

        # --- timestamp ---
        if "timestamp" not in norm or not isinstance(norm["timestamp"], str):
            norm["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")

        # --- filename ---
        if "filename" not in norm or not isinstance(norm["filename"], str):
            norm["filename"] = "unknown"

        # --- objects ---
        objs = norm.get("objects")
        if not isinstance(objs, int):
            if isinstance(objs, str):
                try:
                    objs = int(objs)
                except ValueError:
                    objs = 0
            else:
                objs = 0
            log.warning(
                "History entry %d has non-integer objects; defaulted to %d.", idx, objs
            )
        norm["objects"] = objs

        # --- status (optional but useful for filtering) ---
        if "status" not in norm or not isinstance(norm["status"], str):
            norm["status"] = "unknown"

        # --- anything that looks like a legacy JSON-encoded string of a dict ---
        #     These cannot be reliably unpickled, but if the original JSON
        #     round-tripped through dumps/loads, it would be a dict.  Strings
        #     here are dropped and reported, never fabricated.

        # --- keep useful metadata if present ---
        if "kind" not in norm or not isinstance(norm.get("kind"), str):
            norm["kind"] = "unknown"

        cleaned.append(norm)

    return cleaned