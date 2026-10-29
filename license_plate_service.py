"""License plate detection + OCR service (no Streamlit imports; fully testable).

Design rules
------------
* Plate detection needs a plate-trained YOLO model. A general COCO model (e.g.
  yolo11n.pt) is REJECTED: its class names contain no plate class, and this
  module refuses to report "plates" from it.
* Detection confidence and OCR confidence are separate numbers and are never mixed.
* OCR is optional. If EasyOCR (or its model files) is missing, detection still
  works and each plate is marked ``ocr_status="unavailable"``.
* Nothing here sends data over a network. EasyOCR itself downloads its model
  files on first use; see ``ocr_dependency_status``.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol

import cv2
import numpy as np
import pandas as pd

import plate_registration as preg

log = logging.getLogger("visionforge.plates")

PLATE_MODEL_ENV = "VISIONFORGE_PLATE_MODEL"
DEFAULT_PLATE_MODEL = Path("models") / "license_plate_detector.pt"
PLATE_NAME_PATTERN = re.compile(
    r"(licen[sc]e|number[\s_-]?plate|\bplates?\b|registration|\blp\b|\bnp\b)", re.I
)
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
LOW_DET_CONF = 0.50
LOW_OCR_CONF = 0.50
MAX_IMAGE_MB = 20


class PlateModelError(RuntimeError):
    """The plate model is missing, unloadable, or not a plate detector."""


class OCRUnavailableError(RuntimeError):
    """OCR cannot run (package missing, or model files could not be loaded)."""


# ------------------------------------------------------------------ configuration
def default_plate_model_path(project_root: Path) -> Path:
    """Configured path: env var VISIONFORGE_PLATE_MODEL, else models/license_plate_detector.pt."""
    env = os.environ.get(PLATE_MODEL_ENV, "").strip().strip('"')
    return Path(env).expanduser() if env else Path(project_root) / DEFAULT_PLATE_MODEL


def plate_model_file_status(path: str | Path | None) -> dict[str, Any]:
    """Cheap file-level check (does NOT load the model)."""
    if not path:
        return {"state": "missing", "path": None, "detail": "No path configured."}
    p = Path(path)
    if not p.is_file():
        return {"state": "missing", "path": str(p), "detail": "File not found."}
    if p.suffix.lower() != ".pt":
        return {"state": "invalid", "path": str(p), "detail": "Expected a .pt weights file."}
    size = p.stat().st_size
    if size == 0:
        return {"state": "invalid", "path": str(p), "detail": "File is empty."}
    return {"state": "found", "path": str(p), "size_mb": round(size / 1024 / 1024, 1),
            "detail": "Weights file present (not yet loaded)."}


def ocr_dependency_status() -> dict[str, Any]:
    """Report EasyOCR readiness without importing torch or downloading anything."""
    installed = importlib.util.find_spec("easyocr") is not None
    info: dict[str, Any] = {"installed": installed, "version": None, "weights_cached": None}
    if installed:
        try:
            info["version"] = metadata.version("easyocr")
        except metadata.PackageNotFoundError:
            pass
        model_dir = Path(os.environ.get("EASYOCR_MODULE_PATH", Path.home() / ".EasyOCR")) / "model"
        needed = ("craft_mlt_25k.pth", "english_g2.pth")
        info["model_dir"] = str(model_dir)
        info["weights_cached"] = all((model_dir / n).is_file() for n in needed)
    return info


# ------------------------------------------------------------------ geometry
def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int,
              pad: float = 0.0) -> tuple[int, int, int, int] | None:
    """Order, pad (fraction of box size) and clamp a box to the image.

    Returns integer (x1, y1, x2, y2) or None if less than 2 px remain on a side.
    """
    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    nx1, ny1 = max(0, math.floor(x1 - px)), max(0, math.floor(y1 - py))
    nx2, ny2 = min(width, math.ceil(x2 + px)), min(height, math.ceil(y2 + py))
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return None
    return int(nx1), int(ny1), int(nx2), int(ny2)


def resolve_plate_class_ids(names: dict[int, str]) -> tuple[list[int], list[str]]:
    """Pick the class ids that mean 'license plate'. Raises for general models."""
    if not names:
        raise PlateModelError("The model reports no class names, so it cannot be verified as a plate detector.")
    matches = [int(i) for i, n in names.items() if PLATE_NAME_PATTERN.search(str(n))]
    if matches:
        return matches, []
    if len(names) == 1:
        only_id, only_name = next(iter(names.items()))
        return [int(only_id)], [
            f"The model has a single class named '{only_name}'; it is being treated as the plate class."]
    preview = ", ".join(str(n) for n in list(names.values())[:8])
    more = "…" if len(names) > 8 else ""
    raise PlateModelError(
        f"None of this model's {len(names)} classes look like license plates ({preview}{more}). "
        "This looks like a general-purpose model (for example COCO), not a plate detector. "
        "Use weights trained to detect license plates.")


# ------------------------------------------------------------------ OCR
@dataclass
class OCRReading:
    """One OCR result. Raw and processed text are kept in separate fields."""
    text: str                       # processed/final text (what the UI shows); equals the raw text when nothing was changed
    confidence: float | None        # the recognizer's own score, 0..1; NOT the detection confidence
    variant: str = ""               # preprocessing candidate that produced this reading
    correction_note: str = ""       # kept for compatibility; same as correction_explanation
    correction_applied: bool = False  # True only when a position-justified character correction changed the text
    correction_details: list[str] = field(default_factory=list)  # e.g. ["I->1@4"]
    raw_text: str = ""              # everything the engine returned for the chosen variant, before any processing
    registration_status: str = ""   # valid | corrected | uncertain | empty
    registration_format: str = ""   # e.g. state_standard, bharat_series
    correction_explanation: str = ""  # why text was removed/corrected, or why it was left uncertain
    removed_text: list[str] = field(default_factory=list)  # incidental text ignored (e.g. ["IND"])
    pre_correction_text: str = ""   # text after incidental-text removal but BEFORE any character correction
    candidates: list[dict[str, Any]] = field(default_factory=list)  # audit trail of every candidate scored


class OCRBackend(Protocol):
    name: str

    def read(self, image_bgr: np.ndarray) -> list[tuple[Any, str, float]]:
        """Return (box_points, text, confidence) segments."""


class EasyOCRBackend:
    """Thin lazy wrapper around easyocr.Reader (CPU by default)."""

    name = "EasyOCR"

    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False,
                 readtext_kwargs: dict[str, Any] | None = None) -> None:
        self.languages = languages
        self.gpu = gpu
        self.readtext_kwargs = dict(readtext_kwargs or {})  # e.g. {"decoder": "beamsearch", "mag_ratio": 1.5}
        self._reader: Any = None

    def _get_reader(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise OCRUnavailableError(
                    "EasyOCR is not installed. Install it with: pip install easyocr") from exc
            try:
                self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu, verbose=False)
            except Exception as exc:  # download/network/permission/torch problems
                log.exception("EasyOCR reader could not be created")
                raise OCRUnavailableError(
                    "EasyOCR could not load its model files. The first run downloads them "
                    f"(needs internet, roughly 100 MB). Details: {exc}") from exc
        return self._reader

    def read(self, image_bgr: np.ndarray) -> list[tuple[Any, str, float]]:
        reader = self._get_reader()
        kwargs = {"allowlist": OCR_ALLOWLIST, "detail": 1, "paragraph": False, **self.readtext_kwargs}
        return reader.readtext(image_bgr, **kwargs)


class PaddleOCRBackend:
    """Optional lazy wrapper around PaddleOCR (paddleocr 2.x classic API).

    Design rules
    ------------
    * This is an ADDITIONAL backend: EasyOCR remains the default and is never
      replaced by this class.
    * ``paddleocr`` (and its runtime deps such as ``imgaug``) is imported
      lazily, so this module imports cleanly even when PaddleOCR is absent.
    * PaddleOCR downloads detection + recognition model files on first use
      (internet required, roughly 100-200 MB).
    * ``read()`` returns the same ``(box_points, text, confidence)`` segment
      list as ``EasyOCRBackend.read`` so the rest of the pipeline is backend
      agnostic.

    Environment note (Python 3.10, Windows, CPU, this session):
    paddlepaddle 3.3.1 installs and imports, but PaddleOCR 3.x/PaddleX
    static-graph inference raises ``NotImplementedError:
    ConvertPirAttribute2RuntimeAttribute not support`` on this build, and
    PaddleOCR 2.x requires ``numpy<2.0`` (this env has numpy 2.2.6) plus
    ``opencv-contrib-python``; ``cv2.pyd`` cannot be replaced here
    ([WinError 5] Access is denied -- the .venv lives under OneDrive). As a
    result ``read()`` raises :class:`OCRUnavailableError` with a precise
    message until the environment is fixed. See the OCR evaluation report.
    """

    name = "PaddleOCR"

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = False,
        show_log: bool = False,
    ) -> None:
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self._ocr: Any = None

    def _get_reader(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR as _PaddleOCR
            except ImportError as exc:
                raise OCRUnavailableError(
                    "PaddleOCR is not installed or its runtime dependencies "
                    "(e.g. imgaug) are missing. Install with: "
                    "pip install paddleocr==2.9.1 <runtime deps>"
                ) from exc
            try:
                self._ocr = _PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    show_log=self.show_log,
                )
            except Exception as exc:  # download/network/model/runtime problems
                log.exception("PaddleOCR reader could not be created")
                raise OCRUnavailableError(
                    "PaddleOCR could not initialise (model download or paddle "
                    "runtime failure). Details: %s" % exc
                ) from exc
        return self._ocr

    def read(self, image_bgr: np.ndarray) -> list[tuple[Any, str, float]]:
        ocr = self._get_reader()
        result = ocr.ocr(image_bgr)
        segments: list[tuple[Any, str, float]] = []
        if not result:
            return segments
        page = result[0]
        if not page:
            return segments
        for box, (text, score) in page:
            text = re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()
            if text:
                segments.append((box, text, float(score)))
        return segments


@dataclass(frozen=True)
class ReadOptions:
    """Knobs for ``read_plate_text``. Defaults are the production behaviour.

    The non-default values exist so each design decision can be measured on the same dataset
    (see evaluate_ocr.py): ``stop_early_conf`` re-creates the old early exit, ``rank='confidence'``
    ranks by OCR confidence only, ``variants='legacy'`` uses the old preprocessing.
    """
    stop_early_conf: float | None = None   # None = evaluate every variant
    rank: str = "evidence"                 # "evidence" or "confidence"
    reject_incidental: bool = True
    correct: bool = True
    max_changes: int = 2
    variants: str = "revised"              # "revised" or "legacy"


DEFAULT_READ_OPTIONS = ReadOptions()


def preprocess_variants(crop_bgr: np.ndarray, variant_set: str = "revised") -> list[tuple[str, np.ndarray]]:
    """Preprocessing variants for OCR (delegates to helpers; heavy image work lives there)."""
    import helpers

    if variant_set == "legacy":
        return helpers.create_ocr_variants_legacy(crop_bgr)
    return helpers.create_ocr_variants(crop_bgr)


# ------------------------------------------------------------------ segment handling
def _segment_items(segments: list[tuple[Any, str, float]]) -> list[dict[str, Any]]:
    items = []
    for box, text, conf in segments:
        raw = str(text)
        cleaned = preg.clean(raw)
        if not cleaned:
            continue
        pts = np.array(box, dtype=float).reshape(-1, 2)
        items.append({"raw": raw, "text": cleaned, "conf": float(conf),
                      "cx": float(pts[:, 0].mean()), "cy": float(pts[:, 1].mean()),
                      "h": float(max(np.ptp(pts[:, 1]), 1.0))})
    return items


def _reading_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lines top->bottom, then left->right. Line tolerance comes from the TALL segments so that a
    small incidental marking (e.g. IND) does not distort the grouping."""
    if not items:
        return []
    hs = [d["h"] for d in items]
    tall = [h for h in hs if h >= 0.6 * max(hs)]
    tol = 0.6 * float(np.median(tall))
    items = sorted(items, key=lambda d: d["cy"])
    lines: list[list[dict]] = [[items[0]]]
    for it in items[1:]:
        if abs(it["cy"] - np.mean([d["cy"] for d in lines[-1]])) <= tol:
            lines[-1].append(it)
        else:
            lines.append([it])
    return [d for line in lines for d in sorted(line, key=lambda d: d["cx"])]


def assemble_text(segments: list[tuple[Any, str, float]]) -> OCRReading:
    """Join OCR segments in reading order (lines top→bottom, left→right)."""
    ordered = _reading_order(_segment_items(segments))
    if not ordered:
        return OCRReading("", None)
    text = " ".join(d["text"] for d in ordered)
    reading = OCRReading(text, float(np.mean([d["conf"] for d in ordered])))
    reading.raw_text = " ".join(d["raw"] for d in ordered)
    return reading


def _flag_incidental(ordered: list[dict[str, Any]], img_w: float) -> dict[int, str]:
    """Segments that MAY be non-registration markings. They are only candidates for removal:
    whether they really are removed is decided by registration-format evidence, not by a fixed crop."""
    flags: dict[int, str] = {}
    for i, d in enumerate(ordered):
        sp = preg.split_incidental(d["text"])
        if sp:
            flags[i] = f"incidental word '{sp[0]}'"
    top = max((d["h"] for d in ordered), default=0.0)
    for i, d in enumerate(ordered):
        if i not in flags and len(ordered) > 1 and d["h"] < 0.55 * top \
                and d["cx"] < 0.30 * img_w and len(d["text"]) <= 4:
            flags[i] = "small text near the left edge"
    return flags


@dataclass
class _Candidate:
    variant: str
    variant_index: int
    text: str                       # kept characters, concatenated (before any correction)
    conf: float                     # character-weighted mean of the kept segments' confidences
    removed: list[tuple[str, str]]  # (removed text, reason)
    raw_text: str                   # everything the engine returned for this variant
    evidence: preg.Evidence
    correction: preg.Correction | None
    why_not: str
    leftover_incidental: bool
    score: float = 0.0
    status: str = "uncertain"

    @property
    def final_text(self) -> str:
        return self.correction.text if self.correction else self.text


def _weighted_conf(items: list[dict[str, Any]]) -> float:
    w = sum(len(d["text"]) for d in items)
    return float(sum(d["conf"] * len(d["text"]) for d in items) / w) if w else 0.0


def _candidates_for_variant(name: str, index: int, segments: list, img_w: float,
                            opts: ReadOptions) -> list[_Candidate]:
    ordered = _reading_order(_segment_items(segments))
    if not ordered:
        return []
    raw_all = " ".join(d["raw"] for d in ordered)
    alternatives: list[tuple[list[dict], list[tuple[str, str]]]] = [(ordered, [])]
    flags = _flag_incidental(ordered, img_w) if opts.reject_incidental else {}
    if flags:
        kept, removed = [], []
        for i, d in enumerate(ordered):
            if i not in flags:
                kept.append(d)
                continue
            sp = preg.split_incidental(d["text"])
            if sp and sp[1]:                       # IND glued to other text: keep the remainder
                kept.append({**d, "text": sp[1]})
                removed.append((sp[0], flags[i]))
            else:
                removed.append((d["text"], flags[i]))
        if kept:
            alternatives.append((kept, removed))
    out = []
    for kept, removed in alternatives:
        text = "".join(d["text"] for d in kept)
        if not text:
            continue
        ev = preg.validate(text)
        corr, why = (None, "")
        if not ev.valid and opts.correct:
            corr, why = preg.propose_correction(text, opts.max_changes)
        elif not ev.valid:
            why = "automatic correction disabled"
        leftover = any(preg.split_incidental(d["text"]) for d in kept)
        out.append(_Candidate(name, index, text, _weighted_conf(kept), removed, raw_all, ev, corr, why, leftover))
    return out


def _score(c: _Candidate, opts: ReadOptions) -> None:
    if c.evidence.valid:
        known = c.evidence.state_known is not False
        base = (100 if c.evidence.tier == 2 else 80) if known else (65 if c.evidence.tier == 2 else 50)
        c.status = "valid"
    elif c.correction:
        base = 55 - 15 * len(c.correction.changes)
        c.status = "corrected"
    else:
        base, c.status = 0.0, "uncertain"
    if opts.rank == "confidence":
        c.score = c.conf * 100.0
        return
    c.score = base + 25.0 * c.conf + (3.0 if len(c.text) in (9, 10) else 0.0) - (40.0 if c.leftover_incidental else 0.0)


def _explain(c: _Candidate) -> str:
    parts = []
    if c.removed:
        parts.append("ignored non-registration text: " + ", ".join(f"{t} ({r})" for t, r in c.removed))
    if c.correction:
        parts.append(f"corrected {len(c.correction.changes)} character(s) by position: "
                     + ", ".join(c.correction.describe()))
    elif c.status == "uncertain":
        parts.append("not a recognised Indian registration format; raw OCR text kept"
                     + (f" ({c.why_not})" if c.why_not else ""))
    return "; ".join(parts)


def read_plate_text(ocr: OCRBackend, crop_bgr: np.ndarray,
                    options: ReadOptions | None = None) -> OCRReading:
    """Read a plate crop. Every variant is evaluated; candidates are ranked by registration evidence.

    Per variant the OCR segments are grouped in reading order. Candidates are built with and without
    segments that look like incidental markings (``IND`` ...). Each candidate is scored on:
      * format evidence (supported Indian formats, known state/UT code, non-zero district),
      * a position-justified correction when the raw text is invalid (see plate_registration),
      * OCR confidence (character-weighted), as a secondary signal,
      * a penalty for leftover incidental words, and a bonus for agreement between variants.
    The raw OCR text, the processed text and any correction are reported separately; nothing is
    forced to match a pattern. The source crop is never modified.
    """
    import helpers

    opts = options or DEFAULT_READ_OPTIONS
    empty = OCRReading("", None, registration_status="empty")
    if not helpers.is_valid_crop(crop_bgr):
        empty.correction_explanation = empty.correction_note = "invalid or empty crop; OCR not run"
        return empty

    per_variant: list[_Candidate] = []
    all_candidates: list[_Candidate] = []
    for idx, (name, img) in enumerate(preprocess_variants(crop_bgr, opts.variants)):
        segments = ocr.read(img)
        cands = _candidates_for_variant(name, idx, segments, float(img.shape[1]), opts)
        for c in cands:
            _score(c, opts)
        all_candidates.extend(cands)
        if cands:
            per_variant.append(max(cands, key=lambda c: (c.score, c.conf)))
        if opts.stop_early_conf is not None and per_variant and \
                max(c.conf for c in per_variant) >= opts.stop_early_conf:
            break
    if not per_variant:
        return empty

    if opts.rank == "evidence":  # agreement between variants is extra evidence
        votes: dict[str, int] = {}
        for c in per_variant:
            votes[c.final_text] = votes.get(c.final_text, 0) + 1
        for c in all_candidates:
            c.score += min(6.0 * (votes.get(c.final_text, 1) - 1), 18.0) if c in per_variant else 0.0
    best = max(per_variant, key=lambda c: (c.score, c.conf, -c.variant_index))

    only = preg.split_incidental(best.text)
    if opts.reject_incidental and only and not only[1]:  # nothing but an incidental marking was read
        empty.raw_text = best.raw_text
        empty.removed_text = [only[0]]
        empty.correction_explanation = empty.correction_note = (
            f"only non-registration text was found ({only[0]}); no registration number read")
        return empty

    reading = OCRReading(best.final_text, best.conf, variant=best.variant)
    reading.raw_text = best.raw_text
    reading.pre_correction_text = best.text
    reading.registration_status = best.status
    reading.registration_format = best.evidence.format or (best.correction.format if best.correction else "")
    reading.correction_applied = best.correction is not None
    reading.correction_details = best.correction.describe() if best.correction else []
    reading.correction_explanation = reading.correction_note = _explain(best)
    reading.removed_text = [t for t, _ in best.removed]
    reading.candidates = [{"variant": c.variant, "text": c.text, "final_text": c.final_text,
                           "confidence": round(c.conf, 4), "score": round(c.score, 2), "status": c.status,
                           "removed": [t for t, _ in c.removed]} for c in all_candidates]
    return reading


# ------------------------------------------------------------------ legacy baseline (verbatim logic)
def _assemble_text_legacy(segments: list[tuple[Any, str, float]]) -> OCRReading:
    """The original assemble_text, kept so the baseline pipeline is reproduced exactly."""
    items = []
    for box, text, conf in segments:
        text = re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()
        if not text:
            continue
        pts = np.array(box, dtype=float).reshape(-1, 2)
        items.append({"text": text, "conf": float(conf), "cx": pts[:, 0].mean(),
                      "cy": pts[:, 1].mean(), "h": max(np.ptp(pts[:, 1]), 1.0)})
    if not items:
        return OCRReading("", None)
    items.sort(key=lambda d: d["cy"])
    tol = 0.6 * float(np.median([d["h"] for d in items]))
    lines: list[list[dict]] = [[items[0]]]
    for it in items[1:]:
        if abs(it["cy"] - np.mean([d["cy"] for d in lines[-1]])) <= tol:
            lines[-1].append(it)
        else:
            lines.append([it])
    ordered = [d for line in lines for d in sorted(line, key=lambda d: d["cx"])]
    text = " ".join(d["text"] for d in ordered)
    return OCRReading(text, float(np.mean([d["conf"] for d in ordered])))


def read_plate_text_legacy(ocr: OCRBackend, crop_bgr: np.ndarray) -> OCRReading:
    """The ORIGINAL pipeline, preserved for before/after comparison. Do not use in production.

    Decisions are unchanged from the original: variants from ``create_ocr_variants_legacy``, the
    highest mean-confidence variant wins, evaluation stops at the first best >= 0.6, and weak reads
    (< 0.4) get the position-agnostic ``helpers.correct_plate_format`` swap. Only observability was
    added: ``raw_text`` records the text BEFORE that correction (the original overwrote it), and the
    meaningless note "certificate" was replaced by a plain description.
    """
    import helpers

    best = OCRReading("", None)
    for name, img in helpers.create_ocr_variants_legacy(crop_bgr):
        reading = _assemble_text_legacy(ocr.read(img))
        reading.variant = name
        if reading.text:
            if best.confidence is None or (reading.confidence or 0) > (best.confidence or 0):
                best = reading
            if (best.confidence or 0) >= 0.6:
                break
    if not best.text:
        best.registration_status = "empty"
        return best
    best.raw_text = best.pre_correction_text = best.text
    if (best.confidence or 0) < 0.4:
        corrected, corrections, note = helpers.correct_plate_format(best.text)
        best.correction_note = note
        best.correction_applied = len(corrections) > 0 and corrected != best.text
        if best.correction_applied:
            best.text = corrected
            best.correction_details = corrections
    else:
        best.correction_note = "raw OCR kept (confidence >= 0.4)"
        best.correction_applied = False
    best.correction_explanation = best.correction_note
    return best


# ------------------------------------------------------------------ named pipelines (for evaluation)
PIPELINES: dict[str, Callable[[OCRBackend, np.ndarray], OCRReading]] = {
    "legacy": read_plate_text_legacy,
    "revised": lambda ocr, crop: read_plate_text(ocr, crop, ReadOptions()),
    # ablations: change ONE design decision relative to "revised"
    "revised_early_exit": lambda ocr, crop: read_plate_text(ocr, crop, ReadOptions(stop_early_conf=0.6)),
    "revised_confidence_rank": lambda ocr, crop: read_plate_text(
        ocr, crop, ReadOptions(rank="confidence", reject_incidental=False, correct=False)),
    "revised_legacy_variants": lambda ocr, crop: read_plate_text(ocr, crop, ReadOptions(variants="legacy")),
}


# ------------------------------------------------------------------ service
class LicensePlateService:
    """Detect plates with a plate-trained YOLO model and read them with OCR."""

    # class-level defaults keep partially constructed instances (tools, tests) safe
    ocr: Any = None
    model_path: str = ""
    read_options: "ReadOptions" = DEFAULT_READ_OPTIONS
    model_warnings: list = []

    def __init__(self, model: Any, ocr: OCRBackend | None = None, model_path: str = "",
                 read_options: ReadOptions | None = None) -> None:
        self.model = model
        self.ocr = ocr
        self.model_path = model_path
        self.read_options = read_options or DEFAULT_READ_OPTIONS
        names = getattr(model, "names", None)
        self.class_names: dict[int, str] = dict(names) if names else {}
        self.class_ids, self.model_warnings = resolve_plate_class_ids(self.class_names)

    @classmethod
    def from_weights(cls, path: str | Path, ocr: OCRBackend | None = None,
                     loader: Callable[[str], Any] | None = None) -> "LicensePlateService":
        status = plate_model_file_status(path)
        if status["state"] != "found":
            raise PlateModelError(f"Plate model unavailable: {status['detail']} ({status['path']})")
        if loader is None:
            try:
                from ultralytics import YOLO as loader  # type: ignore[assignment]
            except ImportError as exc:
                raise PlateModelError("Ultralytics is not installed.") from exc
        try:
            model = loader(str(path))
        except Exception as exc:
            log.exception("Plate model failed to load: %s", path)
            raise PlateModelError(
                f"Could not load '{Path(path).name}': {exc}. The file may be corrupt or incompatible "
                "with the installed Ultralytics version.") from exc
        return cls(model, ocr=ocr, model_path=str(path))

    def analyze_image(self, image_bgr: np.ndarray, confidence: float = 0.25,
                      run_ocr: bool = True) -> dict[str, Any]:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image array.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0.")
        height, width = image_bgr.shape[:2]

        t0 = perf_counter()
        predictions = self.model.predict(source=image_bgr, conf=confidence, verbose=False)
        detection_ms = (perf_counter() - t0) * 1000
        if not predictions:
            raise RuntimeError("The plate model returned no prediction result.")
        boxes = predictions[0].boxes

        raw = []
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                if class_id not in self.class_ids:
                    continue
                raw.append((float(box.conf.item()), class_id, [float(v) for v in box.xyxy[0].tolist()]))
        raw.sort(key=lambda r: -r[0])

        detections: list[dict[str, Any]] = []
        skipped = 0
        for conf, class_id, xyxy in raw:
            clamped = clamp_box(*xyxy, width, height)
            padded = clamp_box(*xyxy, width, height, pad=0.06)
            if clamped is None or padded is None:
                skipped += 1
                continue
            x1, y1, x2, y2 = clamped
            px1, py1, px2, py2 = padded
            warnings = []
            if conf < LOW_DET_CONF:
                warnings.append("Low detection confidence")
            if y2 - y1 < 24:
                warnings.append("Plate region is very small; OCR is unreliable")
            if x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height:
                warnings.append("Plate touches the image edge and may be cut off")
            detections.append({
                "plate_id": len(detections) + 1, "class_id": class_id,
                "class_name": self.class_names.get(class_id, str(class_id)),
                "detection_confidence": round(conf, 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop": image_bgr[py1:py2, px1:px2].copy(),
                "crop_size": {"width": int(px2 - px1), "height": int(py2 - py1)},
                "plate_text": "", "ocr_confidence": None, "ocr_status": "disabled",
                "ocr_error": None, "ocr_variant": "", "warnings": warnings,
                # raw vs processed OCR output, kept in separate fields
                "raw_ocr_text": "", "ocr_candidate": "", "registration_status": "",
                "registration_format": "", "correction_applied": False,
                "correction_explanation": "", "removed_text": [],
            })

        ocr_error: str | None = None
        ocr_ms = 0.0
        if run_ocr:
            t1 = perf_counter()
            if self.ocr is None:
                ocr_error = "No OCR backend is configured."
                for d in detections:
                    d["ocr_status"] = "unavailable"
            else:
                for d in detections:
                    if ocr_error:  # OCR already known to be unavailable
                        d["ocr_status"] = "unavailable"
                        continue
                    try:
                        reading = read_plate_text(self.ocr, d["crop"], self.read_options)
                    except OCRUnavailableError as exc:
                        ocr_error = str(exc)
                        d["ocr_status"] = "unavailable"
                        continue
                    except Exception as exc:  # one bad plate must not lose the others
                        log.exception("OCR failed for plate %s", d["plate_id"])
                        d["ocr_status"], d["ocr_error"] = "error", f"{type(exc).__name__}: {exc}"
                        d["warnings"].append("OCR failed for this plate")
                        continue
                    d["ocr_variant"] = d["ocr_candidate"] = reading.variant
                    d["raw_ocr_text"] = reading.raw_text
                    d["registration_status"] = reading.registration_status
                    d["registration_format"] = reading.registration_format
                    d["correction_applied"] = reading.correction_applied
                    d["correction_explanation"] = reading.correction_explanation
                    d["removed_text"] = list(reading.removed_text)
                    if reading.text:
                        d["plate_text"], d["ocr_confidence"], d["ocr_status"] = reading.text, reading.confidence, "ok"
                        if reading.confidence is not None and reading.confidence < LOW_OCR_CONF:
                            d["warnings"].append("Low OCR confidence; verify the text")
                        if reading.removed_text:
                            d["warnings"].append("Ignored non-registration text: " + ", ".join(reading.removed_text))
                        if reading.correction_applied:
                            d["warnings"].append(f"Automatic correction applied ({', '.join(reading.correction_details)}); "
                                                 f"raw OCR read '{reading.raw_text}'")
                        elif reading.registration_status == "uncertain":
                            d["warnings"].append("Text does not match a known Indian registration format; verify it")
                    else:
                        d["ocr_status"] = "no_text"
                        d["warnings"].append("No text recognised")
            ocr_ms = (perf_counter() - t1) * 1000

        return {
            "image_width": width, "image_height": height,
            "detections": detections, "plate_count": len(detections), "skipped_degenerate": skipped,
            "annotated_image": self._annotate(image_bgr, detections),
            "detection_time_ms": round(detection_ms, 2), "ocr_time_ms": round(ocr_ms, 2),
            "inference_time_ms": round(detection_ms + ocr_ms, 2),
            "ocr_requested": bool(run_ocr), "ocr_backend": getattr(self.ocr, "name", None),
            "ocr_error": ocr_error, "model_path": self.model_path,
            "model_warnings": list(self.model_warnings),
        }

    @staticmethod
    def _annotate(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        thick = max(2, round(min(h, w) / 300))
        scale = max(0.5, min(h, w) / 900)
        for d in detections:
            b = d["bbox"]
            cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (238, 216, 63), thick)
            label = f"#{d['plate_id']} det {d['detection_confidence']:.2f}"
            if d["plate_text"]:
                label += f" | {d['plate_text']}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            top = b["y1"] - th - base - 4
            if top < 0:
                top = b["y2"]
            cv2.rectangle(out, (b["x1"], top), (min(w, b["x1"] + tw + 6), top + th + base + 4), (40, 30, 10), -1)
            cv2.putText(out, label, (b["x1"] + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 1, cv2.LINE_AA)
        return out


# ------------------------------------------------------------------ exports + history
ROW_COLUMNS = ["image", "plate_id", "x1", "y1", "x2", "y2", "detection_confidence",
               "ocr_text", "ocr_confidence", "ocr_status", "manual_correction", "warnings",
               "raw_ocr_text", "ocr_candidate", "registration_status", "registration_format",
               "correction_applied", "correction_explanation"]


def plate_rows(result: dict[str, Any], filename: str,
               corrections: dict[int, str] | None = None) -> list[dict[str, Any]]:
    """Export rows.

    ``raw_ocr_text`` is the engine output before any processing; ``ocr_text`` is the processed text
    (equal to the raw text when nothing was changed); ``manual_correction`` is the user's edit and is
    always separate from both.
    """
    corrections = corrections or {}
    rows = []
    for d in result["detections"]:
        fixed = (corrections.get(d["plate_id"]) or "").strip()
        rows.append({
            "image": filename, "plate_id": d["plate_id"], **d["bbox"],
            "detection_confidence": d["detection_confidence"],
            "ocr_text": d["plate_text"], "ocr_confidence": d["ocr_confidence"],
            "ocr_status": d["ocr_status"],
            "manual_correction": fixed if fixed and fixed != d["plate_text"] else "",
            "warnings": "; ".join(d["warnings"]),
            "raw_ocr_text": d.get("raw_ocr_text", ""), "ocr_candidate": d.get("ocr_candidate", ""),
            "registration_status": d.get("registration_status", ""),
            "registration_format": d.get("registration_format", ""),
            "correction_applied": bool(d.get("correction_applied", False)),
            "correction_explanation": d.get("correction_explanation", ""),
        })
    return rows


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    return pd.DataFrame(rows, columns=ROW_COLUMNS).to_csv(index=False)


def build_json_report(result: dict[str, Any], filename: str, threshold: float,
                      corrections: dict[int, str] | None = None) -> str:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image": {"name": filename, "width": result["image_width"], "height": result["image_height"]},
        "settings": {"detection_confidence_threshold": threshold, "ocr_requested": result["ocr_requested"]},
        "model": {"path_name": Path(result["model_path"]).name if result["model_path"] else None,
                  "warnings": result["model_warnings"]},
        "ocr": {"backend": result["ocr_backend"], "error": result["ocr_error"]},
        "timing_ms": {"detection": result["detection_time_ms"], "ocr": result["ocr_time_ms"],
                      "total": result["inference_time_ms"]},
        "note": "ocr_confidence is the OCR engine's own score and is not the detection confidence.",
        "plates": plate_rows(result, filename, corrections),
    }
    return json.dumps(report, indent=2)


def summarize_plate(filename: str, result: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Session-history record (no image data) for the Command Center and Analytics."""
    dets = result["detections"]
    confs = [d["detection_confidence"] for d in dets]
    attempted = result["ocr_requested"]
    return {
        "kind": "plate", "filename": filename, "threshold": float(threshold),
        "objects": len(dets), "inference_ms": float(result["inference_time_ms"]),
        "processing_seconds": float(result["inference_time_ms"]) / 1000.0,
        "width": result["image_width"], "height": result["image_height"],
        "mean_confidence": float(np.mean(confs)) if confs else None,
        "classes": ["license_plate"] * len(dets), "confidences": confs,
        "ocr_attempted": bool(attempted and dets),
        "ocr_read": sum(d["ocr_status"] == "ok" for d in dets),
        "ocr_failed": sum(d["ocr_status"] in ("no_text", "error", "unavailable") for d in dets) if attempted else 0,
    }