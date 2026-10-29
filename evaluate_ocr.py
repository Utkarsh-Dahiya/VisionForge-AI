"""
VisionForge AI - OCR evaluation (end-to-end, raw vs processed, fair before/after comparison).

Usage (plates are whole photos; a plate detector is used):
    python evaluate_ocr.py --images_dir path/to/photos --csv ground_truth.csv --model_path models/license_plate_detector.pt
Usage (images are already tight plate crops; detection is skipped and detection recall is not reported):
    python evaluate_ocr.py --images_dir path/to/crops --csv ground_truth.csv --mode crops

The CSV must contain:  image_filename,ground_truth       (spaces/hyphens in the truth are ignored)

Design rules
------------
* Every labelled row is one item in the end-to-end denominator. Missing files, unreadable images,
  failed detections, empty OCR results and errors count as WRONG; none is dropped.
* Raw OCR text and processed text are recorded independently. Raw is never rebuilt from processed.
* All pipelines run in ONE invocation on the SAME images, the SAME detection and the SAME rules, so
  before/after numbers are comparable. The detector runs once per image and every pipeline reads the
  same crop.
* Detection "recall" here means "at least one plate box was found". Whether that box is correct needs
  box ground truth, which this CSV does not carry.
* Nothing is estimated: if no labelled data or no OCR engine is available the script says so and exits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import helpers  # noqa: E402
import license_plate_service as lps  # noqa: E402
import plate_registration as preg  # noqa: E402

log = logging.getLogger("visionforge.ocr_eval")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# statuses that mean the OCR step never produced a reading for this item
NOT_ATTEMPTED = {"image_missing", "image_unreadable", "detection_error", "no_detection", "crop_invalid"}
DEFAULT_PIPELINES = ("legacy", "revised")


# ------------------------------------------------------------------ text metrics
def normalize_plate(text: str) -> str:
    """Comparison form: upper-case A-Z/0-9 only (so 'KL 01 BT-1650' == 'KL01BT1650')."""
    return preg.clean(text)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def character_error_rate(predicted: str, truth: str) -> float:
    """Standard CER for one item: edit distance / len(truth). An empty prediction scores 1.0."""
    if not truth:
        raise ValueError("ground truth must not be empty")
    return edit_distance(predicted, truth) / len(truth)


_character_error_rate = character_error_rate  # backward-compatible name (now the standard definition)


# ------------------------------------------------------------------ dataset
@dataclass(frozen=True)
class Item:
    name: str            # file name as written in the CSV
    path: Path | None    # None when the file is missing
    truth: str           # normalised ground truth


def load_ground_truth_detailed(csv_path: Path) -> tuple[dict[str, str], list[str], list[str]]:
    """Return (labels, blank_rows, duplicate_rows). Nothing is dropped silently."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    labels: dict[str, str] = {}
    blank: list[str] = []
    dup: list[str] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_filename", "ground_truth"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {required}, found: {reader.fieldnames}")
        for row in reader:
            name = (row["image_filename"] or "").strip()
            truth = normalize_plate(row["ground_truth"] or "")
            if not name:
                continue
            if not truth:
                blank.append(name)
            elif name in labels:
                dup.append(name)
            else:
                labels[name] = truth
    return labels, blank, dup


def load_ground_truth(csv_path: Path) -> dict[str, str]:
    """Labels only (kept for backward compatibility)."""
    return load_ground_truth_detailed(csv_path)[0]


def collect_items(images_dir: Path, labels: dict[str, str]) -> tuple[list[Item], list[str]]:
    """One Item per labelled row (missing files included) + images that have no label."""
    on_disk = {p.name.lower(): p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    items, used = [], set()
    for name, truth in labels.items():
        path = on_disk.get(name.lower())
        if path is not None:
            used.add(path.name.lower())
        items.append(Item(name, path, truth))
    unlabelled = sorted(p.name for k, p in on_disk.items() if k not in used)
    return items, unlabelled


# ------------------------------------------------------------------ OCR cache (optional, accuracy-only)
class CachingOCR:
    """Memoises engine output per image so several pipelines share identical segments.
    Latency measured with this enabled is NOT representative (cache hits are free)."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.name = inner, getattr(inner, "name", "ocr")
        self._cache: dict[str, Any] = {}
        self.hits = self.misses = 0

    def read(self, image: np.ndarray):
        key = hashlib.sha1(image.tobytes() + str(image.shape).encode()).hexdigest()
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        self._cache[key] = self.inner.read(image)
        return self._cache[key]


# ------------------------------------------------------------------ per-item evaluation
def _blank_row(item: Item, pipeline: str) -> dict[str, Any]:
    return {"image": item.name, "ground_truth": item.truth, "pipeline": pipeline, "status": "",
            "detected": False, "n_detections": 0, "detection_confidence": None,
            "raw_text": "", "processed_text": "", "pre_correction_text": "",
            "raw_exact": False, "processed_exact": False,
            "raw_edit": len(item.truth), "processed_edit": len(item.truth),
            "ocr_confidence": None, "candidate": "", "registration_status": "",
            "correction_applied": False, "correction_explanation": "",
            "detect_ms": None, "ocr_ms": None, "total_ms": None,
            "crop_width": None, "crop_height": None, "error": ""}


def _fill_prediction(row: dict[str, Any], reading: lps.OCRReading, truth: str) -> None:
    """Raw and processed come from DIFFERENT fields of the reading; neither is derived from the other."""
    row["raw_text"] = reading.raw_text or ""
    row["processed_text"] = reading.text or ""
    row["pre_correction_text"] = reading.pre_correction_text or ""
    raw_n, proc_n = normalize_plate(row["raw_text"]), normalize_plate(row["processed_text"])
    row["raw_exact"], row["processed_exact"] = raw_n == truth, proc_n == truth
    row["raw_edit"], row["processed_edit"] = edit_distance(raw_n, truth), edit_distance(proc_n, truth)
    row["ocr_confidence"] = reading.confidence
    row["candidate"] = reading.variant
    row["registration_status"] = reading.registration_status
    row["correction_applied"] = bool(reading.correction_applied)
    row["correction_explanation"] = reading.correction_explanation or reading.correction_note or ""
    row["status"] = "ok" if proc_n or raw_n else "empty_ocr"


def evaluate_item(item: Item, service: lps.LicensePlateService,
                  pipelines: dict[str, Callable[[Any, np.ndarray], lps.OCRReading]],
                  mode: str = "e2e", det_conf: float = 0.25,
                  crop_dir: Path | None = None) -> list[dict[str, Any]]:
    """Evaluate one labelled item under every pipeline. Returns one row per pipeline."""
    rows = {name: _blank_row(item, name) for name in pipelines}

    def finish(status: str, error: str = "", only: list[str] | None = None) -> list[dict[str, Any]]:
        for n in (only or list(rows)):
            rows[n]["status"], rows[n]["error"] = status, error
        return list(rows.values())

    if item.path is None:
        return finish("image_missing", "file listed in the CSV was not found in the images folder")
    image = cv2.imread(str(item.path))
    if image is None:
        return finish("image_unreadable", "OpenCV could not decode the file")

    detect_ms = 0.0
    if mode == "crops":
        crop, det_conf_val, n_det = image, None, 1
    else:
        t0 = time.perf_counter()
        try:
            result = service.analyze_image(image, confidence=det_conf, run_ocr=False)
        except Exception as exc:  # recorded as a failed item, never skipped
            return finish("detection_error", f"{type(exc).__name__}: {exc}")
        detect_ms = (time.perf_counter() - t0) * 1000
        dets = sorted(result["detections"], key=lambda d: -d["detection_confidence"])
        if not dets:
            for r in rows.values():
                r["detect_ms"] = round(detect_ms, 1)
            return finish("no_detection", "the plate detector found no plate")
        crop, det_conf_val, n_det = dets[0]["crop"], dets[0]["detection_confidence"], len(dets)
    for r in rows.values():
        r.update(detected=True, n_detections=n_det, detection_confidence=det_conf_val, detect_ms=round(detect_ms, 1))
    if not helpers.is_valid_crop(crop):
        return finish("crop_invalid", "the crop passed to OCR is empty or too small")
    for r in rows.values():
        r["crop_width"], r["crop_height"] = int(crop.shape[1]), int(crop.shape[0])
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_dir / f"{Path(item.name).stem}_crop.png"), crop)

    for name, fn in pipelines.items():
        row = rows[name]
        t1 = time.perf_counter()
        try:
            reading = fn(service.ocr, crop)
        except lps.OCRUnavailableError as exc:
            row["status"], row["error"] = "ocr_unavailable", str(exc)
        except Exception as exc:
            row["status"], row["error"] = "ocr_error", f"{type(exc).__name__}: {exc}"
        else:
            _fill_prediction(row, reading, item.truth)
        row["ocr_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        row["total_ms"] = round(detect_ms + row["ocr_ms"], 1)
    return list(rows.values())


def evaluate_dataset(items: list[Item], service: lps.LicensePlateService,
                     pipelines: dict[str, Callable[[Any, np.ndarray], lps.OCRReading]],
                     mode: str = "e2e", det_conf: float = 0.25,
                     crop_dir: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """{pipeline: [row per labelled item]}; every pipeline has exactly len(items) rows."""
    out: dict[str, list[dict[str, Any]]] = {n: [] for n in pipelines}
    for item in items:
        for row in evaluate_item(item, service, pipelines, mode, det_conf, crop_dir):
            out[row["pipeline"]].append(row)
    return out


# ------------------------------------------------------------------ metrics
def _lat(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "n": 0}
    a = np.asarray(values, float)
    return {"mean": round(float(a.mean()), 1), "median": round(float(np.median(a)), 1),
            "p95": round(float(np.percentile(a, 95)), 1), "n": len(values)}


def summarize(rows: list[dict[str, Any]], mode: str = "e2e") -> dict[str, Any]:
    """Metrics for ONE pipeline. The denominator ``n`` is every labelled item (nothing excluded)."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    detected = sum(1 for r in rows if r["detected"])
    readable = n - statuses.get("image_missing", 0) - statuses.get("image_unreadable", 0)
    attempted = [r for r in rows if r["status"] in ("ok", "empty_ocr")]
    gt_len = sum(len(r["ground_truth"]) for r in rows)

    def rate(k: int, d: int) -> float | None:
        return round(k / d, 4) if d else None

    m: dict[str, Any] = {
        "n": n,
        "statuses": statuses,
        "readable_images": readable,
        "detected": detected,
        "detection_rate_of_readable": rate(detected, readable) if mode == "e2e" else None,
        "detection_rate_of_all": rate(detected, n) if mode == "e2e" else None,
        "ocr_attempted": len(attempted),
        "empty_ocr": statuses.get("empty_ocr", 0),
        "failed": sum(statuses.get(k, 0) for k in NOT_ATTEMPTED | {"ocr_error", "ocr_unavailable"}),
        # end-to-end: denominator is ALL labelled items
        "raw_exact_matches": sum(r["raw_exact"] for r in rows),
        "raw_exact_rate": rate(sum(r["raw_exact"] for r in rows), n),
        "processed_exact_matches": sum(r["processed_exact"] for r in rows),
        "processed_exact_rate": rate(sum(r["processed_exact"] for r in rows), n),
        # conditional on OCR having run
        "raw_exact_rate_given_ocr": rate(sum(r["raw_exact"] for r in attempted), len(attempted)),
        "processed_exact_rate_given_ocr": rate(sum(r["processed_exact"] for r in attempted), len(attempted)),
        # corpus CER = total edits / total ground-truth characters; failures cost len(truth) edits
        "raw_cer": round(sum(r["raw_edit"] for r in rows) / gt_len, 4) if gt_len else None,
        "processed_cer": round(sum(r["processed_edit"] for r in rows) / gt_len, 4) if gt_len else None,
        "raw_cer_mean_per_image": round(float(np.mean([r["raw_edit"] / len(r["ground_truth"]) for r in rows])), 4),
        "processed_cer_mean_per_image": round(float(np.mean([r["processed_edit"] / len(r["ground_truth"]) for r in rows])), 4),
        "latency_detect_ms": _lat([r["detect_ms"] for r in rows if r["detect_ms"] is not None]),
        "latency_ocr_ms": _lat([r["ocr_ms"] for r in rows if r["ocr_ms"] is not None]),
        "latency_total_ms": _lat([r["total_ms"] for r in rows if r["total_ms"] is not None]),
    }
    applied = [r for r in rows if r["correction_applied"]]
    m["corrections"] = {
        "applied": len(applied),
        "helped": sum(1 for r in applied if r["processed_exact"] and normalize_plate(r["pre_correction_text"]) != r["ground_truth"]),
        "harmed": sum(1 for r in applied if not r["processed_exact"] and normalize_plate(r["pre_correction_text"]) == r["ground_truth"]),
    }
    by_status: dict[str, dict[str, int]] = {}
    for r in attempted:
        s = by_status.setdefault(r["registration_status"] or "n/a", {"n": 0, "processed_exact": 0})
        s["n"] += 1
        s["processed_exact"] += int(r["processed_exact"])
    m["by_registration_status"] = by_status
    return m


def compare(baseline: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired comparison on the SAME items (rows are index-aligned by construction)."""
    assert [r["image"] for r in baseline] == [r["image"] for r in other], "pipelines must cover identical items"
    out = {}
    for key in ("raw_exact", "processed_exact"):
        b, o = [r[key] for r in baseline], [r[key] for r in other]
        out[key] = {"both_correct": sum(x and y for x, y in zip(b, o)),
                    "fixed": sum((not x) and y for x, y in zip(b, o)),
                    "regressed": sum(x and (not y) for x, y in zip(b, o)),
                    "both_wrong": sum((not x) and (not y) for x, y in zip(b, o))}
    return out


# ------------------------------------------------------------------ reporting
def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def format_summary(metrics: dict[str, dict[str, Any]], comparisons: dict[str, dict[str, Any]],
                   header: dict[str, Any]) -> str:
    L = ["=" * 78, "VisionForge AI OCR evaluation", "=" * 78]
    for k, v in header.items():
        L.append(f"{k:<26}{v}")
    L.append("-" * 78)
    for name, m in metrics.items():
        L += [f"[{name}]  items (denominator) = {m['n']}",
              f"  statuses                 {m['statuses']}",
              (f"  plate found (any box)    {m['detected']}/{m['readable_images']} readable "
               f"= {_pct(m['detection_rate_of_readable'])}   (box correctness is NOT measured)"
               if m["detection_rate_of_readable"] is not None else
               "  plate detection          not applicable (crops mode: the images are the plates)"),
              f"  end-to-end exact  RAW    {m['raw_exact_matches']}/{m['n']} = {_pct(m['raw_exact_rate'])}",
              f"  end-to-end exact  PROC   {m['processed_exact_matches']}/{m['n']} = {_pct(m['processed_exact_rate'])}",
              f"  exact given OCR ran      raw {_pct(m['raw_exact_rate_given_ocr'])}  processed {_pct(m['processed_exact_rate_given_ocr'])}"
              f"   (n={m['ocr_attempted']})",
              f"  CER (corpus)             raw {_pct(m['raw_cer'])}  processed {_pct(m['processed_cer'])}",
              f"  failed / empty OCR       {m['failed']} / {m['empty_ocr']}",
              f"  latency ms (median/p95)  detect {m['latency_detect_ms']['median']}/{m['latency_detect_ms']['p95']}  "
              f"ocr {m['latency_ocr_ms']['median']}/{m['latency_ocr_ms']['p95']}  "
              f"total {m['latency_total_ms']['median']}/{m['latency_total_ms']['p95']}",
              f"  auto-corrections         {m['corrections']}",
              f"  by registration status   {m['by_registration_status']}", ""]
    for name, c in comparisons.items():
        L.append(f"paired vs baseline [{name}]  processed exact: {c['processed_exact']}   raw exact: {c['raw_exact']}")
    L.append("=" * 78)
    return "\n".join(L)


ROW_FIELDS = list(_blank_row(Item("", None, "X"), "").keys())


def write_reports(out_dir: Path, rows_by_pipeline: dict[str, list[dict[str, Any]]],
                  metrics: dict, comparisons: dict, manifest: dict) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / "ocr_evaluation_report.csv", out_dir / "ocr_evaluation_metrics.json"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for rows in rows_by_pipeline.values():
            w.writerows(rows)
    json_path.write_text(json.dumps({"manifest": manifest, "metrics": metrics, "paired_vs_baseline": comparisons},
                                    indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


def parse_kv(text: str) -> dict[str, Any]:
    """'decoder=beamsearch,beamWidth=5,mag_ratio=1.5' -> typed dict."""
    out: dict[str, Any] = {}
    for part in filter(None, (text or "").split(",")):
        k, _, v = part.partition("=")
        for cast in (int, float):
            try:
                out[k.strip()] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k.strip()] = {"true": True, "false": False}.get(v.strip().lower(), v.strip())
    return out


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the VisionForge plate OCR pipeline(s).",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images_dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True, help="columns: image_filename,ground_truth")
    ap.add_argument("--model_path", type=Path, default=None, help="plate-trained YOLO weights (.pt)")
    ap.add_argument("--mode", choices=["auto", "e2e", "crops"], default="auto",
                    help="e2e = detect then OCR; crops = images are already plate crops")
    ap.add_argument("--pipelines", default=",".join(DEFAULT_PIPELINES),
                    help=f"comma list or 'all'. Available: {', '.join(lps.PIPELINES)}. First is the baseline.")
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--ocr_backend", choices=["easyocr", "paddleocr"], default="easyocr")
    ap.add_argument("--readtext_opts", default="", help="EasyOCR readtext options, e.g. decoder=beamsearch,mag_ratio=1.5")
    ap.add_argument("--cache_ocr", action="store_true", help="share engine output between pipelines (faster; latency not valid)")
    ap.add_argument("--save_crops", action="store_true", help="write the crops given to OCR, to inspect the detector")
    ap.add_argument("--output_dir", type=Path, default=Path("ocr_eval_results"))
    ap.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")

    names = list(lps.PIPELINES) if args.pipelines == "all" else [n.strip() for n in args.pipelines.split(",") if n.strip()]
    unknown = [n for n in names if n not in lps.PIPELINES]
    if unknown or not names:
        print(f"Unknown pipeline(s): {unknown}. Available: {list(lps.PIPELINES)}")
        return 1
    pipelines = {n: lps.PIPELINES[n] for n in names}

    labels, blank, dup = load_ground_truth_detailed(args.csv)
    if not labels:
        print("The CSV has no labelled rows, so accuracy cannot be measured. Provide verified ground truth.")
        return 1
    if not args.images_dir.is_dir():
        print(f"Images folder not found: {args.images_dir}")
        return 1
    items, unlabelled = collect_items(args.images_dir, labels)
    if blank or dup:
        log.warning("CSV rows without ground truth (not scorable): %s; duplicate rows ignored: %s", blank, dup)

    # OCR backend: refuse to evaluate with a placeholder
    opts = parse_kv(args.readtext_opts)
    backend: Any = (lps.PaddleOCRBackend(lang="en") if args.ocr_backend == "paddleocr"
                    else lps.EasyOCRBackend(("en",), gpu=False, readtext_kwargs=opts))
    try:
        backend.read(np.full((48, 160, 3), 255, np.uint8))  # probe: loads models, fails loudly if unavailable
    except lps.OCRUnavailableError as exc:
        print(f"OCR backend '{backend.name}' is unavailable, so nothing was evaluated.\n  {exc}")
        return 2
    ocr: Any = CachingOCR(backend) if args.cache_ocr else backend

    mode = args.mode
    model_path = args.model_path or helpers.find_weights(Path.cwd(), "license_plate_detector.pt")
    if mode == "auto":
        mode = "e2e" if model_path else "crops"
        if mode == "crops":
            log.warning("No plate model given; assuming the images are plate crops (--mode crops).")
    if mode == "e2e":
        if not model_path or not Path(model_path).is_file():
            print("--mode e2e needs a plate model: pass --model_path. (Use --mode crops if the images are already crops.)")
            return 1
        service = lps.LicensePlateService.from_weights(str(model_path), ocr=ocr)
    else:
        service = lps.LicensePlateService.__new__(lps.LicensePlateService)
        service.model, service.ocr = None, ocr
        service.class_names, service.class_ids, service.model_warnings = {}, [], []
        service.read_options = lps.DEFAULT_READ_OPTIONS

    log.info("Evaluating %d labelled items, mode=%s, pipelines=%s", len(items), mode, names)
    rows = evaluate_dataset(items, service, pipelines, mode, args.det_conf,
                            args.output_dir / "crops" if args.save_crops else None)
    metrics = {n: summarize(r, mode) for n, r in rows.items()}
    base = names[0]
    comparisons = {n: compare(rows[base], rows[n]) for n in names[1:]}
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"), "mode": mode, "pipelines": names,
        "baseline": base, "labelled_items": len(items), "missing_images": sum(i.path is None for i in items),
        "unlabelled_images_excluded": unlabelled, "csv_rows_without_truth": blank, "csv_duplicates_ignored": dup,
        "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(), "detection_confidence": args.det_conf,
        "ocr_backend": backend.name, "readtext_opts": opts, "ocr_cache": bool(args.cache_ocr),
        "plate_model": Path(model_path).name if (mode == "e2e" and model_path) else None,
        "opencv": cv2.__version__,
    }
    header = {"mode": mode, "labelled items": len(items), "missing image files": manifest["missing_images"],
              "images without a label": f"{len(unlabelled)} (excluded: cannot be scored)",
              "OCR backend": backend.name, "baseline pipeline": base,
              "latency note": "INVALID (--cache_ocr)" if args.cache_ocr else "measured per pipeline"}
    print(format_summary(metrics, comparisons, header))
    csv_path, json_path = write_reports(args.output_dir, rows, metrics, comparisons, manifest)
    print(f"\nPer-image results: {csv_path}\nMetrics + manifest: {json_path}")
    if any(m["statuses"].get("ocr_unavailable") for m in metrics.values()):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())