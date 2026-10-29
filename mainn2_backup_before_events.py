"""VisionForge AI — Visual Intelligence Platform (Streamlit entry point).

Run from the project root:   streamlit run mainn2.py
"""
from __future__ import annotations

import inspect
import io
import logging
import os
import sys
import traceback
import uuid
from importlib import metadata
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
# This file lives in the project root (next to core/). If it is moved into app/, use the parent.
PROJECT_ROOT = APP_DIR if (APP_DIR / "core").is_dir() or not (APP_DIR.parent / "core").is_dir() else APP_DIR.parent
for _p in (str(PROJECT_ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

st.set_page_config(
    page_title="VisionForge AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import components as ui  # noqa: E402
import helpers  # noqa: E402
import video_service as vs  # noqa: E402
from modules import (  # noqa: E402
    APP_PHASE, APP_VERSION, IMPLEMENTED, MODULES, PAGES, PARTIAL, PLANNED,
)
import license_plate_service as lps  # noqa: E402
from core.storage.history_store import HistoryStore  # noqa: E402

try:
    from core.detection.detection_engine import DetectionEngine
    ENGINE_IMPORT_ERROR: str | None = None
except Exception as exc:  # shown in the UI rather than crashing the app
    DetectionEngine = None  # type: ignore[assignment]
    ENGINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("visionforge")
DEBUG = os.environ.get("VISIONFORGE_DEBUG", "").strip() == "1"
HISTORY_DB_PATH = PROJECT_ROOT / "outputs" / "visionforge_history.sqlite3"
try:
    HISTORY_STORE: HistoryStore | None = HistoryStore(HISTORY_DB_PATH)
    HISTORY_STORE_ERROR: str | None = None
except Exception as exc:  # keep inference usable if local storage is unavailable
    HISTORY_STORE = None
    HISTORY_STORE_ERROR = f"{type(exc).__name__}: {exc}"
    log.exception("Persistent history could not be initialized")


def persist_history_record(record: dict) -> None:
    """Save one successful analysis without allowing storage failure to break inference."""
    if HISTORY_STORE is None:
        log.warning("Analysis was not persisted because history storage is unavailable: %s", HISTORY_STORE_ERROR)
        return
    try:
        HISTORY_STORE.save(record)
    except Exception:
        log.exception("Could not persist analysis history record")


def show_error(message: str, exc: BaseException | None = None) -> None:
    """Friendly error for users; full traceback goes to the log (and the page only in debug mode)."""
    st.error(message)
    log.error(message, exc_info=exc)
    if DEBUG and exc is not None:
        with st.expander("Technical details (debug mode)"):
            st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


MODEL_NAME = "yolo11n.pt"
VIDEO_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "videos"
MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024

def _stretch_kwargs() -> dict:
    """Newer Streamlit uses width="stretch"; older uses use_container_width=True.

    Detected from the installed widget signatures rather than a version guess.
    """
    widgets = (st.button, st.download_button, st.image, st.dataframe, st.plotly_chart)
    for fn in widgets:
        param = inspect.signature(fn).parameters.get("width")
        if param is None or "stretch" not in str(param.annotation):
            return {"use_container_width": True}
    return {"width": "stretch"}


STRETCH = _stretch_kwargs()


def ordinal_every(n: int) -> str:
    if n == 1:
        return "every frame"
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"every {n}{suffix} frame"


def model_reference() -> str:
    """Prefer local weights (resolved from the project root); else the bare name."""
    weights = helpers.find_weights(PROJECT_ROOT, MODEL_NAME)
    return str(weights) if weights else MODEL_NAME


@st.cache_resource(show_spinner=False)
def _load_engine(model_ref: str):
    return DetectionEngine(model_path=model_ref)


def get_engine():
    """Single shared engine for images and video. Raises with a clear message."""
    if DetectionEngine is None:
        raise RuntimeError(f"Detection engine could not be imported. {ENGINE_IMPORT_ERROR}")
    try:
        engine = _load_engine(model_reference())
    except Exception as exc:
        st.session_state.engine_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"The YOLO model failed to load: {exc}") from exc
    st.session_state.engine_loaded = True
    st.session_state.engine_error = None
    return engine


def engine_device() -> str | None:
    if not st.session_state.engine_loaded:
        return None
    try:
        return str(get_engine().model.device)
    except Exception:  # device is informational only
        return None


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


# ---------------------------------------------------------------- state
# Initialise everything before any widget is created.
_defaults = {
    "history": [], "latest": None, "video_input": None, "video_result": None,
    "engine_loaded": False, "engine_error": None,
    "image_upload_n": 0, "video_upload_n": 0,
    "plate_latest": None, "plate_upload_n": 0, "plate_loaded": False, "plate_error": None,
}
for _k, _v0 in _defaults.items():
    st.session_state.setdefault(_k, _v0)
# Hydrate session history once per browser session from the persistent database.
# Session state remains the fast working copy used by the existing UI.
if not st.session_state.get("history_initialized", False):
    if HISTORY_STORE is not None:
        try:
            st.session_state.history = HISTORY_STORE.load_all()
        except Exception:
            log.exception("Could not load persistent analysis history")
    st.session_state.history = helpers.normalize_history(st.session_state.history)
    st.session_state.history_initialized = True
if "navigation_target" in st.session_state:
    st.session_state["page_navigation"] = st.session_state.pop("navigation_target")
_qp = getattr(st, "query_params", None)
if "page_navigation" not in st.session_state:
    _wanted = _qp.get("page") if _qp is not None else None
    st.session_state["page_navigation"] = _wanted if _wanted in PAGES else PAGES[0]
if st.session_state["page_navigation"] not in PAGES:  # stale browser/session state
    st.session_state["page_navigation"] = PAGES[0]


def go_to(page: str) -> None:
    st.session_state["navigation_target"] = page


# ---------------------------------------------------------------- shell
# ---------------------------------------------------------------- readiness helpers (used by the sidebar)
def plate_model_path() -> Path:
    typed = str(st.session_state.get("plate_model_path_input", "")).strip().strip('"')
    return Path(typed).expanduser() if typed else lps.default_plate_model_path(PROJECT_ROOT)


def plate_readiness() -> tuple[str, str, list[str]]:
    """(state label, tone, detail lines) for the plate detector. Never claims more than is verified."""
    path = plate_model_path()
    fs = lps.plate_model_file_status(path)
    if st.session_state.plate_error:
        return "Load failed", "warn", [st.session_state.plate_error]
    if st.session_state.plate_loaded:
        return "Loaded", "ok", [f"{path.name} (classes verified as plate classes)"]
    if fs["state"] == "found":
        return "Weights found", "idle", [f"{path.name}, {fs['size_mb']} MB. Not loaded yet; class names are checked on first run."]
    return "No weights", "warn", [f"Expected at {path}", "Detection is disabled until plate-trained weights are added."]


def ocr_readiness() -> tuple[str, str, list[str]]:
    info = lps.ocr_dependency_status()
    if not info["installed"]:
        return "Not installed", "warn", ["Run: pip install easyocr", "Plate detection still works without OCR."]
    lines = [f"EasyOCR {info['version'] or ''}".strip()]
    if info["weights_cached"]:
        return "Ready", "ok", lines + ["Model files are cached locally."]
    return "Installed", "idle", lines + ["Model files are not cached yet; the first OCR run downloads them (internet needed, roughly 100 MB)."]


theme_info = ui.inject_theme()

with st.sidebar:
    ui.sidebar_brand()
    page = st.radio("Navigation", PAGES, key="page_navigation", label_visibility="collapsed")
    if ENGINE_IMPORT_ERROR:
        model_row = ("General model", "Import failed", "warn")
    elif st.session_state.engine_error:
        model_row = ("General model", "Load failed", "warn")
    elif st.session_state.engine_loaded:
        model_row = ("General model", f"Loaded · {MODEL_NAME}", "ok")
    else:
        model_row = ("General model", "Not loaded yet", "idle")
    device = engine_device()
    ffmpeg = vs.find_ffmpeg()
    _pstate, _ptone, _ = plate_readiness()
    _ocr = lps.ocr_dependency_status()
    rows = [
        model_row,
        ("Compute", device if device else "Unknown until model loads", "ok" if device else "idle"),
        ("Plate model", _pstate, _ptone),
        ("Plate OCR", "EasyOCR installed" if _ocr["installed"] else "Not installed",
         "ok" if _ocr["installed"] else "warn"),
        ("Video playback", "H.264 re-encode ready" if ffmpeg else "FFmpeg not found",
         "ok" if ffmpeg else "warn"),
    ]
    if theme_info["stale_sidebar_rule"]:
        rows.append(("Stylesheet", "Old neural.css (guarded)", "warn"))
    ui.sidebar_status(rows)
    ui.sidebar_footer(f"VisionForge AI v{APP_VERSION}\n{APP_PHASE}\nRunning: {Path(__file__).name}")

if _qp is not None:
    try:
        if _qp.get("page") != page:
            _qp["page"] = page
    except Exception:  # query params are a convenience only
        log.debug("Could not update query params", exc_info=True)

# ---------------------------------------------------------------- charts
CLASS_COLOR, CONF_COLOR = "#3fd8ee", "#4c8dff"
KIND_COLORS = {"image": "#4c8dff", "video": "#3fd8ee", "plate": "#a78bfa"}


def style_figure(fig, height: int = 320, legend: bool = False):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9aa9bd", size=12), margin=dict(l=8, r=8, t=48, b=8),
        showlegend=legend, title=dict(font=dict(size=14, color="#e8eef7")),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(140,170,210,.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(140,170,210,.10)", zeroline=False)
    return fig


def chart_classes(labels: pd.Series, title: str):
    import plotly.express as px
    dist = labels.value_counts().rename_axis("Class").reset_index(name="Detections")
    fig = px.bar(dist, x="Class", y="Detections", title=title)
    fig.update_traces(marker_color=CLASS_COLOR, hovertemplate="%{x}: %{y} detections<extra></extra>")
    return style_figure(fig)


def chart_confidence(conf: pd.Series, title: str):
    import plotly.express as px
    fig = px.histogram(pd.DataFrame({"Confidence": conf}), x="Confidence", nbins=10,
                       range_x=[0, 1], title=title)
    fig.update_traces(marker_color=CONF_COLOR,
                      hovertemplate="Confidence %{x}: %{y} detections<extra></extra>")
    fig.update_yaxes(title="Detections")
    return style_figure(fig)


def show_chart(fig) -> None:
    st.plotly_chart(fig, **STRETCH)


# ---------------------------------------------------------------- Command Center
def page_command_center() -> None:
    # Normalize history at the point of consumption: drop non-dict legacy
    # entries and guarantee every record is a valid dict.
    history = helpers.normalize_history(st.session_state.history)
    ui.page_title("Workspace · Overview", "Command Center",
                  "Live status of this session. Every figure comes from analyses you ran "
                  "in this browser session; nothing is estimated.")
    images = [r for r in history if r.get("kind") in ("image", "plate")]
    videos = [r for r in history if r.get("kind") == "video"]
    plates = [r for r in history if r.get("kind") == "plate"]

    def _safe_objects_count(record_list: list[dict[str, Any]]) -> int:
        total = 0
        for r in record_list:
            objs = r.get("objects")
            if isinstance(objs, int):
                total += objs
        return total

    c = st.columns(6)
    with c[0]:
        ui.stat("Images analysed", str(len(images)), "this session")
    with c[1]:
        ui.stat("Videos analysed", str(len(videos)), "this session")
    with c[2]:
        ui.stat("Objects detected", str(_safe_objects_count(images)),
                "general model, images" if images else "no image detections yet")
    with c[3]:
        ui.stat("Plates detected", str(sum(r["objects"] for r in plates)),
                "plate model" if plates else "none yet")
    with c[4]:
        if history:
            last = history[-1]
            last_kind = last.get("kind", "unknown")
            last_file = last.get("filename", "unknown")
            ui.stat("Latest processing", helpers.processing_label(last), f"{last_kind} · {last_file}")
        else:
            ui.stat("Latest processing", "—", "unavailable until you run an analysis")
    with c[5]:
        if st.session_state.engine_error or ENGINE_IMPORT_ERROR:
            ui.stat("General model", "ERROR", "see Platform Overview")
        elif st.session_state.engine_loaded:
            ui.stat("General model", "LOADED", MODEL_NAME)
        else:
            ui.stat("General model", "NOT LOADED", "loads on first analysis")

    ui.label("Start an analysis" if not history else "Quick actions")
    a, b, d = st.columns(3, gap="large")
    with a:
        ui.info_panel("Image detection", "Run YOLO on one image and inspect the annotated result.")
        st.button("Open Vision Studio", key="qa_image", type="primary",
                  on_click=go_to, args=("Vision Studio",), **STRETCH)
    with b:
        ui.info_panel("License plates", "Detect plates with a plate-trained model and read them with OCR.")
        st.button("Open License Plate Studio", key="qa_plate", type="primary",
                  on_click=go_to, args=("License Plate Studio",), **STRETCH)
    with d:
        ui.info_panel("Video analysis", "Detect across video frames; get an annotated video and CSV.")
        st.button("Open Video Intelligence", key="qa_video", type="primary",
                  on_click=go_to, args=("Video Intelligence",), **STRETCH)

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        ui.label("Recent activity")
        if not history:
            ui.empty_state("◌", "A fresh session",
                           "No images, plates or videos have been analysed yet. Choose an action above to begin.")
        else:
            rows = []
            for r in history[::-1][:10]:
                rows.append({
                    "Type": r.get("kind", "unknown").title(),
                    "Source": r.get("filename", "unknown"),
                    "Detections": r.get("objects", 0),
                    "Confidence ≥": r.get("threshold", "—"),
                    "Processing": helpers.processing_label(r),
                    "Completed": r.get("completed_at", ""),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    with right:
        ui.label("Readiness")
        weights = helpers.find_weights(PROJECT_ROOT, MODEL_NAME)
        if ENGINE_IMPORT_ERROR:
            ui.status_card("General detector", "Import failed", "warn", [ENGINE_IMPORT_ERROR])
        elif st.session_state.engine_error:
            ui.status_card("General detector", "Load failed", "warn", [st.session_state.engine_error])
        elif st.session_state.engine_loaded:
            ui.status_card("General detector", "Loaded", "ok", [f"{MODEL_NAME}, shared by images and video."])
        else:
            ui.status_card("General detector", "Not loaded", "idle",
                           [f"Local weights found: {weights.name}." if weights else
                            f"No local {MODEL_NAME}; Ultralytics may try to download it on first run."])
        pstate, ptone, plines = plate_readiness()
        ui.status_card("Plate detector", pstate, ptone, plines)
        ostate, otone, olines = ocr_readiness()
        ui.status_card("Plate OCR", ostate, otone, olines)


# ---------------------------------------------------------------- Vision Studio
def reset_image() -> None:
    st.session_state.latest = None
    st.session_state.image_upload_n += 1


def run_image_detection(image_bgr, filename: str, image_hash: str, threshold: float) -> None:
    try:
        with st.status("Running detection…", expanded=True) as status:
            st.write("Loading model (first run only)…")
            engine = get_engine()
            st.write("Running inference…")
            with helpers.temporary_image_file(image_bgr) as path:
                result = engine.detect_image(image_path=str(path), confidence=threshold)
            st.write("Preparing results…")
            png = helpers.encode_png(result["annotated_image"])
            status.update(label="Detection complete", state="complete", expanded=False)
        st.session_state.latest = {"result": result, "png": png, "filename": filename,
                                   "hash": image_hash, "threshold": threshold}
        st.session_state.history.append(
            helpers.stamp(helpers.summarize_analysis(filename, result, threshold))
        )
        st.session_state.history = helpers.normalize_history(st.session_state.history)
        persist_history_record(st.session_state.history[-1])
    except Exception as exc:  # display, never fake results
        st.session_state.latest = None
        show_error(f"Detection failed: {exc}", exc)


def page_vision_studio() -> None:
    ui.page_title("Workspace · Images", "Vision Studio",
                  "Upload an image, choose a confidence threshold and run YOLO detection.")
    rail, stage = st.columns([1, 3.2], gap="large")
    image_bgr = image_hash = upload = None
    with rail:
        ui.label("Input")
        upload = st.file_uploader("Image", type=helpers.ALLOWED_TYPES,
                                  key=f"image_upload_{st.session_state.image_upload_n}",
                                  label_visibility="collapsed")
        if upload is not None:
            data = upload.getvalue()
            image_bgr = helpers.decode_image(data)
            image_hash = helpers.file_hash(data)
            if image_bgr is None:
                st.error("This file could not be decoded as an image. Upload a valid JPG, PNG or WebP.")
        ui.label("Parameters")
        threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                              key="studio_confidence",
                              help="Detections below this confidence are discarded.")
        run = st.button("Run detection", type="primary", key="studio_run",
                        disabled=image_bgr is None, **STRETCH)
        st.button("Reset", key="studio_reset", on_click=reset_image, **STRETCH)
        if run and image_bgr is not None:
            run_image_detection(image_bgr, upload.name, image_hash, threshold)

        latest = st.session_state.latest
        current = latest if (latest and image_hash and latest["hash"] == image_hash) else None
        if current:
            ui.label("Export")
            frame = helpers.detections_to_frame(current["result"].get("detections"))
            if current["png"]:
                st.download_button("Annotated image (PNG)", current["png"],
                                   "visionforge_annotated.png", "image/png", key="dl_png", **STRETCH)
            st.download_button("Detection table (CSV)", frame.to_csv(index=False),
                               "visionforge_detections.csv", "text/csv",
                               key="dl_csv", disabled=frame.empty, **STRETCH)

    with stage:
        if image_bgr is None:
            ui.viewport_bar("Viewport", "No input")
            ui.empty_state("◈", "Load an image to begin",
                           "Drag and drop a JPG, PNG or WebP into the input panel. "
                           "Detection only runs when you press Run detection.")
            return
        h, w = image_bgr.shape[:2]
        if not current:
            ui.viewport_bar(f"Input · {upload.name}", f"{w} × {h}px")
            st.image(helpers.to_rgb(image_bgr), **STRETCH)
            st.caption("A result for a different image is hidden. Run detection for this one."
                       if latest else "Ready. Adjust the threshold, then press Run detection.")
            return
        result = current["result"]
        frame = helpers.detections_to_frame(result.get("detections"))
        m = st.columns(4)
        with m[0]:
            ui.stat("Objects", str(result.get("object_count", len(frame))))
        with m[1]:
            ui.stat("Inference", f"{result['inference_time_ms']:.1f} ms")
        with m[2]:
            ui.stat("Resolution", f"{result['image_width']} × {result['image_height']}")
        with m[3]:
            ui.stat("Threshold used", f"{current['threshold']:.2f}")
        if abs(current["threshold"] - threshold) > 1e-9:
            st.caption(f"Slider is at {threshold:.2f}, but this result used {current['threshold']:.2f}. "
                       "Run detection again to apply the new value.")
        tab_a, tab_s, tab_o = st.tabs(["Annotated", "Side by side", "Original"])
        annotated_rgb = helpers.to_rgb(result["annotated_image"])
        with tab_a:
            ui.viewport_bar("YOLO output", f"{len(frame)} detections")
            st.image(annotated_rgb, **STRETCH)
        with tab_s:
            c1, c2 = st.columns(2)
            with c1:
                ui.viewport_bar("Original", upload.name)
                st.image(helpers.to_rgb(image_bgr), **STRETCH)
            with c2:
                ui.viewport_bar("Annotated", f"{len(frame)} detections")
                st.image(annotated_rgb, **STRETCH)
        with tab_o:
            ui.viewport_bar("Source", upload.name)
            st.image(helpers.to_rgb(image_bgr), **STRETCH)
        ui.label("Detections")
        if frame.empty:
            ui.empty_state("◌", "No objects detected",
                           f"Nothing met the {current['threshold']:.2f} confidence threshold. "
                           "Try a lower threshold or a different image.")
        else:
            t_col, c_col = st.columns(2, gap="large")
            with t_col:
                st.dataframe(frame, hide_index=True, **STRETCH,
                             column_config={"Confidence": st.column_config.ProgressColumn(
                                 min_value=0.0, max_value=1.0, format="%.2f")})
            with c_col:
                show_chart(chart_classes(frame["Object"], "Class distribution"))


# ---------------------------------------------------------------- Video Intelligence
def _drop_video_result() -> None:
    res = st.session_state.video_result
    if res:
        vs.remove_tree(res["result"].get("run_dir"))
    st.session_state.video_result = None


def _drop_video_input() -> None:
    inp = st.session_state.video_input
    if inp:
        vs.remove_tree(Path(inp["path"]).parent)
    st.session_state.video_input = None


def reset_video() -> None:
    _drop_video_result()
    _drop_video_input()
    st.session_state.video_upload_n += 1


def _read_if_small(path: str) -> tuple[bytes | None, str | None]:
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return None, "The output file is missing or empty."
    if p.stat().st_size > MAX_DOWNLOAD_BYTES:
        return None, f"The file is {helpers.format_bytes(p.stat().st_size)}, too large to serve through the browser. Find it at: {p}"
    return p.read_bytes(), None


def run_video_analysis(name: str, vhash: str, settings: dict) -> None:
    """Executes inference ONLY when called from the Analyze button."""
    inp = st.session_state.video_input
    try:
        with st.status("Analysing video…", expanded=True) as status:
            st.write("Loading model (first run only)…")
            engine = get_engine()
            st.write(f"Detecting on {ordinal_every(settings['stride'])}…")
            bar = st.progress(0.0, text="Starting…")
            state = {"last": 0}

            def on_progress(read: int, total: int | None) -> None:
                if read - state["last"] < 3 and read != total:
                    return
                state["last"] = read
                if total:
                    bar.progress(min(read / total, 1.0), text=f"Read {read} / {total} source frames")
                else:
                    bar.progress(0.0, text=f"Read {read} source frames (total unknown)")

            result = vs.analyze_video(
                engine, inp["path"], VIDEO_OUTPUT_ROOT, settings["confidence"],
                settings["stride"], settings["max_frames"], progress_callback=on_progress)
            bar.progress(1.0, text="Processing complete")
            st.write("Preparing downloads…")
            video_path = result["playback_path"] or result["output_video_path"]
            video_bytes, video_err = _read_if_small(video_path)
            csv_bytes, csv_err = _read_if_small(result["output_csv_path"])
            if csv_bytes is not None:
                try:
                    pd.read_csv(io.BytesIO(csv_bytes))
                except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                    csv_bytes, csv_err = None, f"The CSV report could not be parsed: {exc}"
            status.update(label="Analysis complete", state="complete", expanded=False)
        _drop_video_result()
        st.session_state.video_result = {
            "result": result, "name": name, "hash": vhash, "settings": settings,
            "meta": inp["meta"], "video_bytes": video_bytes, "video_err": video_err,
            "csv_bytes": csv_bytes, "csv_err": csv_err,
            "download_is_h264": result["playback_path"] is not None,
        }
        st.session_state.history.append(
            helpers.stamp(helpers.summarize_video(
                name, result, settings["confidence"], settings["stride"], settings["max_frames"])
            )
        )
        st.session_state.history = helpers.normalize_history(st.session_state.history)
        persist_history_record(st.session_state.history[-1])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        show_error(f"Video analysis failed: {exc}", exc)
    except Exception as exc:  # unexpected: show the full traceback, do not hide it
        show_error(f"Unexpected error during video analysis: {type(exc).__name__}: {exc}", exc)


def page_video() -> None:
    ui.page_title("Workspace · Video", "Video Intelligence",
                  "Run YOLO detection across video frames, review the annotated output and "
                  "export a frame-level CSV report.")
    up_col, cfg_col = st.columns([1.15, 1], gap="large")

    upload = None
    with up_col:
        ui.label("1 · Upload")
        upload = st.file_uploader(
            "Video", type=vs.VIDEO_TYPES, key=f"video_upload_{st.session_state.video_upload_n}",
            label_visibility="collapsed",
            help=f"Accepted: {', '.join(vs.VIDEO_TYPES)}. Max {vs.MAX_VIDEO_MB} MB. "
                 "Decoding depends on your OpenCV build; unreadable files are reported.")
        if upload is None:
            if st.session_state.video_input:  # user cleared the uploader
                _drop_video_input()
        else:
            upload_key = (upload.name, upload.size, getattr(upload, "file_id", None))
            inp = st.session_state.video_input
            if inp is None or inp["key"] != upload_key:
                _drop_video_input()
                data = upload.getbuffer()  # memoryview: no extra copy
                vhash = helpers.file_hash(data)
                try:
                    path = vs.save_upload(data, upload.name)
                    try:
                        meta = vs.probe_video(path)
                    except vs.VideoValidationError:
                        vs.remove_tree(path.parent)
                        raise
                    st.session_state.video_input = {"key": upload_key, "hash": vhash,
                                                    "name": upload.name, "path": str(path),
                                                    "meta": meta}
                except vs.VideoValidationError as exc:
                    st.error(str(exc))

    inp = st.session_state.video_input
    meta = inp["meta"] if inp else None

    with up_col:
        if inp:
            st.caption(f"{inp['name']} · {helpers.format_bytes(meta.size_bytes)}")
            d = st.columns(4)
            with d[0]:
                ui.stat("Resolution", f"{meta.width} × {meta.height}")
            with d[1]:
                ui.stat("FPS", f"{meta.fps:.1f}" if meta.fps else "unknown")
            with d[2]:
                ui.stat("Frames", str(meta.frame_count) if meta.frame_count else "unknown")
            with d[3]:
                ui.stat("Duration", helpers.format_seconds(meta.duration_seconds))
        else:
            ui.empty_state("▶", "No video loaded",
                           "Drag and drop a video above. It is validated and probed before any analysis.")

    with cfg_col:
        ui.label("2 · Configure")
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="video_conf",
                               help="Detections below this score are discarded.")
        stride = st.number_input("Frame stride (N)", 1, 120, 1, 1, key="video_stride",
                                 help="Detection runs on every Nth source frame (frames 0, N, 2N, …). "
                                      "Skipped frames stay in the output video, unannotated. "
                                      "Higher N is faster but may miss short-lived objects.")
        max_frames_in = st.number_input("Max source frames (0 = whole video)", 0, 1_000_000, 0, 50,
                                        key="video_max_frames",
                                        help="Stop after reading this many source frames from the start.")
        max_frames = int(max_frames_in) or None
        if meta:
            est = vs.estimate_frames(meta, int(stride), max_frames)
            if est:
                st.caption(f"Will read {est[0]} source frames and run detection on {est[1]} of them.")
            else:
                st.caption("Frame count is unknown for this file, so no estimate is available.")
        b1, b2 = st.columns([2, 1])
        with b1:
            analyze = st.button("Analyze video", type="primary", key="video_analyze",
                                disabled=inp is None, **STRETCH)
        with b2:
            st.button("Reset", key="video_reset", on_click=reset_video, **STRETCH)

    if analyze and inp:
        ui.label("3 · Processing")
        run_video_analysis(inp["name"], inp["hash"],
                           {"confidence": float(confidence), "stride": int(stride),
                            "max_frames": max_frames})

    res = st.session_state.video_result
    if not res:
        return
    if inp and res["hash"] != inp["hash"]:
        st.info(f"The stored result belongs to '{res['name']}', not the video currently loaded. "
                "Analyze the loaded video to replace it, or press Reset.")
        return
    render_video_results(res)


def render_video_results(res: dict) -> None:
    r, s = res["result"], res["settings"]
    run_dir = Path(r["run_dir"])

    # Build the display frame directly from processor records so track_id is
    # retained even if the general-purpose helper has an older column schema.
    video_records = r.get("detections") or []
    frame_columns = [
        "frame", "timestamp_seconds", "track_id", "class_id", "label",
        "confidence", "x1", "y1", "x2", "y2",
    ]
    frame = pd.DataFrame.from_records(video_records)
    for column in frame_columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    frame = frame[frame_columns]
    if not frame.empty:
        frame["frame"] = pd.to_numeric(frame["frame"], errors="coerce").fillna(0).astype(int)
        frame["timestamp_seconds"] = pd.to_numeric(
            frame["timestamp_seconds"], errors="coerce"
        ).fillna(0.0)
        frame["track_id"] = pd.to_numeric(frame["track_id"], errors="coerce").astype("Int64")
        frame["class_id"] = pd.to_numeric(frame["class_id"], errors="coerce")
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
        for column in ("x1", "y1", "x2", "y2"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["label"] = frame["label"].fillna("unknown").astype(str)
        frame = frame.sort_values(["frame", "timestamp_seconds"], kind="stable").reset_index(drop=True)

    ui.label("Annotated video")
    if not run_dir.is_dir():
        st.error("The output files for this analysis no longer exist on disk. Analyze the video again.")
        return

    player, side = st.columns([2.2, 1], gap="large")
    with player:
        playback = r.get("playback_path")
        if playback and Path(playback).is_file():
            ui.viewport_bar("Annotated output", "H.264 · browser playback")
            st.video(playback)
        else:
            st.warning(r.get("playback_note", "Browser playback is unavailable."))
            n = max(int(r.get("frames_read", 0)) - 1, 0)
            if n > 0:
                idx = st.slider("Preview annotated frame", 0, n, 0, key="video_frame_preview")
            else:
                idx = 0
            img = vs.read_frame(r["output_video_path"], idx)
            if img is None:
                st.error("Could not read a frame from the output video.")
            else:
                ui.viewport_bar(f"Annotated frame {idx}", "still preview (not a player)")
                st.image(helpers.to_rgb(img), **STRETCH)

    tracking_enabled = bool(r.get("tracking_enabled", False))
    unique_tracks = int(r.get("unique_tracks", 0) or 0)
    with side:
        if tracking_enabled:
            tracking_message = (
                f"Tracking is enabled. The tracker produced {unique_tracks} unique track IDs "
                "in the processed frames. Track IDs may change when an object is lost or occluded."
            )
        else:
            tracking_message = (
                "Tracking IDs were not available for this run. Detection boxes may still be present."
            )
        ui.info_panel(
            "Object tracking",
            f"Detection ran on {ordinal_every(s['stride'])}; skipped frames remain unannotated. "
            "Total detections count bounding boxes, while unique tracks count distinct tracker IDs. "
            + tracking_message,
        )
        if playback and Path(playback).is_file():
            st.caption(r.get("playback_note", "Browser-playable output is available."))

    ui.label("Summary")
    m = st.columns(7)
    with m[0]:
        ui.stat("Frames read", str(r["frames_read"]), "source frames decoded")
    with m[1]:
        ui.stat("Frames processed", str(r["frames_processed"]), "detection actually ran")
    with m[2]:
        ui.stat("Total detections", str(r["total_detections"]), "bounding boxes across processed frames")
    with m[3]:
        ui.stat("Classes seen", str(len(r["unique_classes"])), ", ".join(r["unique_classes"][:4]) or "none")
    with m[4]:
        ui.stat("Processing time", helpers.format_seconds(r["processing_time_seconds"]), "excludes re-encode")
    with m[5]:
        throughput = r.get("throughput_fps")
        ui.stat("Throughput", f"{throughput:.1f} fps" if throughput else "n/a",
                "source frames read per second")
    with m[6]:
        ui.stat("Unique tracks", str(unique_tracks), "distinct tracker IDs")

    st.caption(
        f"Source {r['video_width']} × {r['video_height']} @ {r['video_fps']:.1f} fps · "
        f"threshold {s['confidence']:.2f} · stride {s['stride']} · "
        f"max frames {s['max_frames'] or 'whole video'}"
    )

    ui.label("Detection analytics")
    if frame.empty:
        ui.empty_state("◌", "No objects detected",
                       "The model found nothing above the confidence threshold. Try a lower threshold.")
    else:
        a, b, c = st.columns(3, gap="large")
        with a:
            show_chart(chart_classes(frame["label"], "Class distribution"))
        with b:
            show_chart(chart_confidence(frame["confidence"], "Confidence distribution"))
        with c:
            import plotly.express as px
            per = helpers.detections_per_frame(frame)
            fig = px.line(per, x="timestamp_seconds", y="Detections", markers=True,
                          title="Detections per processed frame",
                          labels={"timestamp_seconds": "Time (s)"})
            fig.update_traces(line_color=CLASS_COLOR,
                              hovertemplate="t=%{x}s: %{y} detections<extra></extra>")
            show_chart(style_figure(fig))
            st.caption("Frames with zero detections are omitted.")

    ui.label("Object tracking details")
    track_summaries = r.get("track_summaries") or []
    if not tracking_enabled:
        st.info("Tracking IDs were not available for this analysis.")
    elif not track_summaries:
        st.info(
            "No track IDs were produced. Try a video with clearly visible objects and check the confidence threshold."
        )
    else:
        st.dataframe(pd.DataFrame(track_summaries), hide_index=True, **STRETCH)

    ui.label("Detection records")
    if frame.empty:
        st.caption("No records.")
    else:
        st.dataframe(frame, hide_index=True, **STRETCH)

    ui.label("Downloads")
    d1, d2 = st.columns(2, gap="large")
    stem = Path(res["name"]).stem or "video"
    with d1:
        if res["video_bytes"] is not None:
            label = ("Annotated video (H.264 MP4)" if res["download_is_h264"]
                     else "Annotated video (MP4, mp4v codec)")
            st.download_button(label, res["video_bytes"], f"{stem}_annotated.mp4", "video/mp4",
                               key="dl_video", **STRETCH)
        else:
            st.error(res["video_err"] or "Annotated video is unavailable.")
    with d2:
        if res["csv_bytes"] is not None:
            st.download_button("Detection and tracking report (CSV)", res["csv_bytes"],
                               f"{stem}_detections.csv", "text/csv",
                               key="dl_video_csv", **STRETCH)
        else:
            st.error(res["csv_err"] or "CSV report is unavailable.")


# ---------------------------------------------------------------- License Plate Studio
@st.cache_resource(show_spinner=False)
def _ocr_backend():
    return lps.EasyOCRBackend()


@st.cache_resource(show_spinner=False)
def _load_plate_service(path: str, mtime_ns: int):
    """Cached per (path, modification time) so replacing the weights reloads them."""
    return lps.LicensePlateService.from_weights(path, ocr=_ocr_backend())


def get_plate_service(path: Path):
    fs = lps.plate_model_file_status(path)
    if fs["state"] != "found":
        st.session_state.plate_error = f"Plate model unavailable: {fs['detail']} ({fs['path']})"
        st.session_state.plate_loaded = False
        raise lps.PlateModelError(st.session_state.plate_error)
    try:
        service = _load_plate_service(str(path.resolve()), path.stat().st_mtime_ns)
    except lps.PlateModelError as exc:
        st.session_state.plate_error = str(exc)
        st.session_state.plate_loaded = False
        raise
    st.session_state.plate_loaded = True
    st.session_state.plate_error = None
    return service


def reset_plates() -> None:
    st.session_state.plate_latest = None
    st.session_state.plate_upload_n += 1


def run_plate_analysis(image_bgr, filename: str, image_hash: str, threshold: float,
                       run_ocr: bool, model_path: Path) -> None:
    """Runs inference ONLY when called from the Detect button."""
    try:
        with st.status("Analysing license plates…", expanded=True) as status:
            st.write("Loading plate model (first run only)…")
            service = get_plate_service(model_path)
            st.write("Detecting plates" + (" and reading text (the first OCR run may download models)…"
                                           if run_ocr else "…"))
            result = service.analyze_image(image_bgr, confidence=threshold, run_ocr=run_ocr)
            png = helpers.encode_png(result["annotated_image"])
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state.plate_latest = {
            "result": result, "png": png, "filename": filename, "hash": image_hash,
            "threshold": threshold, "analysis_id": uuid.uuid4().hex[:8]}
        st.session_state.history.append(
            helpers.stamp(lps.summarize_plate(filename, result, threshold))
        )
        st.session_state.history = helpers.normalize_history(st.session_state.history)
        persist_history_record(st.session_state.history[-1])
    except lps.PlateModelError as exc:
        st.session_state.plate_latest = None
        show_error(f"The plate model could not be used. {exc}", exc)
    except (ValueError, RuntimeError, OSError) as exc:
        st.session_state.plate_latest = None
        show_error(f"Plate analysis failed: {exc}", exc)
    except Exception as exc:  # unexpected: logged with traceback, shown with detail in debug mode
        st.session_state.plate_latest = None
        show_error(f"Unexpected error during plate analysis: {type(exc).__name__}: {exc}", exc)


def render_plate_setup_help(path: Path) -> None:
    st.markdown(
        "**A plate-trained YOLO model is required.** The general `yolo11n.pt` model has no license-plate "
        "class, so the app deliberately refuses to use it for plates.\n\n"
        f"1. Obtain or train plate weights (a `.pt` file).\n"
        f"2. Save them as `{path}` (or set the environment variable `{lps.PLATE_MODEL_ENV}`, or type the path "
        "under *Model settings*).\n"
        "3. Reload this page. The model's class names are checked when it loads.\n\n"
        "Training your own (needs a labelled plate dataset; check each dataset's license):\n"
        "```\nyolo detect train data=plates.yaml model=yolo11n.pt epochs=50 imgsz=640\n```\n"
        "The result is saved as `runs/detect/train/weights/best.pt`. See `models/README.md`.")


def page_license_plate() -> None:
    ui.page_title("Workspace · License plates", "License Plate Studio",
                  "Detect plates with a plate-trained YOLO model, crop each one and read it with OCR. "
                  "Detection confidence and OCR confidence are reported separately.")
    with st.expander("Model settings"):
        st.text_input("Plate model path (.pt)", value=str(lps.default_plate_model_path(PROJECT_ROOT)),
                      key="plate_model_path_input",
                      help="Local file on this computer. Nothing is uploaded.")
        st.caption("Only load weights you trust: .pt files can execute code when loaded. "
                   "Weights are read from this path; the app does not accept uploaded model files.")
    model_path = plate_model_path()
    fs = lps.plate_model_file_status(model_path)
    pstate, ptone, plines = plate_readiness()
    ostate, otone, olines = ocr_readiness()
    r1, r2 = st.columns(2, gap="large")
    with r1:
        ui.status_card("Plate detection model", pstate, ptone, plines)
    with r2:
        ui.status_card("OCR engine (EasyOCR)", ostate, otone, olines)
    if fs["state"] != "found":
        with st.expander("How to set up the plate model", expanded=True):
            render_plate_setup_help(model_path)

    rail, stage = st.columns([1, 3], gap="large")
    image_bgr = image_hash = upload = None
    with rail:
        ui.label("Input")
        upload = st.file_uploader("Vehicle image", type=helpers.ALLOWED_TYPES,
                                  key=f"plate_upload_{st.session_state.plate_upload_n}",
                                  label_visibility="collapsed")
        if upload is not None:
            if upload.size > lps.MAX_IMAGE_MB * 1024 * 1024:
                st.error(f"Image is {upload.size / 1024 / 1024:.0f} MB; the limit is {lps.MAX_IMAGE_MB} MB.")
            else:
                data = upload.getvalue()
                image_bgr = helpers.decode_image(data)
                image_hash = helpers.file_hash(data)
                if image_bgr is None:
                    st.error("This file could not be decoded as an image. Upload a valid JPG, PNG or WebP.")
        ui.label("Parameters")
        threshold = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05, key="plate_confidence",
                              help="Plates detected below this confidence are discarded.")
        run_ocr = st.checkbox("Read plate text with OCR", value=True, key="plate_run_ocr")
        ready = image_bgr is not None and fs["state"] == "found"
        run = st.button("Detect & read plates", type="primary", key="plate_run", disabled=not ready, **STRETCH)
        if fs["state"] != "found":
            st.caption("Add plate weights to enable detection.")
        st.button("Reset", key="plate_reset", on_click=reset_plates, **STRETCH)
        if run and ready:
            run_plate_analysis(image_bgr, upload.name, image_hash, threshold, run_ocr, model_path)

    latest = st.session_state.plate_latest
    current = latest if (latest and image_hash and latest["hash"] == image_hash) else None
    with stage:
        if image_bgr is None:
            ui.viewport_bar("Viewport", "No input")
            ui.empty_state("⌗", "Load a vehicle image",
                           "Drag and drop a JPG, PNG or WebP. Analysis only runs when you press Detect & read plates.")
            return
        h, w = image_bgr.shape[:2]
        if not current:
            ui.viewport_bar(f"Input · {upload.name}", f"{w} × {h}px")
            st.image(helpers.to_rgb(image_bgr), **STRETCH)
            st.caption("A result for a different image is hidden. Run detection for this one."
                       if latest else "Ready. Press Detect & read plates.")
            return
        render_plate_results(current, image_bgr, upload.name)


def render_plate_results(current: dict, image_bgr, filename: str) -> None:
    res = current["result"]
    dets = res["detections"]
    aid = current["analysis_id"]
    m = st.columns(5)
    with m[0]:
        ui.stat("Plates detected", str(res["plate_count"]))
    with m[1]:
        ui.stat("Detection time", f"{res['detection_time_ms']:.0f} ms")
    with m[2]:
        ui.stat("OCR time", f"{res['ocr_time_ms']:.0f} ms" if res["ocr_requested"] else "off")
    with m[3]:
        ui.stat("Total", f"{res['inference_time_ms']:.0f} ms")
    with m[4]:
        ui.stat("Image", f"{res['image_width']} × {res['image_height']}")
    for note in res["model_warnings"]:
        st.info(note)
    if res["skipped_degenerate"]:
        st.warning(f"{res['skipped_degenerate']} detection(s) were skipped because the box was too small to crop.")
    if res["ocr_requested"] and res["ocr_error"]:
        st.warning(f"OCR unavailable: {res['ocr_error']} Plate detection results are still shown.")

    annotated = helpers.to_rgb(res["annotated_image"])
    tab_a, tab_s, tab_o = st.tabs(["Annotated", "Side by side", "Original"])
    with tab_a:
        ui.viewport_bar("Plate detections", f"{len(dets)} plate(s)")
        st.image(annotated, **STRETCH)
    with tab_s:
        c1, c2 = st.columns(2)
        with c1:
            ui.viewport_bar("Original", filename)
            st.image(helpers.to_rgb(image_bgr), **STRETCH)
        with c2:
            ui.viewport_bar("Annotated", f"{len(dets)} plate(s)")
            st.image(annotated, **STRETCH)
    with tab_o:
        ui.viewport_bar("Source", filename)
        st.image(helpers.to_rgb(image_bgr), **STRETCH)

    ui.label("Plates")
    if not dets:
        ui.empty_state("◌", "No license plate detected",
                       f"Nothing met the {current['threshold']:.2f} detection threshold. Try a lower threshold, "
                       "a sharper or closer image, or check that the model was trained on similar plates.")
        return
    corrections: dict[int, str] = {}
    for d in dets:
        left, right = st.columns([1, 2.4], gap="large")
        with left:
            ui.viewport_bar(f"Plate #{d['plate_id']}", d["class_name"])
            st.image(helpers.to_rgb(d["crop"]), **STRETCH)
        with right:
            s = st.columns(4)
            with s[0]:
                ui.stat("Detection confidence", f"{d['detection_confidence']:.2f}", "plate detector")
            with s[1]:
                ocr_conf = d["ocr_confidence"]
                if ocr_conf is None:
                    ui.stat("OCR status", "—", "not requested")
                else:
                    ocr_label = (
                        "OCR text" if not d.get("correction_applied") else
                        "Corrected OCR"
                    )
                    ui.stat(ocr_label,
                            d["plate_text"] or "—",
                            {"ok": "recognised", "no_text": "no text found",
                             "error": "OCR error", "unavailable": "OCR unavailable"}[d["ocr_status"]])
            with s[2]:
                ocr_conf = d["ocr_confidence"]
                ui.stat("OCR confidence", f"{ocr_conf:.2f}" if ocr_conf is not None else "n/a",
                        "recogniser's own score")
            with s[3]:
                if d.get("correction_applied"):
                    st.caption("↓ correction applied")
                else:
                    st.caption("—")
            b = d["bbox"]
            st.caption(f"Box: ({b['x1']}, {b['y1']}) – ({b['x2']}, {b['y2']})")
            for warning in d["warnings"]:
                st.warning(warning)
            if d["ocr_error"]:
                st.caption(f"OCR error detail: {d['ocr_error']}")
            # Manual correction (kept strictly separate from machine recognition)
            key = f"plate_fix_{aid}_{d['plate_id']}"
            st.text_input("Manual correction (kept separate from the OCR result)",
                          value=d["plate_text"], key=key)
            corrections[d["plate_id"]] = st.session_state.get(key, "")
            # Copy raw OCR code block (for transparency)
            if d["plate_text"]:
                st.code(d["plate_text"], language=None)
        st.divider()

    ui.label("Results and exports")
    rows = lps.plate_rows(res, filename, corrections)
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    d1, d2, d3 = st.columns(3)
    with d1:
        if current["png"]:
            st.download_button("Annotated image (PNG)", current["png"], "visionforge_plates.png",
                               "image/png", key="plate_dl_png", **STRETCH)
    with d2:
        st.download_button("Results (CSV)", lps.rows_to_csv(rows), "visionforge_plates.csv",
                           "text/csv", key="plate_dl_csv", **STRETCH)
    with d3:
        st.download_button("Report (JSON)", lps.build_json_report(res, filename, current["threshold"], corrections),
                           "visionforge_plates.json", "application/json", key="plate_dl_json", **STRETCH)
    st.caption("Exports contain the original OCR text and any manual correction in separate columns.")


# ---------------------------------------------------------------- Analytics
def page_analytics() -> None:
    import plotly.express as px
    ui.page_title("Workspace · Insights", "Analytics",
                  "Historical analysis records are stored locally in SQLite when storage is available.")
    # Normalize history at the point of consumption: drop non-dict legacy
    # entries and guarantee every record is a valid dict.
    history = helpers.normalize_history(st.session_state.history)
    if not history:
        ui.empty_state("▥", "No data to analyse",
                       "Charts are generated from real detections. Analyse an image, plate or video first.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.button("Open Vision Studio", key="an_img", on_click=go_to, args=("Vision Studio",), **STRETCH)
        with c2:
            st.button("Open License Plate Studio", key="an_plate", on_click=go_to,
                      args=("License Plate Studio",), **STRETCH)
        with c3:
            st.button("Open Video Intelligence", key="an_vid", on_click=go_to,
                      args=("Video Intelligence",), **STRETCH)
        return

    # Build a display frame from valid records only.
    det_rows = []
    for i, r in enumerate(history, start=1):
        for lbl, conf in zip(r.get("classes", []), r.get("confidences", [])):
            det_rows.append({
                "Analysis": i,
                "Image": r.get("filename", "unknown"),
                "Object": lbl,
                "Confidence": conf,
            })
    det = pd.DataFrame(det_rows, columns=["Analysis", "Image", "Object", "Confidence"]) if det_rows else pd.DataFrame(columns=["Analysis", "Image", "Object", "Confidence"])

    runs_data = []
    for i, r in enumerate(history, start=1):
        runs_data.append({
            "Run": i,
            "Type": r.get("kind", "unknown").title(),
            "kind": r.get("kind", "unknown"),
            "Source": r.get("filename", "unknown"),
            "Detections": r.get("objects", 0),
            "Processing (s)": (r.get("inference_ms") or 0) / 1000.0,
            "Completed": r.get("completed_at", ""),
        })
    runs = pd.DataFrame(runs_data)
    plate_runs = [r for r in history if r.get("kind") == "plate"]
    ocr_read = sum(1 for r in plate_runs if r.get("ocr_status") == "ok")
    ocr_failed = sum(1 for r in plate_runs if r.get("ocr_status") in ("unavailable", "error", "no_text"))

    c = st.columns(6)
    with c[0]:
        ui.stat("Analyses", str(len(history)))
    with c[1]:
        if runs.empty:
            img_plate_vid = "0 / 0 / 0"
        else:
            img_plate_vid = f"{(runs.kind == 'image').sum()} / {(runs.kind == 'plate').sum()} / {(runs.kind == 'video').sum()}"
        ui.stat("Images / plates / videos", img_plate_vid)
    with c[2]:
        ui.stat("Detections", str(int(runs["Detections"].sum())) if not runs.empty else "0")
    with c[3]:
        ui.stat("Classes seen", str(det["Object"].nunique()) if len(det) else "0")
    with c[4]:
        mean_conf = det["Confidence"].mean() if len(det) else None
        ui.stat("Mean confidence", f"{mean_conf:.1%}" if mean_conf is not None else "n/a",
                "detector confidence only")
    with c[5]:
        ui.stat("Plate OCR read / not read", f"{ocr_read} / {ocr_failed}" if plate_runs else "n/a",
                "plates with OCR requested" if plate_runs else "no plate analyses")

    ui.label("Per-run comparison")
    a, b, c3 = st.columns(3, gap="large")
    color_map = {k.title(): v for k, v in KIND_COLORS.items()}
    with a:
        fig = px.bar(runs, x="Run", y="Detections", color="Type", color_discrete_map=color_map,
                     hover_data={"Source": True, "Completed": True}, title="Detections per analysis run")
        fig.update_xaxes(dtick=1)
        show_chart(style_figure(fig, legend=True))
    with b:
        by_kind = runs.groupby("Type").size().reset_index(name="Analyses")
        fig = px.bar(by_kind, x="Type", y="Analyses", color="Type", color_discrete_map=color_map,
                     title="Analyses by type")
        fig.update_yaxes(dtick=1)
        show_chart(style_figure(fig))
    with c3:
        fig = px.bar(runs, x="Run", y="Processing (s)", color="Type", color_discrete_map=color_map,
                     hover_data={"Source": True, "Completed": True}, title="Processing duration per run (seconds)")
        fig.update_xaxes(dtick=1)
        show_chart(style_figure(fig, legend=True))

    ui.label("Detection distributions")
    if det.empty:
        ui.empty_state("◌", "No detections recorded", "The analyses so far returned no objects.")
    else:
        x, y = st.columns(2, gap="large")
        with x:
            show_chart(chart_classes(det["Object"], "Class distribution (all runs)"))
        with y:
            show_chart(chart_confidence(det["Confidence"], "Detection confidence distribution (all runs)"))

    if plate_runs and (ocr_read + ocr_failed) > 0:
        ui.label("License plate OCR")
        ocr_df = pd.DataFrame({"Outcome": ["Read", "Not read"], "Plates": [ocr_read, ocr_failed]})
        fig = px.bar(ocr_df, x="Outcome", y="Plates", color="Outcome",
                     color_discrete_map={"Read": "#3ddc97", "Not read": "#f2b84b"},
                     title="OCR outcomes (plates where OCR was requested)")
        fig.update_yaxes(dtick=1)
        left, _ = st.columns(2, gap="large")
        with left:
            show_chart(style_figure(fig))

    ui.label("Analysis history")
    log_rows = []
    for i, r in enumerate(history, start=1):
        log_rows.append({
            "#": i,
            "Type": r.get("kind", "unknown").title(),
            "Source": r.get("filename", "unknown"),
            "Threshold": r.get("threshold", "—"),
            "Detections": r.get("objects", 0),
            "Mean confidence": r.get("mean_confidence"),
            "Processing (s)": (r.get("inference_ms") or 0) / 1000.0,
            "Frames read": r.get("frames_read"),
            "Frames processed": r.get("frames_processed"),
            "Completed": r.get("completed_at", ""),
        })
    log = pd.DataFrame(log_rows)
    st.dataframe(log, hide_index=True, **STRETCH, column_config={
        "Mean confidence": st.column_config.NumberColumn(format="%.2f"),
        "Processing (s)": st.column_config.NumberColumn(format="%.2f")})
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download analysis history (CSV)", log.to_csv(index=False),
                           "visionforge_analysis_history.csv", "text/csv", key="dl_history_csv", **STRETCH)
    with d2:
        st.download_button("Download all session detections (CSV)", det.to_csv(index=False),
                           "visionforge_session_detections.csv", "text/csv",
                           key="dl_session_csv", disabled=det.empty, **STRETCH)


# ---------------------------------------------------------------- Platform Overview
def page_platform() -> None:
    ui.page_title("System · About", "Platform Overview",
                  "VisionForge AI is a local computer vision platform: upload images, plates or video, run "
                  "pretrained YOLO detectors, inspect annotated results and export the data. "
                  "Nothing is sent to external services by the app itself.")
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        ui.label("Architecture")
        st.code(
            "Browser\n"
            "  └─ Streamlit UI                 mainn2.py (entry point)\n"
            "       ├─ components.py · assets/neural.css   design system\n"
            "       ├─ helpers.py              tables, temp files, summaries\n"
            "       ├─ video_service.py        validation, per-run folders, H.264 re-encode\n"
            "       ├─ license_plate_service.py plate model check, crops, OCR, exports\n"
            "       └─ modules.py              module status registry\n"
            "            │\n"
            "   ┌────────┼───────────────────┐\n"
            "   ▼        ▼                   ▼\n"
            " DetectionEngine  VideoProcessor   LicensePlateService\n"
            " core/detection/  core/video/      plate YOLO model + EasyOCR\n"
            " (general YOLO)   (general YOLO)   (separate weights)\n"
            "   │        │                   │\n"
            "   └────────┴───── Ultralytics ─┘\n"
            "  image: annotated array + detections\n"
            "  video: annotated MP4 + frame-level CSV\n"
            "  plates: crops + OCR text + CSV/JSON",
            language="text")
        ui.label("Known limitations")
        st.markdown(
            "- **Plate detection needs plate-trained weights that you supply.** None ship with this project, "
            "and the general `yolo11n.pt` is refused for plates.\n"
            "- OCR accuracy depends on the plate image; every result shows its own OCR confidence.\n"
            "- Video analysis is per-frame detection: **no object tracking**, so totals count detections, "
            "not unique objects.\n"
            "- History lives in the browser session and is lost on reload.\n"
            "- In-browser video playback needs FFmpeg for H.264 re-encoding.\n"
            "- CPU inference can be slow on long or high-resolution video; use frame stride.")
    with b:
        ui.label("Readiness (measured on this machine)")
        ocr = lps.ocr_dependency_status()
        pstate, _, plines = plate_readiness()
        ffmpeg = vs.find_ffmpeg()
        weights = helpers.find_weights(PROJECT_ROOT, MODEL_NAME)
        rows = [
            ("Entry file", Path(__file__).name),
            ("Project root", str(PROJECT_ROOT)),
            ("General model", f"{MODEL_NAME}: " + (str(weights) if weights else "local file not found")),
            ("General model state", "load failed" if st.session_state.engine_error else
             ("loaded" if st.session_state.engine_loaded else "not loaded yet")),
            ("Compute device", engine_device() or "unknown until a model loads"),
            ("Plate model path", str(plate_model_path())),
            ("Plate model state", pstate),
            ("EasyOCR", (f"installed {ocr['version'] or ''}".strip() +
                         (", model files cached" if ocr["weights_cached"] else ", model files not cached"))
             if ocr["installed"] else "not installed"),
            ("FFmpeg", ffmpeg or "not found (video plays only after download)"),
            ("Debug mode", "on" if DEBUG else "off (set VISIONFORGE_DEBUG=1 to show tracebacks)"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Item", "Status"]), hide_index=True, **STRETCH)
        ui.label("Installed versions")
        techs = [("Python", sys.version.split()[0]), ("Streamlit", st.__version__),
                 ("Ultralytics", pkg_version("ultralytics")), ("PyTorch", pkg_version("torch")),
                 ("OpenCV", pkg_version("opencv-python") if pkg_version("opencv-python") != "not installed"
                  else pkg_version("opencv-python-headless")),
                 ("Pandas", pkg_version("pandas")), ("Plotly", pkg_version("plotly")),
                 ("EasyOCR", pkg_version("easyocr")), ("imageio-ffmpeg", pkg_version("imageio-ffmpeg"))]
        st.dataframe(pd.DataFrame(techs, columns=["Component", "Version"]), hide_index=True, **STRETCH)

    ui.label("Configuration")
    st.markdown(
        f"- **Plate model:** save weights as `models/license_plate_detector.pt`, or set `{lps.PLATE_MODEL_ENV}` "
        "to a full path, or type a path under *Model settings* in License Plate Studio.\n"
        "- **OCR:** `pip install easyocr`. The first OCR run downloads its model files (needs internet, roughly "
        "100 MB); after that it works offline.\n"
        "- **Video playback:** install FFmpeg, or `pip install imageio-ffmpeg`.\n"
        "- **Debug:** set `VISIONFORGE_DEBUG=1` before launching to show technical details on errors.")

    ui.label("Module status")
    for group, title in ((IMPLEMENTED, "Implemented"), (PARTIAL, "Partial"), (PLANNED, "Planned")):
        items = [m for m in MODULES if m["status"] == group]
        if not items:
            continue
        st.markdown(f"**{title}**")
        cols = st.columns(4)
        for i, m in enumerate(items):
            with cols[i % 4]:
                ui.module_card(m["icon"], m["name"], m["detail"], m["status"])
        st.write("")


PAGE_RENDERERS = {
    "Command Center": page_command_center,
    "Vision Studio": page_vision_studio,
    "License Plate Studio": page_license_plate,
    "Video Intelligence": page_video,
    "Analytics": page_analytics,
    "Platform Overview": page_platform,
}
PAGE_RENDERERS[page]()

st.markdown(f'<div class="vf-foot">VisionForge AI v{APP_VERSION} · {APP_PHASE}</div>',
            unsafe_allow_html=True)