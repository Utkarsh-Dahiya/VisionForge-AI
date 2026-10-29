"""Video Intelligence service layer (no Streamlit imports, fully testable).

Wraps the existing ``core.video.video_processor.VideoProcessor``; it does not
implement a second pipeline. It adds: upload validation, safe temp storage,
metadata probing, per-run output folders, and H.264 re-encoding for browser
playback when an FFmpeg binary is available.
"""
from __future__ import annotations

import atexit
import inspect
import logging
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2

log = logging.getLogger("visionforge.video")

# Extensions we accept for upload. Whether a file actually decodes depends on
# the installed OpenCV build, so every file is probed before analysis.
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
MAX_VIDEO_MB = 200  # keep in sync with .streamlit/config.toml maxUploadSize


class VideoValidationError(ValueError):
    """The uploaded file cannot be used for analysis."""


class ProcessorVersionError(RuntimeError):
    """video_service.py and core/video/video_processor.py come from different releases.

    Raised when a tracking option was requested that the installed VideoProcessor
    cannot honour. The option is never dropped silently.
    """


# Options added to VideoProcessor after its first release, each with the value that
# means "not requested". With an older processor an option holding its neutral value
# is omitted (and logged); any other value raises ProcessorVersionError.
_NEUTRAL_OPTION_VALUES: dict[str, Any] = {"iou": None, "min_track_observations": 3}


def _processor_options(processor_cls: type, requested: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of ``requested`` that ``processor_cls.process_video`` accepts.

    Every requested option is either forwarded, deliberately omitted because it still
    has its neutral value and the processor predates it, or reported as an error.
    """
    parameters = inspect.signature(processor_cls.process_video).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    forwarded: dict[str, Any] = {}
    for name, value in requested.items():
        if accepts_any or name in parameters:
            forwarded[name] = value
        elif name in _NEUTRAL_OPTION_VALUES and value == _NEUTRAL_OPTION_VALUES[name]:
            log.warning("Installed VideoProcessor has no '%s' option; omitting it (value %r is the "
                        "neutral default).", name, value)
        else:
            try:
                where = inspect.getsourcefile(processor_cls) or "unknown location"
            except TypeError:
                where = "unknown location"
            raise ProcessorVersionError(
                f"The video option '{name}={value!r}' was requested, but the installed VideoProcessor "
                f"({where}) does not support it. video_service.py and core/video/video_processor.py are "
                "from different VisionForge releases. Replace core/video/video_processor.py with the "
                "matching version, delete the __pycache__ folders, and fully restart Streamlit."
            )
    return forwarded


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int | None  # None when the container does not report it
    duration_seconds: float | None
    size_bytes: int


def validate_upload(name: str, size_bytes: int) -> None:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext not in VIDEO_TYPES:
        raise VideoValidationError(
            f"Unsupported file type '.{ext}'. Accepted: "
            + ", ".join(f".{t}" for t in VIDEO_TYPES) + "."
        )
    if size_bytes <= 0:
        raise VideoValidationError("The uploaded file is empty.")
    if size_bytes > MAX_VIDEO_MB * 1024 * 1024:
        raise VideoValidationError(
            f"File is {size_bytes / 1024 / 1024:.0f} MB; the limit is {MAX_VIDEO_MB} MB."
        )


def _register_cleanup(path: Path) -> None:
    atexit.register(shutil.rmtree, str(path), True)


def remove_tree(path: str | Path | None) -> None:
    if path:
        shutil.rmtree(str(path), ignore_errors=True)


def save_upload(data: bytes | memoryview, name: str) -> Path:
    """Persist an upload to a unique temp folder. Returns the file path.

    The stored name is generic (``input<ext>``) so odd or duplicate user
    filenames can never collide or escape the folder.
    """
    validate_upload(name, len(data))
    folder = Path(tempfile.mkdtemp(prefix="visionforge_in_"))
    _register_cleanup(folder)
    target = folder / f"input{Path(name).suffix.lower()}"
    target.write_bytes(data)
    return target


def probe_video(path: str | Path) -> VideoMeta:
    """Open the file with OpenCV and read its properties and first frame."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoValidationError(
                "OpenCV could not open this video. The file may be corrupt or "
                "use a codec your OpenCV build does not support."
            )
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoValidationError("The video contains no readable frames.")
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
    finally:
        capture.release()
    fps_valid = fps if fps > 0 else 0.0
    frame_count = count if count > 0 else None
    duration = frame_count / fps_valid if frame_count and fps_valid else None
    return VideoMeta(width, height, fps_valid, frame_count, duration, path.stat().st_size)


def estimate_frames(
    meta: VideoMeta, stride: int, max_frames: int | None
) -> tuple[int, int] | None:
    """(source frames to read, frames detection will run on), or None if unknown.

    Follows VideoProcessor semantics: detection runs on source frames
    0, N, 2N, ... of those read.
    """
    if meta.frame_count is None:
        return None
    to_read = min(meta.frame_count, max_frames) if max_frames else meta.frame_count
    to_process = (to_read + stride - 1) // stride
    return to_read, to_process


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # optional: pip install imageio-ffmpeg bundles a binary
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # package missing or binary unavailable
        return None


def make_web_playable(src: str | Path, timeout: int = 600) -> Path | None:
    """Re-encode to H.264/yuv420p so browsers can play it.

    OpenCV's ``mp4v`` output is generally not playable in browsers. Returns the
    new file, or None if FFmpeg is unavailable or encoding fails.
    """
    src = Path(src)
    exe = find_ffmpeg()
    if exe is None:
        return None
    out = src.with_name(src.stem + "_web.mp4")
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        out.unlink(missing_ok=True)
        return None
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        return None
    return out


def analyze_video(
    engine: Any,
    input_path: str | Path,
    output_root: str | Path,
    confidence: float,
    frame_stride: int,
    max_frames: int | None,
    progress_callback: Callable[[int, int | None], None] | None = None,
    transcode: bool = True,
    *,
    enable_tracking: bool = True,
    tracker: str = "bytetrack.yaml",
    iou: float | None = None,
    trail_length: int = 30,
    min_track_observations: int = 3,
) -> dict[str, Any]:
    """Run the existing VideoProcessor into a fresh per-run folder.

    Returns the processor's result dict plus: ``run_dir``, ``playback_path``
    (browser-playable file or None), ``playback_note``, ``wall_seconds``,
    ``throughput_fps`` and ``transcode_seconds``.

    The keyword-only tracking options are forwarded to
    ``VideoProcessor.process_video``; their defaults match the processor, so
    existing callers behave exactly as before. If the installed processor
    predates an option, that option is omitted only while it still has its
    neutral default; otherwise ``ProcessorVersionError`` is raised. No option is
    ever dropped silently.
    Exceptions from the processor propagate unchanged; the run folder is
    removed if processing fails.
    """
    from core.video.video_processor import VideoProcessor  # lazy: pulls in YOLO

    run_dir = Path(output_root) / uuid.uuid4().hex[:10]
    started = perf_counter()
    try:
        options = _processor_options(VideoProcessor, {
            "enable_tracking": enable_tracking,
            "tracker": tracker,
            "iou": iou,
            "trail_length": trail_length,
            "min_track_observations": min_track_observations,
        })
        processor = VideoProcessor(engine, output_dir=run_dir)
        result = processor.process_video(
            input_path,
            confidence=confidence,
            frame_stride=frame_stride,
            max_frames=max_frames,
            progress_callback=progress_callback,
            **options,
        )
    except BaseException:
        remove_tree(run_dir)
        raise
    _register_cleanup(run_dir)

    for key in ("output_video_path", "output_csv_path"):
        p = Path(result[key])
        if not p.is_file() or p.stat().st_size == 0:
            remove_tree(run_dir)
            raise RuntimeError(f"Expected output file is missing or empty: {p.name}")

    result["run_dir"] = str(run_dir)
    t0 = perf_counter()
    playback = make_web_playable(result["output_video_path"]) if transcode else None
    result["transcode_seconds"] = round(perf_counter() - t0, 3)
    result["playback_path"] = str(playback) if playback else None
    if playback:
        result["playback_note"] = "Re-encoded to H.264 for browser playback."
    elif not transcode:
        result["playback_note"] = "Re-encoding was skipped."
    else:
        result["playback_note"] = (
            "FFmpeg was not found, so the output stays in OpenCV's mp4v codec, "
            "which many browsers cannot play. Download the file to watch it, or "
            "install FFmpeg (or `pip install imageio-ffmpeg`) for in-browser playback."
        )
    wall = perf_counter() - started
    result["wall_seconds"] = round(wall, 3)
    pt = result["processing_time_seconds"]
    result["throughput_fps"] = round(result["frames_read"] / pt, 2) if pt > 0 else None
    return result


def read_frame(path: str | Path, index: int):
    """Return frame ``index`` (BGR) from a video, or None if unreadable."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()
