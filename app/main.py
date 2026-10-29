from pathlib import Path
import sys
import time
import hashlib
import tempfile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.detection.detection_engine import DetectionEngine


st.set_page_config(
    page_title="VisionForge AI | Command Center",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Premium visual system
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    :root {
        --bg: #080d18;
        --panel: #101827;
        --panel-light: #151f31;
        --border: rgba(137, 164, 201, 0.18);
        --text: #eaf2ff;
        --muted: #91a3bd;
        --cyan: #43e5f5;
        --violet: #a78bfa;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 0%,
                rgba(29, 91, 131, 0.19),
                transparent 36%
            ),
            radial-gradient(
                ellipse at 90% 10%,
                rgba(101, 64, 170, 0.13),
                transparent 34%
            ),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 24, 0.75);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0d1524 0%,
            #0a101c 100%
        );
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        max-width: 1550px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #f1f7ff;
    }

    .brand span {
        color: var(--cyan);
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 2.4px;
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--cyan);
        margin-bottom: 0.7rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem;
        border-radius: 22px;
        border: 1px solid rgba(105, 211, 245, 0.24);
        background:
            linear-gradient(
                120deg,
                rgba(17, 36, 60, 0.98),
                rgba(13, 26, 45, 0.94) 58%,
                rgba(42, 29, 76, 0.92)
            );
        margin-bottom: 1.6rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -70px;
        top: -90px;
        border: 1px solid rgba(67, 229, 245, 0.23);
        border-radius: 50%;
        box-shadow:
            0 0 0 25px rgba(67, 229, 245, 0.025),
            0 0 0 55px rgba(67, 229, 245, 0.02);
        pointer-events: none;
    }

    .hero h1 {
        color: white !important;
        font-size: clamp(2rem, 4vw, 3.4rem) !important;
        margin: 0.1rem 0 0.7rem 0;
    }

    .hero p {
        color: #b9cce5;
        max-width: 690px;
        font-size: 1rem;
        line-height: 1.8;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(67, 229, 245, 0.08);
        border: 1px solid rgba(67, 229, 245, 0.24);
        color: #9af5fc;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.6px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #43e5a0;
        box-shadow: 0 0 12px rgba(67, 229, 160, 0.8);
    }

    .section-label {
        color: var(--muted);
        font-size: 0.74rem;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        font-weight: 700;
        margin: 1.5rem 0 0.9rem 0;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(21, 33, 52, 0.96),
            rgba(12, 21, 35, 0.98)
        );
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        min-height: 115px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.6rem;
    }

    .metric-value {
        color: #f3f8ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .metric-note {
        color: var(--cyan);
        font-size: 0.73rem;
        margin-top: 0.4rem;
    }

    .panel {
        background: rgba(16, 24, 39, 0.78);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 1.15rem;
    }

    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f7ff;
        margin-bottom: 0.35rem;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.6;
    }

    .module-card {
        border: 1px solid var(--border);
        background: rgba(16, 24, 39, 0.8);
        border-radius: 15px;
        padding: 1.1rem;
        min-height: 135px;
    }

    .module-icon {
        font-size: 1.35rem;
        margin-bottom: 0.7rem;
    }

    .module-name {
        color: #edf4ff;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .module-desc {
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.5;
    }

    div[data-testid="stMetric"] {
        background: rgba(16, 24, 39, 0.9);
        border: 1px solid var(--border);
        padding: 1rem;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: #f1f7ff;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(67, 229, 245, 0.36);
        background: linear-gradient(
            100deg,
            #123c54,
            #17435b
        );
        color: #e9fcff;
        font-weight: 700;
        min-height: 43px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--cyan);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 5px 22px rgba(67, 229, 245, 0.12);
    }

    .stDownloadButton > button {
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(67, 229, 245, 0.34);
        border-radius: 14px;
        background: rgba(17, 34, 52, 0.42);
        padding: 0.6rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    hr {
        border-color: var(--border);
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Detection engine
# ---------------------------------------------------------

@st.cache_resource
def load_detection_engine() -> DetectionEngine:
    return DetectionEngine(model_path=str(PROJECT_ROOT / "yolo11n.pt"))


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0

if "last_detection_result" not in st.session_state:
    st.session_state.last_detection_result = None

if "last_detection_filename" not in st.session_state:
    st.session_state.last_detection_filename = None

if "last_detection_input_hash" not in st.session_state:
    st.session_state.last_detection_input_hash = None


# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
if "navigation_target" in st.session_state:
    st.session_state["page_navigation"] = (
        st.session_state.pop("navigation_target")
    )

with st.sidebar:
    st.markdown(
        '<div class="brand">VISION<span>FORGE</span> AI</div>',
        unsafe_allow_html=True,
    )

    st.caption("NEURAL VISION PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Command Center",
            "Vision Studio",
            "Analytics",
            "Platform Overview",
        ],
        label_visibility="visible",
        key="page_navigation",
    )

    st.divider()

    st.markdown("**SYSTEM STATUS**")

    st.markdown(
        """
        <div class="status">
            <span class="status-dot"></span>
            APPLICATION ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("YOLO11 Nano · Image Inference")

    st.divider()

    st.caption("VISIONFORGE AI")
    st.caption("Computer Vision & Visual Intelligence")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">NEXT-GENERATION COMPUTER VISION</div>
        <h1>See beyond the pixels.</h1>
        <p>
            An AI-powered visual intelligence workspace for object
            detection, visual analysis, and evidence-driven insights.
            Explore your images through a unified computer vision engine.
        </p>
        <div class="status">
            <span class="status-dot"></span>
            CORE DETECTION ENGINE READY
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Metric helper
# ---------------------------------------------------------

def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Command Center
# ---------------------------------------------------------

if page == "Command Center":

    st.markdown(
        '<div class="section-label">OPERATIONS OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        render_metric(
            "Images analyzed this session",
            str(st.session_state.analysis_count),
            "Current browser session",
        )

    with m2:
        render_metric(
            "Objects detected",
            str(st.session_state.total_detections),
            "Across successful analyses",
        )

    with m3:
        render_metric(
            "Detection engine",
            "READY",
            "YOLO11 Nano configured",
        )

    with m4:
        render_metric(
            "AI modules",
            "01 ACTIVE",
            "Image detection available",
        )

    st.markdown(
        '<div class="section-label">WORKSPACE MODULES</div>',
        unsafe_allow_html=True,
    )

    modules = [
        (
            "◈",
            "Vision Studio",
            "Upload images and run real YOLO object detection.",
        ),
        (
            "⌁",
            "Video Intelligence",
            "Planned: frame-by-frame video analysis.",
        ),
        (
            "⬡",
            "Safety Intelligence",
            "Planned: PPE and configurable safety rules.",
        ),
        (
            "◎",
            "Tracking & Events",
            "Planned: object tracking and event detection.",
        ),
        (
            "⌗",
            "OCR Studio",
            "Planned: text extraction from visual inputs.",
        ),
        (
            "▥",
            "Analytics",
            "Explore the results of completed analyses.",
        ),
    ]

    for row_start in range(0, len(modules), 3):
        cols = st.columns(3)

        for col, item in zip(
            cols,
            modules[row_start:row_start + 3],
        ):
            icon, name, description = item

            with col:
                st.markdown(
                    f"""
                    <div class="module-card">
                        <div class="module-icon">{icon}</div>
                        <div class="module-name">{name}</div>
                        <div class="module-desc">{description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")

    st.markdown(
        '<div class="section-label">LATEST ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    last_result = st.session_state.last_detection_result

    if last_result is None:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Your workspace is ready</div>
                <div class="panel-subtitle">
                    No image analysis has been completed in this
                    session yet. Open Vision Studio to run your
                    first analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Open Vision Studio", type="primary"):
            st.session_state.navigation_target = "Vision Studio"
            st.rerun()

    else:
        st.success(
            f"Latest image: "
            f"{st.session_state.last_detection_filename or 'Uploaded image'}"
        )

        a, b = st.columns(2)

        with a:
            st.metric(
                "Latest object count",
                last_result["object_count"],
            )

        with b:
            st.metric(
                "Last inference time",
                f'{last_result["inference_time_ms"]:.2f} ms',
            )


# ---------------------------------------------------------
# Vision Studio
# ---------------------------------------------------------

elif page == "Vision Studio":

    st.markdown(
        '<div class="section-label">IMAGE INTELLIGENCE WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Visual inspection pipeline</div>
            <div class="panel-subtitle">
                Upload an image, configure the confidence threshold,
                and execute real object detection using the YOLO model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.sidebar:
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Adjust the minimum confidence threshold.",
        )

    uploaded_file = st.file_uploader(
        "Upload an image for analysis",
        type=["jpg", "jpeg", "png", "webp"],
        key="vision_studio_upload",
    )

    current_image_hash = None

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        current_image_hash = hashlib.sha256(image_bytes).hexdigest()

        image_array = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image_array is None:
            st.error("The uploaded file could not be decoded.")
        else:
            original_rgb = cv2.cvtColor(
                image_array,
                cv2.COLOR_BGR2RGB,
            )

            st.image(
                original_rgb,
                caption="Input image",
                use_container_width=True,
            )

            if st.button(
                "▶ Run AI Detection",
                type="primary",
                use_container_width=True,
            ):
                temporary_path = None
                output_directory = PROJECT_ROOT / "outputs"
                output_directory.mkdir(parents=True, exist_ok=True)

                try:
                    # A unique temporary file prevents concurrent uploads
                    # from overwriting one another.
                    with tempfile.NamedTemporaryFile(
                        prefix="visionforge_upload_",
                        suffix=".png",
                        dir=output_directory,
                        delete=False,
                    ) as temporary_file:
                        temporary_path = Path(temporary_file.name)

                    with st.spinner("Running object detection..."):
                        if not cv2.imwrite(str(temporary_path), image_array):
                            raise RuntimeError(
                                "Could not prepare the input image."
                            )

                        engine = load_detection_engine()
                        result = engine.detect_image(
                            image_path=str(temporary_path),
                            confidence=confidence,
                        )

                    st.session_state.last_detection_result = result
                    st.session_state.last_detection_filename = uploaded_file.name
                    st.session_state.last_detection_input_hash = current_image_hash
                    st.session_state.analysis_count += 1
                    st.session_state.total_detections += result["object_count"]

                except Exception as exc:
                    st.error("Detection failed.")
                    st.exception(exc)

                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)

    result = st.session_state.last_detection_result
    show_current_result = (
        uploaded_file is not None
        and current_image_hash is not None
        and st.session_state.last_detection_input_hash is not None
        and current_image_hash == st.session_state.last_detection_input_hash
    )

    if result is not None and show_current_result:

        st.divider()

        st.markdown(
            '<div class="section-label">INFERENCE OUTPUT</div>',
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)

        a.metric("Objects", result["object_count"])
        b.metric(
            "Prediction time",
            f'{result["inference_time_ms"]:.2f} ms',
        )
        c.metric(
            "Resolution",
            f'{result["image_width"]} × {result["image_height"]}',
        )

        result_col, details_col = st.columns(
            [1.5, 1],
            gap="large",
        )

        with result_col:
            st.markdown("#### Detection Visualization")

            st.image(
                cv2.cvtColor(
                    result["annotated_image"],
                    cv2.COLOR_BGR2RGB,
                ),
                use_container_width=True,
            )

            success, encoded_image = cv2.imencode(
                ".png",
                result["annotated_image"],
            )

            if success:
                st.download_button(
                    "Download annotated image",
                    data=encoded_image.tobytes(),
                    file_name="visionforge_detection.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with details_col:
            st.markdown("#### Detected Objects")

            rows = []

            for index, detection in enumerate(
                result["detections"],
                start=1,
            ):
                rows.append(
                    {
                        "ID": index,
                        "Object": detection["label"],
                        "Confidence": detection["confidence"],
                    }
                )

            if rows:
                dataframe = pd.DataFrame(rows)

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

                distribution = (
                    dataframe["Object"]
                    .value_counts()
                    .rename_axis("Object")
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    distribution,
                    x="Object",
                    y="Count",
                    color="Object",
                    template="plotly_dark",
                    title="Object distribution",
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(l=5, r=5, t=45, b=5),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:
                st.info(
                    "No detections were returned for this image."
                )

        if rows:
            csv_data = pd.DataFrame(rows).to_csv(index=False)

            st.download_button(
                "Download detection table (CSV)",
                data=csv_data,
                file_name="visionforge_detections.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------
# Analytics
# ---------------------------------------------------------

elif page == "Analytics":

    st.markdown(
        '<div class="section-label">VISUAL ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    result = st.session_state.last_detection_result

    if result is None:
        st.info(
            "Run an image analysis in Vision Studio first. "
            "Analytics will then display its actual detection results."
        )

    else:
        detections = result["detections"]

        if not detections:
            st.info(
                "The latest analysis returned no detections. "
                "Try another image or adjust the confidence threshold."
            )
        else:
            df = pd.DataFrame(detections)

            c1, c2, c3 = st.columns(3)

            c1.metric("Total detections", len(df))
            c2.metric("Unique classes", df["label"].nunique())
            c3.metric(
                "Mean confidence",
                f'{df["confidence"].mean():.1%}',
            )

            distribution = (
                df["label"]
                .value_counts()
                .rename_axis("Object class")
                .reset_index(name="Count")
            )

            fig = px.bar(
                distribution,
                x="Object class",
                y="Count",
                color="Object class",
                template="plotly_dark",
                title="Detected class distribution",
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            confidence_fig = px.histogram(
                df,
                x="confidence",
                nbins=10,
                template="plotly_dark",
                title="Detection confidence distribution",
            )

            confidence_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                confidence_fig,
                use_container_width=True,
            )


# ---------------------------------------------------------
# Platform overview
# ---------------------------------------------------------

elif page == "Platform Overview":

    st.markdown(
        '<div class="section-label">SYSTEM ARCHITECTURE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">VisionForge AI architecture</div>
            <div class="panel-subtitle">
                The application separates its user interface from
                its computer vision engine so additional modules
                can be integrated without duplicating inference logic.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.code(
        """
        USER
          |
          v
        STREAMLIT APPLICATION
          |
          v
        DETECTION ENGINE
          |
          v
        YOLO MODEL
          |
          v
        STRUCTURED DETECTIONS
          |
          +--> Annotated image
          +--> Object counts
          +--> Confidence scores
          +--> Inference timing
          +--> Analytics and exports
        """,
        language="text",
    )

    st.subheader("Implementation status")

    status_rows = [
        {
            "Module": "YOLO image detection",
            "Status": "Implemented; basic inference tested",
        },
        {
            "Module": "Command Center",
            "Status": "Initial interface implementation",
        },
        {
            "Module": "Image analytics",
            "Status": "Session-level results",
        },
        {
            "Module": "Video Intelligence",
            "Status": "Planned",
        },
        {
            "Module": "PPE monitoring",
            "Status": "Planned",
        },
        {
            "Module": "Object tracking",
            "Status": "Planned",
        },
        {
            "Module": "OCR",
            "Status": "Planned",
        },
        {
            "Module": "Persistent database",
            "Status": "Planned",
        },
        {
            "Module": "AI reporting",
            "Status": "Planned",
        },
    ]

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "VISIONFORGE AI · NEURAL VISION PLATFORM · "
    "Phase 1: Object Detection Studio"
)