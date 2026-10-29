"""Reusable UI components for VisionForge AI.

All dynamic text is HTML-escaped. The stylesheet loader also appends a small
"navigation guard" so the sidebar can never be hidden by a stale neural.css.
"""
from __future__ import annotations

import logging
import re
from html import escape
from pathlib import Path

import streamlit as st

log = logging.getLogger("visionforge.ui")

_HERE = Path(__file__).resolve().parent
_CSS_CANDIDATES = (
    _HERE / "assets" / "neural.css",
    _HERE / "neural.css",
    _HERE / "app" / "assets" / "neural.css",
)

# An old stylesheet that hides the Streamlit sidebar is the cause of the
# "only Command Center, no navigation" problem. Only rules whose selector IS the
# sidebar (or its expand control) count; e.g. `[data-testid="stSidebar"] label{...}` does not.
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_SIDEBAR_SELECTOR = re.compile(
    r'\s*\[data-testid="(stSidebar|stSidebarCollapsedControl|stExpandSidebarButton|stSidebarContent)"\]\s*')


def css_hides_sidebar(css: str) -> bool:
    """True if any rule applies display:none to the sidebar itself."""
    for selectors, body in _RULE.findall(css):
        if re.search(r"display\s*:\s*none", body) and any(
                _SIDEBAR_SELECTOR.fullmatch(sel) for sel in selectors.split(",")):
            return True
    return False


# Appended AFTER the stylesheet. It only overrides display/visibility, so the
# native collapse/expand behaviour (which uses transforms) keeps working.
NAV_GUARD_CSS = """
[data-testid="stSidebar"]{display:flex !important;visibility:visible !important;}
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"]{display:flex !important;visibility:visible !important;}
[data-testid="stSidebar"] [data-testid="stRadio"]{display:block !important;visibility:visible !important;}
"""


def inject_theme() -> dict:
    """Inject neural.css plus the navigation guard. Returns diagnostics."""
    info = {"path": None, "stale_sidebar_rule": False, "found": False}
    css = ""
    for candidate in _CSS_CANDIDATES:
        if candidate.is_file():
            try:
                css = candidate.read_text(encoding="utf-8")
            except OSError as exc:
                log.warning("Could not read stylesheet %s: %s", candidate, exc)
                continue
            info.update(path=str(candidate), found=True)
            break
    if not info["found"]:
        log.warning("neural.css not found; using default styling plus navigation guard.")
    if css_hides_sidebar(css):
        info["stale_sidebar_rule"] = True
        log.warning("Stylesheet %s hides the sidebar; the navigation guard overrides it. "
                    "Replace it with the supplied assets/neural.css.", info["path"])
    st.markdown(f"<style>{css}\n{NAV_GUARD_CSS}</style>", unsafe_allow_html=True)
    return info


def _html(fragment: str) -> None:
    st.markdown(fragment, unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
def sidebar_brand() -> None:
    _html(
        """<div class="vf-sb-brand"><span class="vf-mark">◈</span>
        <div><div class="vf-sb-name">VISION<b>FORGE</b> AI</div>
        <div class="vf-sb-sub">Visual Intelligence Platform</div></div></div>"""
    )


def sidebar_status(rows: list) -> None:
    """rows: (label, value, tone) tuples with tone ok | warn | idle. Plain strings also accepted."""
    items = []
    for row in rows:
        if isinstance(row, str):
            items.append(f'<div class="vf-sb-row"><span>{escape(row)}</span></div>')
            continue
        key, value, tone = (tuple(row) + ("idle",))[:3]
        items.append(
            f'<div class="vf-sb-row"><span>{escape(str(key))}</span>'
            f'<span class="vf-sb-val {escape(str(tone))}"><i class="vf-dot"></i>{escape(str(value))}</span></div>'
        )
    _html(f'<div class="vf-sb-status"><div class="vf-label">System</div>{"".join(items)}</div>')


def sidebar_footer(text: str) -> None:
    _html(f'<div class="vf-sb-foot">{escape(text).replace(chr(10), "<br>")}</div>')


# ---------------------------------------------------------------- page chrome
def page_title(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    _html(f'<div class="vf-title"><div class="vf-eyebrow">{escape(eyebrow)}</div>'
          f"<h1>{escape(title)}</h1>{sub}</div>")


def label(text: str) -> None:
    _html(f'<div class="vf-label">{escape(text)}</div>')


def stat(name: str, value: str, note: str = "") -> None:
    note_html = f'<div class="vf-stat-note">{escape(note)}</div>' if note else ""
    _html(f'<div class="vf-stat"><div class="vf-stat-name">{escape(name)}</div>'
          f'<div class="vf-stat-value">{escape(value)}</div>{note_html}</div>')


def status_badge(status: str) -> str:
    return f'<span class="vf-badge {escape(status.lower())}">{escape(status)}</span>'


def module_card(icon: str, name: str, desc: str, status: str) -> None:
    _html(f'<div class="vf-module {escape(status.lower())}"><div class="vf-module-top">'
          f'<span class="vf-icon">{escape(icon)}</span>{status_badge(status)}</div>'
          f'<div class="vf-module-name">{escape(name)}</div>'
          f'<div class="vf-module-desc">{escape(desc)}</div></div>')


def empty_state(icon: str, title: str, body: str) -> None:
    _html(f'<div class="vf-empty"><div class="vf-empty-icon">{escape(icon)}</div>'
          f'<div class="vf-empty-title">{escape(title)}</div>'
          f'<div class="vf-empty-body">{escape(body)}</div></div>')


def viewport_bar(left: str, right: str = "") -> None:
    _html(f'<div class="vf-vp-bar"><span>{escape(left)}</span><span>{escape(right)}</span></div>')


def info_panel(title: str, body: str) -> None:
    _html(f'<div class="vf-panel"><div class="vf-panel-title">{escape(title)}</div>'
          f'<div class="vf-panel-body">{escape(body)}</div></div>')


def status_card(title: str, state: str, tone: str, lines: list[str]) -> None:
    """Readiness card: tone ok | warn | idle."""
    body = "".join(f"<div>{escape(x)}</div>" for x in lines)
    _html(f'<div class="vf-panel vf-readiness {escape(tone)}"><div class="vf-panel-title">'
          f'<i class="vf-dot"></i>{escape(title)} <span class="vf-state">{escape(state)}</span></div>'
          f'<div class="vf-panel-body">{body}</div></div>')
