import html
import os
import random
import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
from matplotlib.text import Text

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

try:
    from campus_environment import CampusDeliveryEnv
except ImportError:
    from campus_delivery_env import CampusDeliveryEnv


# ==========================================
# 0. SETTINGS (ANIMATION & MODEL)
# ==========================================

INTERP_FRAMES = 4                # sub-frames per step for smooth gliding (no teleporting)
FRAME_DELAY = 0.14               # seconds per sub-frame (increase to slow down movement)
STEP_PAUSE = 0.35                # brief pause after arriving at each node
USE_ANTI_BACKTRACKING = False    # False to match evaluate_agent() in Colab
OBSTACLE_MODE = "random"         # must match OBSTACLE_MODE used in training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "Q-Learning": os.path.join(BASE_DIR, "q_table.npy"),
    "SARSA":      os.path.join(BASE_DIR, "sarsa_table.npy"),
    "A2C":        os.path.join(BASE_DIR, "a2c_table.npy"),
    "REINFORCE":  os.path.join(BASE_DIR, "reinforce_table.npy"),
}


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TAR UMT AI Food Delivery",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. SMART STICKY HEADER JAVASCRIPT
# Hide header on scroll down, reveal on scroll up
# ==========================================

smart_scroll_js = """
<script>
const parentWin = window.parent;
const parentDoc = window.parent.document;

if (!parentWin._campusSmartHeaderInit) {
    let lastScrollY = 0;
    let ticking = false;

    const scrollHandler = function(e) {
        if (!ticking) {
            parentWin.requestAnimationFrame(function() {
                let currentScrollY = parentWin.scrollY;
                if (e.target && e.target.scrollTop !== undefined && e.target.tagName !== 'IFRAME') {
                    currentScrollY = e.target.scrollTop;
                }

                if (currentScrollY <= 70) {
                    parentDoc.body.classList.remove('hide-smart-header');
                } else if (currentScrollY > lastScrollY + 15) {
                    parentDoc.body.classList.add('hide-smart-header');
                } else if (currentScrollY < lastScrollY - 15) {
                    parentDoc.body.classList.remove('hide-smart-header');
                }
                lastScrollY = currentScrollY;
                ticking = false;
            });
            ticking = true;
        }
    };

    parentDoc.addEventListener('scroll', scrollHandler, true);
    parentWin._campusSmartHeaderInit = true;
}
</script>
"""
components.html(smart_scroll_js, height=0, width=0)


# ==========================================
# 3. ADVANCED CSS (LIGHT MODE + BENTO + 3D UI)
# ==========================================

st.markdown("""
<style>
    :root { color-scheme: light; }

    /* ---- Global light background & spacing ---- */
    html, body, .stApp,
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stBottom"], .main {
        background-color: #f4f8fb !important;
        color: #0f172a !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #eef4f9 !important;
    }
    .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 1.8rem !important;
        max-width: 1440px;
    }

    /* ---- Base text visibility ---- */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p {
        color: #1e293b !important;
    }

    /* ---- Smart Sticky Centered Banner (No Emoji) ---- */
    div.element-container:has(.campus-header) {
        position: sticky !important;
        top: 0.7rem !important;
        z-index: 9999 !important;
        transition: transform 0.38s cubic-bezier(0.3, 0, 0.2, 1) !important;
    }
    body.hide-smart-header div.element-container:has(.campus-header) {
        transform: translateY(-190px) !important;
    }

    .campus-header {
        width: 100%;
        padding: 30px 32px;
        margin-bottom: 12px;
        border-radius: 18px;
        position: relative;
        overflow: hidden;
        background: linear-gradient(95deg, #0b192c 0%, #0e4a66 50%, #0b192c 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: 0 10px 28px rgba(11, 25, 44, 0.22), inset 0 0 38px rgba(6, 182, 212, 0.18);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .campus-header::before {
        content: "";
        position: absolute; left: 10%; top: -60px; width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.36) 0%, transparent 70%);
        pointer-events: none;
    }
    .campus-header::after {
        content: "";
        position: absolute; right: 10%; bottom: -65px; width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.34) 0%, transparent 70%);
        pointer-events: none;
    }
    .header-title {
        position: relative;
        z-index: 2;
        margin: 0;
        color: #ffffff !important;
        font-size: 30px;
        font-weight: 900;
        letter-spacing: 0.4px;
        text-shadow: 0 0 16px rgba(125, 211, 252, 0.55), 0 0 28px rgba(6, 182, 212, 0.35);
        line-height: 1.18;
    }
    .header-subtitle {
        position: relative;
        z-index: 2;
        margin-top: 6px;
        color: #bae6fd !important;
        font-size: 13.5px;
        font-weight: 500;
        letter-spacing: 0.9px;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    /* ---- Bento Card System with Hover Lift ---- */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .bento-marker) {
        background: linear-gradient(180deg, #ffffff 0%, #f9fcff 100%) !important;
        border: 1px solid #dbeafe !important;
        border-top: 4px solid #0891b2 !important;
        border-radius: 18px !important;
        padding: 10px 16px 16px 16px !important;
        box-shadow: 0 8px 22px rgba(14, 116, 144, 0.08) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .bento-marker):hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(14, 116, 144, 0.14) !important;
    }

    /* ---- Section Headers with Glowing Dot ---- */
    .section-header {
        display: flex; align-items: center; justify-content: space-between;
        margin: 2px 0 12px 0;
        flex-wrap: wrap;
        gap: 8px;
    }
    .section-header-left {
        display: flex; align-items: center; gap: 10px;
    }
    .section-header .dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: linear-gradient(135deg, #22d3ee, #0d9488);
        box-shadow: 0 0 9px rgba(6, 182, 212, 0.65);
    }
    .section-header span.label {
        font-size: 16.5px; font-weight: 800; color: #0f172a !important;
    }
    .section-chip {
        background: #ecfeff; color: #0e7490 !important;
        border: 1px solid #a5f3fc; border-radius: 999px;
        font-size: 11px; font-weight: 700; padding: 3px 10px;
    }
    .status-pill-ok {
        display: inline-flex; align-items: center; gap: 6px;
        background: #ecfeff; color: #0e7490 !important;
        border: 1px solid #67e8f9; border-radius: 999px;
        font-size: 12px; font-weight: 700; padding: 4px 12px;
    }
    .status-pill-warn {
        display: inline-flex; align-items: center; gap: 6px;
        background: #fef3c7; color: #92400e !important;
        border: 1px solid #fcd34d; border-radius: 999px;
        font-size: 12px; font-weight: 700; padding: 4px 12px;
    }

    /* ---- 3D Hover Metric Summary Cards ---- */
    .stApp [data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fdff 100%) !important;
        border: 1px solid #dbeafe !important;
        border-top: 4px solid #0891b2 !important;
        border-radius: 14px !important;
        padding: 13px 16px !important;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06) !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease !important;
    }
    .stApp [data-testid="stMetric"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 14px 26px rgba(8, 145, 178, 0.18) !important;
    }
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] *,
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* ---- Dropdowns (Selectbox) ---- */
    .stApp [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stApp [data-baseweb="select"] > div:hover {
        border-color: #0891b2 !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.12) !important;
    }
    .stApp [data-baseweb="select"] *,
    .stApp [data-baseweb="select"] input {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    .stApp [data-baseweb="select"] svg { fill: #475569 !important; }
    [data-baseweb="popover"], [data-baseweb="popover"] > div,
    [data-baseweb="menu"], ul[role="listbox"],
    [data-baseweb="popover"] li, [data-baseweb="menu"] li, ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }
    [data-baseweb="popover"] li *, ul[role="listbox"] li * {
        color: #0f172a !important;
    }
    [data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {
        background-color: #e0f2fe !important;
    }

    /* ---- 3D Tactile Primary Buttons ---- */
    .stApp div.stButton > button,
    .stApp [data-testid="stBaseButton-secondary"],
    .stApp [data-testid="stBaseButton-primary"] {
        width: 100%;
        background: linear-gradient(180deg, #0d9488 0%, #0891b2 55%, #0e7490 100%) !important;
        border: none !important;
        border-bottom: 3px solid #155e75 !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 800 !important;
        box-shadow: 0 5px 0 #155e75, 0 8px 18px rgba(8, 145, 178, 0.28) !important;
        transform: translateY(0);
        transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease !important;
    }
    .stApp div.stButton > button,
    .stApp div.stButton > button *,
    .stApp [data-testid="stBaseButton-secondary"] *,
    .stApp [data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
    }
    .stApp div.stButton > button:hover,
    .stApp [data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(180deg, #14b8a6 0%, #06b6d4 55%, #0891b2 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 7px 0 #155e75, 0 14px 24px rgba(8, 145, 178, 0.36) !important;
    }
    .stApp div.stButton > button:active,
    .stApp [data-testid="stBaseButton-secondary"]:active {
        transform: translateY(3px);
        box-shadow: 0 2px 0 #155e75, 0 4px 8px rgba(8, 145, 178, 0.26) !important;
    }

    /* ---- 3D White Download Button ---- */
    div.stDownloadButton > button {
        background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%) !important;
        color: #0e7490 !important;
        border: 1px solid #bae6fd !important;
        border-bottom: 3px solid #0e7490 !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 800 !important;
        width: 100%;
        box-shadow: 0 5px 0 #0e7490, 0 8px 16px rgba(14, 116, 144, 0.12) !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease !important;
    }
    div.stDownloadButton > button * { color: #0e7490 !important; }
    div.stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 0 #0e7490, 0 14px 22px rgba(14, 116, 144, 0.2) !important;
    }
    div.stDownloadButton > button:active {
        transform: translateY(3px);
        box-shadow: 0 2px 0 #0e7490, 0 4px 8px rgba(14, 116, 144, 0.14) !important;
    }

    /* ---- Order Snapshot Mini-Cards (View 1) ---- */
    .snapshot-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin: 12px 0 14px 0;
    }
    .snapshot-item {
        background: linear-gradient(180deg, #ffffff 0%, #f4fbff 100%);
        border: 1px solid #dbeafe;
        border-bottom: 3px solid #7dd3fc;
        border-radius: 12px;
        padding: 10px 13px;
        box-shadow: 0 3px 8px rgba(14, 116, 144, 0.05);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .snapshot-item:hover {
        transform: translateY(-3px);
        border-bottom-color: #0891b2;
        box-shadow: 0 8px 16px rgba(14, 116, 144, 0.13);
    }
    .snap-label {
        color: #64748b; font-size: 10.5px; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px;
    }
    .snap-val {
        color: #0f172a; font-size: 13.5px; font-weight: 800; line-height: 1.3;
    }

    /* ---- Pre-Dispatch Advisor Strip (View 1) ---- */
    .advisor-strip {
        background: radial-gradient(circle at 92% 10%, rgba(6, 182, 212, 0.10), transparent 55%),
                    linear-gradient(135deg, #f0fdfa 0%, #f0f9ff 100%);
        border: 1px solid #a5f3fc;
        border-left: 5px solid #0d9488;
        border-radius: 12px;
        padding: 11px 14px;
        margin-bottom: 14px;
        font-size: 13px;
        color: #0f172a;
        line-height: 1.5;
    }
    .risk-badge {
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font-size: 11px; font-weight: 800; margin-left: 6px;
    }
    .risk-low { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .risk-med { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }

    /* ---- View 2: Integrated Side-by-Side Live Log Panel ---- */
    .log-panel-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 12px;
        height: 545px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .log-panel-box::-webkit-scrollbar { width: 6px; }
    .log-panel-box::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 6px; }
    .log-entry {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0891b2;
        border-radius: 10px;
        padding: 9px 11px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .log-entry:hover {
        transform: translateX(2px);
        box-shadow: 0 4px 10px rgba(14, 116, 144, 0.10);
    }
    
    .log-entry.goal { border-left-color: #16a34a; background: #f0fdf4; }
    .log-entry.wait, .log-entry.retreat { border-left-color: #f59e0b; background: #fffbeb; }
    .log-entry.sidestep { border-left-color: #8b5cf6; background: #f5f3ff; }
    .log-entry.collision { border-left-color: #dc2626; background: #fef2f2; }
    .log-entry-top {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 3px;
    }
    .log-step-tag {
        font-size: 11px; font-weight: 800; color: #0e7490;
        text-transform: uppercase; letter-spacing: 0.4px;
    }
    .log-reward-pos {
        font-size: 11px; font-weight: 800; color: #15803d;
        background: #dcfce7; padding: 1px 7px; border-radius: 999px;
    }
    .log-reward-neg {
        font-size: 11px; font-weight: 800; color: #475569;
        background: #f1f5f9; padding: 1px 7px; border-radius: 999px;
    }
    .log-reward-bad {
        font-size: 11px; font-weight: 800; color: #b91c1c;
        background: #fee2e2; padding: 1px 7px; border-radius: 999px;
    }
    .log-msg {
        font-size: 12.5px; font-weight: 600; color: #1e293b; line-height: 1.35;
    }
    .log-empty {
        margin: auto; text-align: center; color: #64748b;
        font-size: 13px; padding: 20px;
    }

    /* ---- Live Obstacle Pills (Below Map inside Same Box) ---- */
    .obstacle-pill-grid {
        display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px;
    }
    .obstacle-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 5px 11px;
        font-size: 12px;
        color: #1e293b;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .obstacle-pill.car { border-left-color: #ef4444; }

    /* ---- Hero Result Card (View 2 Final Outcome) ---- */
    .hero-result-card {
        background: radial-gradient(circle at 88% 0%, rgba(6, 182, 212, 0.10), transparent 45%),
                    linear-gradient(180deg, #ffffff 0%, #f8fdff 100%);
        border: 1px solid #bae6fd;
        border-radius: 18px;
        padding: 20px 26px;
        margin: 14px 0 16px 0;
        box-shadow: 0 4px 0 #bae6fd, 0 12px 26px rgba(14, 116, 144, 0.13);
        position: relative;
        overflow: hidden;
    }
    .hero-result-card::after {
        content: "";
        position: absolute; bottom: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #0d9488, #22d3ee, #0d9488);
        background-size: 200% 100%;
        animation: gradientMove 3.5s linear infinite;
    }
    .hero-result-card.failed {
        border-color: #fecaca;
        box-shadow: 0 4px 0 #fecaca, 0 12px 26px rgba(220, 38, 38, 0.11);
    }
    .hero-result-card.failed::after {
        background: linear-gradient(90deg, #dc2626, #fb7185, #dc2626);
        background-size: 200% 100%;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. ENVIRONMENT, HELPERS & MODEL LOADER
# ==========================================

env = CampusDeliveryEnv(obstacle_mode=OBSTACLE_MODE)

NOTE_STYLES = {
    "info":    ("#e0f2fe", "#0ea5e9", "#0c4a6e"),
    "success": ("#dcfce7", "#16a34a", "#14532d"),
    "warning": ("#fef3c7", "#f59e0b", "#78350f"),
    "error":   ("#fee2e2", "#dc2626", "#7f1d1d"),
}


def note(kind, text):
    bg, border, fg = NOTE_STYLES[kind]
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border};'
        f'border-left:6px solid {border};border-radius:12px;padding:12px 16px;'
        f'margin:6px 0 12px 0;color:{fg};font-weight:600;font-size:13.5px;line-height:1.5;">'
        f'{html.escape(text)}</div>',
        unsafe_allow_html=True
    )


def shortest_path_info(env, start, target, avoid_uncovered=False):
    """BFS shortest hop count between two campus nodes."""
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for nxt in env.graph[node]:
            if nxt in visited:
                continue
            if avoid_uncovered and frozenset({node, nxt}) in env.uncovered_paths:
                continue
            visited.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def lerp_coord(p1, p2, t):
    """Linearly interpolate between two (x, y) coordinates."""
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def arc_offset(p1, p2, t, amp=2.5):
    """Smooth lateral arc offset for SIDESTEP obstacle avoidance."""
    x, y = lerp_coord(p1, p2, t)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    bump = amp * np.sin(np.pi * t)  # 0 at endpoints, peak at midpoint t=0.5
    return (x - dy / length * bump, y + dx / length * bump)


def render_obstacle_pills_html(env):
    """Render live obstacle locations & intents as compact pills."""
    if env.obstacle_mode != "routes":
        peds = ", ".join(env.node_names[n] for n in env.pedestrian_positions)
        cars = ", ".join(env.node_names[n] for n in env.car_positions)
        return (
            f'<div class="obstacle-pill-grid">'
            f'<div class="obstacle-pill">🚶 <b>Pedestrians:</b>&nbsp;{html.escape(peds)}</div>'
            f'<div class="obstacle-pill car">🚗 <b>Cars:</b>&nbsp;{html.escape(cars)}</div>'
            f'</div>'
        )

    intents = env.obstacle_intents()
    pills = []
    for kind, label, icon, css_cls in (
        ("ped", "P", "🚶", ""),
        ("car", "C", "🚗", "car"),
    ):
        for i, it in enumerate(intents[kind]):
            here = html.escape(env.node_names[it["node"]])
            if it["goal"] is not None:
                what = f"➜ {html.escape(env.node_names[it['goal']])}"
            elif it["waiting"]:
                what = "waiting"
            else:
                what = "picking next stop"
            pills.append(
                f'<div class="obstacle-pill {css_cls}">'
                f'{icon} <b>{label}{i + 1}</b>&nbsp;at {here}&nbsp;'
                f'<span style="color:#64748b;">({what})</span></div>'
            )
    return f'<div class="obstacle-pill-grid">{"".join(pills)}</div>'


def render_log_panel_html(log_entries):
    """Render newest-first color-coded simulation log entries inside a scrollable panel."""
    if not log_entries:
        return (
            '<div class="log-panel-box">'
            '<div class="log-empty">Waiting for robot dispatch...<br>'
            '<span style="font-size:11.5px;color:#94a3b8;">Step-by-step decisions will appear here in real time.</span>'
            '</div></div>'
        )

    cards = []
    for entry in reversed(log_entries):
        outcome = entry["outcome"]
        if outcome == "goal":
            cls = "goal"
        elif entry.get("sidestepped") or outcome == "sidestep":
            cls = "sidestep"
        elif outcome in ("wait", "retreat"):
            cls = outcome
        elif outcome in ("ped_collision", "car_collision", "invalid"):
            cls = "collision"
        else:
            cls = ""

        rew = entry["reward"]
        if rew > 0:
            rew_cls = "log-reward-pos"
        elif rew <= -15:
            rew_cls = "log-reward-bad"
        else:
            rew_cls = "log-reward-neg"

        cards.append(
            f'<div class="log-entry {cls}">'
            f'  <div class="log-entry-top">'
            f'    <span class="log-step-tag">Step {entry["step"]}</span>'
            f'    <span class="{rew_cls}">{rew:+} pts</span>'
            f'  </div>'
            f'  <div class="log-msg">{html.escape(entry["text"])}</div>'
            f'</div>'
        )

    return f'<div class="log-panel-box">{"".join(cards)}</div>'


@st.cache_resource
def _load_table_cached(path, mtime):
    """mtime ensures Streamlit auto-reloads whenever you overwrite the .npy file."""
    table = np.load(path)
    expected_shape = (36, 36, 2, 3, 3, 6, 38)

    if table.shape != expected_shape:
        st.warning(
            f"{os.path.basename(path)} shape mismatch. Expected {expected_shape}, "
            f"but received {table.shape}."
        )
        return None
    return table


def load_model_table(model_name):
    """Load the trained 5D policy/Q-table for the selected RL model if present."""
    path = MODEL_FILES.get(model_name)
    if not path or not os.path.exists(path):
        return None
    return _load_table_cached(path, os.path.getmtime(path))


# ==========================================
# 5. 2D CAMPUS MAP RENDERER (WITH SMOOTH INTERPOLATION)
# ==========================================

def _round_box(x, y, w, h, r, fc, ec, lw=1.2):
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                          fc=fc, ec=ec, lw=lw)


def icon_robot(scale=1.0):
    k = scale
    da = DrawingArea(30 * k, 38 * k, 0, 0)
    for cx in (9, 21):
        da.add_artist(Circle((cx * k, 4 * k), 3.6 * k, fc="#334155", ec="white", lw=0.8))
    da.add_artist(_round_box(6 * k, 6 * k, 18 * k, 11 * k, 2.5 * k, "#0e7490", "#164e63"))
    da.add_artist(Rectangle((8 * k, 9 * k), 14 * k, 3 * k, fc="#67e8f9", ec="none"))
    da.add_artist(Rectangle((14 * k, 29 * k), 2 * k, 5 * k, fc="#164e63", ec="none"))
    da.add_artist(Circle((15 * k, 35 * k), 2.4 * k, fc="#f97316", ec="#9a3412", lw=0.8))
    da.add_artist(_round_box(4 * k, 17 * k, 22 * k, 12 * k, 3.5 * k, "#06b6d4", "#164e63", 1.5))
    for cx in (11, 19):
        da.add_artist(Circle((cx * k, 23 * k), 3 * k, fc="white", ec="#164e63", lw=0.8))
        da.add_artist(Circle((cx * k, 23 * k), 1.3 * k, fc="#0f172a", ec="none"))
    return da


def icon_person(scale=1.0):
    k = scale
    da = DrawingArea(22 * k, 33 * k, 0, 0)
    for lx in (6, 12):
        da.add_artist(Rectangle((lx * k, 1 * k), 4 * k, 10 * k, fc="#a16207", ec="#713f12", lw=0.8))
    da.add_artist(_round_box(3.5 * k, 9 * k, 15 * k, 13 * k, 5 * k, "#facc15", "#854d0e", 1.3))
    da.add_artist(Circle((11 * k, 27.5 * k), 5 * k, fc="#fde68a", ec="#854d0e", lw=1.3))
    return da


def icon_car(scale=1.0):
    k = scale
    da = DrawingArea(38 * k, 23 * k, 0, 0)
    da.add_artist(Polygon([(10 * k, 13 * k), (13 * k, 21 * k), (27 * k, 21 * k), (30 * k, 13 * k)],
                          closed=True, fc="#ef4444", ec="#7f1d1d", lw=1.3))
    da.add_artist(Polygon([(13 * k, 14 * k), (15 * k, 19.5 * k), (19.5 * k, 19.5 * k), (19.5 * k, 14 * k)],
                          closed=True, fc="#bae6fd", ec="none"))
    da.add_artist(Polygon([(20.5 * k, 14 * k), (20.5 * k, 19.5 * k), (26 * k, 19.5 * k), (28 * k, 14 * k)],
                          closed=True, fc="#bae6fd", ec="none"))
    da.add_artist(_round_box(2 * k, 6 * k, 34 * k, 8 * k, 3 * k, "#ef4444", "#7f1d1d", 1.3))
    da.add_artist(Circle((33.5 * k, 10 * k), 1.5 * k, fc="#fef08a", ec="none"))
    for cx in (11, 27):
        da.add_artist(Circle((cx * k, 6 * k), 4.2 * k, fc="#1f2937", ec="white", lw=0.8))
        da.add_artist(Circle((cx * k, 6 * k), 1.6 * k, fc="#9ca3af", ec="none"))
    return da


def place_icon(ax, da, x, y, dx=0, dy=0, zorder=10):
    ab = AnnotationBbox(da, (x, y), xybox=(dx, dy), xycoords="data",
                        boxcoords="offset points", frameon=False,
                        box_alignment=(0.5, 0.5), pad=0)
    ab.set_zorder(zorder)
    ax.add_artist(ab)


def draw_campus_background(ax):
    ax.add_patch(Rectangle((-28, 2), 130, 101, facecolor="#b8d99b", edgecolor="none", zorder=0))

    green_spaces = [
        (5, 88, 13, 7), (39, 82, 10, 5), (47, 49, 8, 6),
        (70, 12, 12, 6), (4, 5, 9, 5), (88, 72, 9, 5)
    ]
    for x, y, w, h in green_spaces:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=2",
            facecolor="#a5cf85", edgecolor="none", alpha=0.8, zorder=0.3
        ))

    tree_positions = [
        (4, 83), (20, 65), (25, 57), (47, 75),
        (58, 83), (88, 88), (96, 78), (5, 40),
        (16, 15), (35, 10), (68, 10), (76, 18),
        (94, 17), (96, 56), (72, 58), (51, 57)
    ]
    for x, y in tree_positions:
        ax.add_patch(Circle((x, y), 1.5, facecolor="#388e3c", edgecolor="#246b2b", linewidth=0.7, zorder=0.9))
        ax.add_patch(Circle((x - 0.4, y + 0.4), 0.5, facecolor="#70ad52", edgecolor="none", zorder=1))


def icon_waypoint(node_id, node_name=""):
    name = node_name.lower()
    if "roof top" in name or "rooftop" in name:
        return None

    da = DrawingArea(30, 26, 0, 0)
    canteens = {0, 4, 22, 30}
    pools = {20}
    carparks = {12, 25, 29, 35}
    bus_stops = {24, 32}

    if node_id in canteens or any(w in name for w in ("canteen", "cafe", "yum yum", "red bricks")):
        da.add_artist(Rectangle((5, 5), 20, 10, fc="#d97706", ec="#92400e", lw=0.8))
        da.add_artist(Polygon([(3, 15), (27, 15), (24, 23), (6, 23)], closed=True, fc="#fb923c", ec="#9a3412", lw=0.8))
        for x in (8, 14, 20):
            da.add_artist(Rectangle((x, 15), 3, 7, fc="#fff7ed", ec="none"))

    elif node_id in pools or "swimming pool" in name:
        da.add_artist(FancyBboxPatch((3, 5), 24, 17, boxstyle="round,pad=0.2,rounding_size=3", fc="#38bdf8", ec="#0369a1", lw=1))
        for y in (10, 15):
            da.add_artist(Line2D([7, 12, 17, 22], [y, y + 2, y, y + 2], color="white", linewidth=1))

    elif node_id in carparks or "car park" in name or "carpark" in name:
        da.add_artist(FancyBboxPatch((5, 4), 20, 21, boxstyle="round,pad=0.2,rounding_size=2", fc="#2563eb", ec="#1e40af", lw=0.8))
        da.add_artist(Text(15, 14, "P", ha="center", va="center", fontsize=17, fontweight="bold", color="white"))

    elif node_id in bus_stops or "bus stop" in name:
        da.add_artist(Rectangle((14, 3), 2, 15, fc="#475569", ec="none"))
        da.add_artist(Rectangle((4, 16), 22, 9, fc="#0f766e", ec="#134e4a", lw=0.8))
        da.add_artist(Text(15, 20, "BUS", ha="center", va="center", fontsize=6, fontweight="bold", color="white"))

    elif "basketball" in name:
        da.add_artist(Rectangle((3, 5), 24, 17, fc="#e9a15b", ec="#92400e", lw=1))
        da.add_artist(Line2D([15, 15], [5, 22], color="white", linewidth=0.8))
        da.add_artist(Circle((15, 13.5), 3.5, fc="none", ec="white", lw=0.8))

    elif "library" in name:
        for x, color in ((5, "#2563eb"), (13, "#16a34a"), (21, "#f59e0b")):
            da.add_artist(Rectangle((x, 5), 5 if x == 21 else 7, 18, fc=color, ec="#334155", lw=0.8))

    elif "hostel" in name:
        da.add_artist(Rectangle((4, 7), 23, 3, fc="#64748b", ec="#334155", lw=0.8))
        da.add_artist(Rectangle((5, 10), 21, 8, fc="#38bdf8", ec="#0369a1", lw=0.8))
        da.add_artist(Rectangle((6, 14), 7, 5, fc="#f8fafc", ec="#64748b", lw=0.7))

    else:
        da.add_artist(Rectangle((5, 5), 20, 15, fc="#c4b49a", ec="#78664d", lw=0.9))
        da.add_artist(Polygon([(3, 20), (15, 25), (27, 20)], closed=True, fc="#8b7355", ec="#78664d", lw=0.8))
        for x in (9, 18):
            da.add_artist(Rectangle((x, 9), 5, 7, fc="#e0f2fe", ec="#64748b", lw=0.6))

    return da


def render_campus_map(env, visited_nodes, current_node, target_node,
                      score, step_num, status_text, show_obstacles=True,
                      prev_positions=None, robot_xy=None, obs_xy=None,
                      extend_trail=True):

    is_rainy = env.current_weather == 1
    bg_color = "#e0f2fe" if is_rainy else "#f8fafc"

    fig, ax = plt.subplots(figsize=(13, 8.2), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    draw_campus_background(ax)

    # Road connections
    drawn_edges = set()
    for u, neighbors in env.graph.items():
        for v in neighbors:
            edge = frozenset({u, v})
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            x1, y1 = env.node_coords[u]
            x2, y2 = env.node_coords[v]
            if edge in env.uncovered_paths:
                ax.plot([x1, x2], [y1, y2], color="#eab308", linestyle="--", linewidth=2.6, alpha=0.9, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], color="#64748b", linestyle="-", linewidth=2.2, alpha=0.8, zorder=1)

    # Robot's travelled route
    for i in range(len(visited_nodes) - 1):
        x1, y1 = env.node_coords[visited_nodes[i]]
        x2, y2 = env.node_coords[visited_nodes[i + 1]]
        ax.plot([x1, x2], [y1, y2], color="#06b6d4", linewidth=5, alpha=0.95, zorder=3)

    # Smoothly extend cyan trail up to the gliding robot position
    if robot_xy is not None and extend_trail and visited_nodes:
        lx, ly = env.node_coords[visited_nodes[-1]]
        ax.plot([lx, robot_xy[0]], [ly, robot_xy[1]], color="#06b6d4", linewidth=5, alpha=0.95, zorder=3)

    # Waypoint icons and labels
    for node_id, (x, y) in env.node_coords.items():
        name = env.node_names[node_id]
        name_lower = name.lower()

        if node_id == current_node:
            label_color = "#0891b2"
        elif node_id == target_node:
            label_color = "#059669"
        elif node_id in env.start_locations:
            label_color = "#c2410c"
        else:
            label_color = "#334155"

        marker = icon_waypoint(node_id, name)
        label_offsets = {
            0: (-30, -28), 1: (12, 18), 2: (-30, 20),
            3: (12, -28), 4: (-30, 18), 5: (12, 20),
        }
        dx, dy = label_offsets.get(node_id, (12, 18))

        if "bus stop" in name_lower:
            if marker is not None:
                place_icon(ax, marker, x, y, dx, dy, zorder=7)
            continue

        if marker is None:
            label_dx, label_dy = 12, 18
        else:
            place_icon(ax, marker, x, y, dx, dy, zorder=7)
            label_dx = dx + 13 if dx < 0 else dx + 17
            label_dy = dy

        ax.annotate(
            name, (x, y), xytext=(label_dx, label_dy), textcoords="offset points",
            ha="left", va="center", fontsize=6.5, fontweight="bold",
            color=label_color, zorder=8
        )

    # Moving obstacles (supports interpolated coordinates)
    if show_obstacles:
        for kind, positions, make_icon, base_dx, base_dy, arrow_color, tag in (
            ("ped", env.pedestrian_positions, icon_person, -15, -12, "#a16207", "P"),
            ("car", env.car_positions, icon_car, 16, -10, "#b91c1c", "C"),
        ):
            for i, node in enumerate(positions):
                if obs_xy and kind in obs_xy and i < len(obs_xy[kind]):
                    x, y = obs_xy[kind][i]
                else:
                    x, y = env.node_coords[node]

                dx, dy = base_dx + 11 * i, base_dy
                if prev_positions and i < len(prev_positions[kind]) and prev_positions[kind][i] != node:
                    px, py = env.node_coords[prev_positions[kind][i]]
                    if abs(x - px) + abs(y - py) > 0.6:
                        ax.annotate("", xy=(x, y), xytext=(px, py),
                                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.2, shrinkA=7, shrinkB=9),
                                    zorder=8)
                place_icon(ax, make_icon(), x, y, dx, dy, zorder=9)
                ax.annotate(f"{tag}{i + 1}", (x, y), xytext=(dx, dy - 22),
                            textcoords="offset points", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="#0f172a",
                            bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none", alpha=0.85),
                            zorder=10)

    # Robot icon (supports smooth gliding coordinates)
    rx, ry = robot_xy if robot_xy is not None else env.node_coords[current_node]
    place_icon(ax, icon_robot(), rx, ry, 0, 9, zorder=12)

    # HUD overlay
    weather_banner = "RAINY" if is_rainy else "SUNNY"
    time_banner = ["DAY", "EVENING", "NIGHT"][env.current_time_of_day]
    obstacle_text = ""
    if show_obstacles:
        obstacle_text = (f"Pedestrians : {env.pedestrian_positions}\n"
                         f"Cars        : {env.car_positions}\n")

    hud_text = (
        "LIVE DELIVERY ENVIRONMENT\n"
        "──────────────────────────\n"
        f"Score       : {score} pts\n"
        f"Steps Taken : {step_num}\n"
        f"Weather     : {weather_banner}\n"
        f"Time        : {time_banner}\n"
        f"Current Loc : {env.node_names[current_node]}\n"
        f"Target Dest : {env.node_names[target_node]}\n"
        f"Order Item  : {env.selected_food}\n"
        f"{obstacle_text}"
        f"Status      : {status_text}"
    )

    ax.text(-27, 98.5, hud_text, fontsize=8.5, color="#0f172a",
            family="monospace", va="top", ha="left", zorder=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff",
                      edgecolor="#06b6d4", linewidth=2, alpha=0.95))

    # Icon key
    ax.add_patch(FancyBboxPatch((-27.5, 50), 21, 22, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc="#ffffff", ec="#cbd5e1", lw=1.2, alpha=0.95, zorder=11))
    ax.text(-26.5, 70.2, "ICON KEY", fontsize=8, fontweight="bold",
            color="#0f172a", va="center", zorder=12)
    for row_y, make_icon, label in ((63.5, icon_robot, "Robot"),
                                    (57.5, icon_person, "Pedestrian"),
                                    (52.5, icon_car, "Car")):
        place_icon(ax, make_icon(0.55), -23.5, row_y, 0, 0, zorder=12)
        ax.text(-19, row_y, label, fontsize=8, color="#1e293b", va="center", ha="left", zorder=12)

    # Legend
    legend_elements = [
        patches.Patch(facecolor="#10b981", edgecolor="#047857", label="Destination"),
        patches.Patch(facecolor="#f97316", edgecolor="#c2410c", label="Restaurant"),
        Line2D([0], [0], color="#06b6d4", lw=4, label="Robot Travelled Path"),
        Line2D([0], [0], color="#64748b", lw=2, label="Covered Walkway"),
        Line2D([0], [0], color="#eab308", lw=2, ls="--", label="Uncovered Path"),
        Line2D([0], [0], color="#b91c1c", lw=2, marker=">", markersize=6, label="Obstacle's last move"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", facecolor="#ffffff",
              edgecolor="#cbd5e1", labelcolor="#1e293b", fontsize=8)

    ax.set_xlim(-28, 102)
    ax.set_ylim(2, 103)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ==========================================
# 6. ACTION POLICY & NAVIGATION CALLBACKS
# ==========================================

def choose_action(env, state, model_name, policy_table, previous_node,
                  consecutive_waits=0, max_consecutive_waits=3):
    """Use the loaded 6D table; block pointless WAIT on clear roads or long wait loops."""
    valid = env.valid_actions()

    if policy_table is not None:
        candidates = list(valid)

        if 36 in candidates and len(candidates) > 1:
            if state[4] == 0 or consecutive_waits >= max_consecutive_waits:
                candidates.remove(36)

        if USE_ANTI_BACKTRACKING and previous_node is not None:
            filtered = [a for a in candidates if a != previous_node or a in (36, 37)]
            if filtered:
                candidates = filtered

        values = policy_table[state][candidates]
        best = np.flatnonzero(values == values.max())
        return int(candidates[int(random.choice(best.tolist()))])

    return random.choice(valid)

def go_to_order():
    st.session_state.page = "order"
    st.session_state.sim_finished = False
    st.session_state.sim_data = None
    st.session_state.show_result_popup = False


def replay_order():
    st.session_state.sim_finished = False
    st.session_state.sim_data = None
    st.session_state.show_result_popup = False


if "page" not in st.session_state:
    st.session_state.page = "order"


# ==========================================
# 7. SMART STICKY CENTERED HEADER (NO EMOJI)
# ==========================================

st.markdown("""
<div class="campus-header">
    <div class="header-title">TAR UMT Autonomous Campus Food Delivery</div>
    <div class="header-subtitle">Reinforcement Learning Routing  •  Smart Campus</div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# VIEW 1: CUSTOMER ORDER SELECTION (SINGLE-SCREEN)
# ==========================================

if st.session_state.page == "order":

    col_form, col_map = st.columns([1.05, 1.35], gap="large")

    # ---- LEFT BENTO CARD: ALL 6 CONTROLS IN ONE UNIFIED BOX ----
    with col_form:
        with st.container(border=True):
            st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div class="section-header">
                    <div class="section-header-left">
                        <span class="dot"></span>
                        <span class="label">Order & Simulation Setup</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            target_options = {env.node_names[n]: n for n in env.target_destinations}
            start_options = {env.node_names[n]: n for n in env.start_locations}

            # Row 1: Restaurant & Destination
            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                selected_start_name = st.selectbox("🏪 Restaurant", list(start_options.keys()))
                chosen_start = start_options[selected_start_name]
            with r1_c2:
                selected_target_name = st.selectbox("🎯 Destination", list(target_options.keys()))
                chosen_target = target_options[selected_target_name]

            # Row 2: Food Item & RL Model
            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                chosen_food = st.selectbox("🍱 Food Item", env.food_menus[chosen_start])
            with r2_c2:
                selected_model = st.selectbox(
                    "🤖 RL Navigation Model",
                    ["Q-Learning", "SARSA", "A2C", "REINFORCE", "DQN"]
                )

            # Row 3: Weather & Time of Day
            r3_c1, r3_c2 = st.columns(2)
            with r3_c1:
                weather_mode = st.selectbox(
                    "🌦️ Weather Condition",
                    ["🎲 Auto Random", "☀️ Force Sunny", "🌧️ Force Rainy"]
                )
            with r3_c2:
                time_mode = st.selectbox(
                    "🕒 Time of Day",
                    ["🎲 Auto Random", "☀️ Day", "🌆 Evening", "🌙 Night"]
                )

            # Live Pre-Dispatch Route & Condition Analytics (Ready for SL ETA integration)
            is_preview_rain = (weather_mode == "🌧️ Force Rainy")
            min_hops = shortest_path_info(env, chosen_start, chosen_target, avoid_uncovered=False)
            sheltered_hops = shortest_path_info(env, chosen_start, chosen_target, avoid_uncovered=True)
            est_hops = (sheltered_hops if (is_preview_rain and sheltered_hops) else min_hops) or 0

            if time_mode == "🌆 Evening" or is_preview_rain:
                risk_label, risk_cls = "Moderate-High Traffic", "risk-med"
                est_steps_str = f"~{est_hops + 2} – {est_hops + 5} steps"
            elif time_mode == "🌙 Night":
                risk_label, risk_cls = "Low Traffic • Dim Light", "risk-low"
                est_steps_str = f"~{est_hops + 1} – {est_hops + 3} steps"
            else:
                risk_label, risk_cls = "Normal Campus Flow", "risk-low"
                est_steps_str = f"~{est_hops + 1} – {est_hops + 4} steps"

            st.markdown(f"""
                <div class="advisor-strip">
                    <b>Dispatch Advisor:</b>
                    {"Rainy mode active — robot prioritises covered walkways (-40 open-air penalty)." if is_preview_rain else "Standard routing active — optimising shortest safe path across campus."}
                    <span class="risk-badge {risk_cls}">{risk_label}</span>
                </div>
            """, unsafe_allow_html=True)

            if st.button("Confirm Order & Launch Live Tracking", use_container_width=True):
                if weather_mode == "☀️ Force Sunny":
                    weather_value = 0
                elif weather_mode == "🌧️ Force Rainy":
                    weather_value = 1
                else:
                    weather_value = random.choice([0, 1])

                order_config = {
                    "start": chosen_start,
                    "target": chosen_target,
                    "food": chosen_food,
                    "weather": weather_value,
                }
                time_mapping = {"☀️ Day": 0, "🌆 Evening": 1, "🌙 Night": 2}
                if time_mode in time_mapping:
                    order_config["time_of_day"] = time_mapping[time_mode]

                st.session_state.order_config = order_config
                st.session_state.selected_model = selected_model
                st.session_state.sim_finished = False
                st.session_state.sim_data = None
                st.session_state.show_result_popup = False
                st.session_state.page = "tracking"
                st.rerun()

    # ---- RIGHT BENTO CARD: CAMPUS MAP PREVIEW ----
    with col_map:
        with st.container(border=True):
            st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div class="section-header">
                    <div class="section-header-left">
                        <span class="dot"></span>
                        <span class="label">Live Campus Map Preview</span>
                    </div>
                    <span class="section-chip">36 Nodes • 2D Topology</span>
                </div>
            """, unsafe_allow_html=True)

            preview_env = CampusDeliveryEnv(obstacle_mode=OBSTACLE_MODE)
            preview_env.current_location = chosen_start
            preview_env.target_destination = chosen_target
            preview_env.current_weather = 1 if weather_mode == "🌧️ Force Rainy" else 0
            preview_time_map = {"☀️ Day": 0, "🌆 Evening": 1, "🌙 Night": 2}
            preview_env.current_time_of_day = preview_time_map.get(time_mode, 0)
            preview_env.selected_food = chosen_food

            preview_fig = render_campus_map(
                preview_env, [chosen_start], chosen_start, chosen_target,
                score=0, step_num=0, status_text="Waiting for Order",
                show_obstacles=False
            )
            st.pyplot(preview_fig, use_container_width=True)
            plt.close(preview_fig)


# ==========================================
# VIEW 2: UNIFIED LIVE TRACKING CONSOLE
# ==========================================

elif st.session_state.page == "tracking":

    if "order_config" not in st.session_state:
        st.session_state.page = "order"
        st.rerun()

    cfg = st.session_state.order_config
    selected_model = st.session_state.get("selected_model", "Q-Learning")
    policy_table = load_model_table(selected_model)

    # ==========================================
    # SUCCESS POP-UP (centered dialog): "Delivery Successful!" + final score,
    # and a "View Summary" button that opens the details.
    # Needs Streamlit >= 1.37 (st.dialog) or 1.34-1.36 (st.experimental_dialog); otherwise it is skipped.
    # ==========================================
    POPUP_ON_SUCCESS = True
    _dialog_decorator = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

    show_result_dialog = None
    if _dialog_decorator is not None:
        # 用不可见字符代替标题，去掉 "Delivery Result"
        @_dialog_decorator("\u200b")
        def show_result_dialog(d, model_name):
            st.markdown(f"""
                <style>
                    /* 紧凑弹窗顶部间距，去掉标题占用的多余空白 */
                    div[role="dialog"] header {{
                        padding-bottom: 0 !important;
                        min-height: 24px !important;
                    }}
                </style>
                <div style="text-align:center; padding: 6px 10px 22px 10px;">
                    <div style="
                        width: 60px; height: 60px; margin: 0 auto 14px auto;
                        border-radius: 50%;
                        background: linear-gradient(135deg, #14b8a6, #0891b2);
                        color: #ffffff; font-size: 30px; font-weight: 900;
                        display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 8px 20px rgba(8, 145, 178, 0.28);
                    ">✓</div>
                    <div style="color:#0f172a; font-size:24px; font-weight:900; letter-spacing:-0.3px;">
                        Delivery Successful
                    </div>
                    <div style="color:#0d9488; font-size:44px; font-weight:900; margin-top:12px; line-height:1.1;">
                        {d["total_score"]} <span style="font-size:22px; font-weight:800;">pts</span>
                    </div>
                    <div style="color:#94a3b8; font-size:11px; font-weight:700; letter-spacing:1.2px; margin-top:4px;">
                        FINAL SCORE
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if st.button("View Summary", key="btn_go_summary", use_container_width=True):
                st.session_state.page = "summary"
                st.rerun()

    # Compact Model Status Pill (supports all 4 linked tabular RL models)
    if policy_table is not None:
        model_pill_html = (
            f'<span class="status-pill-ok">Model: {html.escape(selected_model)} (Trained 6D Table Loaded)</span>'
        )
    elif selected_model in MODEL_FILES:
        fname = os.path.basename(MODEL_FILES[selected_model])
        model_pill_html = (
            f'<span class="status-pill-warn">{html.escape(fname)} not found — Random Fallback</span>'
        )
    else:
        model_pill_html = (
            f'<span class="status-pill-warn">{html.escape(selected_model)} not linked — Random Fallback</span>'
        )

    # ---- 1. COMPACT TOP BAR ----
    top_col1, top_col2, top_col3 = st.columns([3.8, 0.9, 1.0])
    with top_col1:
        st.markdown(f"""
            <div class="section-header" style="margin:4px 0 6px 0;">
                <div class="section-header-left">
                    <span class="dot"></span>
                    <span class="label" style="font-size:18.5px;">
                        Live Mission: {html.escape(env.node_names[cfg['start']])} ➜ {html.escape(env.node_names[cfg['target']])}
                    </span>
                    {model_pill_html}
                </div>
            </div>
        """, unsafe_allow_html=True)
    with top_col2:
        st.button("Replay", on_click=replay_order, use_container_width=True)
    with top_col3:
        st.button("New Order", on_click=go_to_order, use_container_width=True)

    # ---- 2. MAIN UNIFIED BENTO CONSOLE (MAP + SIMULATION LOG SIDE-BY-SIDE) ----
    with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        col_live_map, col_live_log = st.columns([2.55, 1.05], gap="medium")

        with col_live_map:
            st.markdown("""
                <div class="section-header" style="margin-bottom:6px;">
                    <div class="section-header-left">
                        <span class="dot"></span>
                        <span class="label">Live 2D Campus Map</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            map_placeholder = st.empty()
            intent_placeholder = st.empty()

        with col_live_log:
            st.markdown("""
                <div class="section-header" style="margin-bottom:6px;">
                    <div class="section-header-left">
                        <span class="dot"></span>
                        <span class="label">Simulation Log</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            log_placeholder = st.empty()

    # ---- 3. LIVE SUMMARY METRICS STRIP (CLEANLY BELOW MAP & LOG) ----
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    summary_placeholder = st.empty()

    def render_summary_cards(total_score, last_reward, step_num, weather_val, tod_val, obs_count):
        time_labels = ["Day", "Evening", "Night"]
        with summary_placeholder.container():
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Current Score", f"{total_score} pts", delta=f"{last_reward:+} pts")
            m2.metric("Steps Taken", f"{step_num}")
            m3.metric("Weather", "Rainy" if weather_val == 1 else "Sunny")
            m4.metric("Time of Day", time_labels[tod_val])
            m5.metric("Active Obstacles", f"{obs_count}")

    # ---- RUN LIVE SMOOTH SIMULATION ----
    if not st.session_state.get("sim_finished", False):
        try:
            state, info = env.reset(options=cfg)
        except ValueError as err:
            note("error", f"Invalid order: {err}")
            st.button("🏠 Back to Order Selection", on_click=go_to_order)
            st.stop()

        total_score = 0
        step_num = 0
        terminated = truncated = False
        previous_node = None
        visited_nodes = [env.current_location]
        log_entries = []
        log_records = []
        retreat_count = 0
        rain_violations = 0
        consecutive_waits = 0

        render_summary_cards(total_score, 0, step_num, env.current_weather, env.current_time_of_day,
                             len(env.pedestrian_positions) + len(env.car_positions))
        fig = render_campus_map(env, visited_nodes, env.current_location,
                                env.target_destination, total_score, step_num, "Order Dispatched")
        map_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        intent_placeholder.markdown(render_obstacle_pills_html(env), unsafe_allow_html=True)
        log_placeholder.markdown(render_log_panel_html(log_entries), unsafe_allow_html=True)
        time.sleep(0.75)

        reward = 0
        status_msg = "Order Dispatched"

        while not (terminated or truncated):
            step_num += 1
            action = choose_action(
                env, state, selected_model, policy_table, previous_node,
                consecutive_waits=consecutive_waits, max_consecutive_waits=3
            )
            previous_location = env.current_location
            prev_positions = {"ped": list(env.pedestrian_positions), "car": list(env.car_positions)}

            state, reward, terminated, truncated, info = env.step(action)
            total_score += reward

            outcome = info.get("outcome")
            sidestepped = info.get("sidestepped", False)
            consecutive_waits = (consecutive_waits + 1) if outcome == "wait" else 0
            if info.get("retreated"):
                retreat_count += 1
            if info.get("rain_violation"):
                rain_violations += 1

            moved = env.current_location != previous_location
            here = env.node_names[env.current_location]

            if outcome == "wait":
                status_msg, log_line = "Robot pulled over (WAIT)", "WAIT (Safe pull-over)"
            elif sidestepped and outcome != "goal":
                status_msg = f"Dodged obstacle -> {here}"
                log_line = f"SIDESTEP ➜ {here} (Dodged obstacle)"
            elif outcome == "retreat":
                blocked_idx = info.get("blocked_node")
                blocked = env.node_names[blocked_idx] if blocked_idx is not None else "obstacle"
                status_msg = f"Blocked at {blocked} -> Retreat"
                log_line = f"RETREAT — blocked at {blocked}"
            elif outcome == "invalid":
                status_msg, log_line = "Invalid action", "INVALID action"
            elif outcome == "ped_collision":
                status_msg, log_line = f"Pedestrian collision at {here}", f"MOVE ➜ {here} (Pedestrian Collision)"
            elif outcome == "car_collision":
                status_msg, log_line = f"Car collision at {here}", f"MOVE ➜ {here} (Car Collision)"
            elif outcome == "goal":
                if sidestepped:
                    status_msg, log_line = f"Dodged & Delivered at {here}!", f"SIDESTEP ➜ {here} (Dodged & Delivered!)"
                else:
                    status_msg, log_line = f"Delivered at {here}!", f"MOVE ➜ {here} (Delivered!)"
            else:
                status_msg, log_line = f"Moved to {here}", f"MOVE ➜ {here}"

            # Update log panel & summary cards right as the step begins
            log_entries.append({
                "step": step_num,
                "text": log_line,
                "reward": reward,
                "outcome": outcome,
                "sidestepped": sidestepped,
            })
            log_records.append({
                "Step": step_num,
                "From": env.node_names[previous_location],
                "To": here,
                "Outcome": "sidestep_goal" if (sidestepped and outcome == "goal") else outcome,
                "Reward": reward,
                "Cumulative_Score": total_score,
                "Weather": info["weather_str"],
                "Time_of_Day": info["time_of_day_str"],
                "Traffic": info["traffic_str"],
                "Rain_Violation": info["rain_violation"],
            })

            log_placeholder.markdown(render_log_panel_html(log_entries), unsafe_allow_html=True)
            render_summary_cards(total_score, reward, step_num, env.current_weather, env.current_time_of_day,
                                 len(env.pedestrian_positions) + len(env.car_positions))

            # ---- SMOOTH GLIDING ANIMATION ACROSS INTERP_FRAMES ----
            start_xy = env.node_coords[previous_location]
            end_xy = env.node_coords[env.current_location]
            blocked_idx = info.get("blocked_node")
            blocked_xy = env.node_coords[blocked_idx] if (outcome == "retreat" and blocked_idx is not None) else None

            for f in range(1, INTERP_FRAMES + 1):
                t = f / INTERP_FRAMES

                # Robot position: curved arc on sidestep, straight glide on move, bounce on retreat
                if sidestepped:
                    curr_robot_xy = arc_offset(start_xy, end_xy, t)
                    extend_trail = True
                elif moved:
                    curr_robot_xy = lerp_coord(start_xy, end_xy, t)
                    extend_trail = True
                elif blocked_xy is not None:
                    bounce_t = 0.42 * (1.0 - abs(2.0 * t - 1.0))
                    curr_robot_xy = lerp_coord(start_xy, blocked_xy, bounce_t)
                    extend_trail = False
                else:
                    curr_robot_xy = end_xy
                    extend_trail = False

                # Obstacles glide smoothly along their edges too
                curr_obs_xy = {
                    "ped": [
                        lerp_coord(env.node_coords[prev_positions["ped"][i]], env.node_coords[node], t)
                        for i, node in enumerate(env.pedestrian_positions)
                    ],
                    "car": [
                        lerp_coord(env.node_coords[prev_positions["car"][i]], env.node_coords[node], t)
                        for i, node in enumerate(env.car_positions)
                    ],
                }

                if f < INTERP_FRAMES:
                    if sidestepped:
                        frame_status = f"Sidestepping obstacle to {here}..."
                    elif moved:
                        frame_status = f"Moving to {here}..."
                    else:
                        frame_status = status_msg
                else:
                    frame_status = status_msg

                fig = render_campus_map(
                    env, visited_nodes, env.current_location, env.target_destination,
                    total_score, step_num, frame_status,
                    prev_positions=prev_positions,
                    robot_xy=curr_robot_xy,
                    obs_xy=curr_obs_xy,
                    extend_trail=extend_trail
                )
                map_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)
                if f < INTERP_FRAMES:
                    time.sleep(FRAME_DELAY)

            if moved:
                previous_node = previous_location
                visited_nodes.append(env.current_location)

            intent_placeholder.markdown(render_obstacle_pills_html(env), unsafe_allow_html=True)
            time.sleep(STEP_PAUSE)

        # Save snapshot so Export CSV does not replay the animation
        st.session_state.sim_data = {
            "total_score": total_score,
            "last_reward": reward,
            "step_num": step_num,
            "final_outcome": info.get("outcome"),
            "truncated": truncated,
            "current_location": env.current_location,
            "target_destination": env.target_destination,
            "current_weather": env.current_weather,
            "current_time_of_day": env.current_time_of_day,
            "selected_food": env.selected_food,
            "pedestrian_positions": list(env.pedestrian_positions),
            "car_positions": list(env.car_positions),
            "visited_nodes": visited_nodes,
            "status_msg": status_msg,
            "log_entries": log_entries,
            "log_records": log_records,
            "retreat_count": retreat_count,
            "rain_violations": rain_violations,
            "obstacle_pills_html": render_obstacle_pills_html(env),
        }
        st.session_state.sim_finished = True
       
        if info.get("outcome") == "goal" and POPUP_ON_SUCCESS:
            st.session_state.show_result_popup = True
    else:
        # Restore finished state cleanly on rerun (e.g., after Export CSV)
        d = st.session_state.sim_data
        env.current_location = d["current_location"]
        env.target_destination = d["target_destination"]
        env.current_weather = d["current_weather"]
        env.current_time_of_day = d["current_time_of_day"]
        env.selected_food = d["selected_food"]
        env.pedestrian_positions = list(d["pedestrian_positions"])
        env.car_positions = list(d["car_positions"])

        fig = render_campus_map(env, d["visited_nodes"], d["current_location"],
                                d["target_destination"], d["total_score"],
                                d["step_num"], d["status_msg"])
        map_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        intent_placeholder.markdown(d["obstacle_pills_html"], unsafe_allow_html=True)
        log_placeholder.markdown(render_log_panel_html(d["log_entries"]), unsafe_allow_html=True)
        render_summary_cards(d["total_score"], d["last_reward"], d["step_num"],
                             d["current_weather"], d["current_time_of_day"],
                             len(d["pedestrian_positions"]) + len(d["car_positions"]))

    # ======================================
    # 4. HERO RESULT CARD & CSV EXPORT
    # ======================================
    d = st.session_state.sim_data
    final_outcome = d["final_outcome"]

    if final_outcome == "goal":
        hero_cls = ""
        hero_title = "Delivery Completed Successfully"
        hero_sub = (
            f"Delivered <b>{html.escape(d['selected_food'])}</b> to "
            f"<b>{html.escape(env.node_names[d['target_destination']])}</b> "
            f"using <b>{html.escape(selected_model)}</b>."
        )
        accent_col = "#0d9488"
    elif final_outcome in ("ped_collision", "car_collision"):
        kind = "pedestrian" if final_outcome == "ped_collision" else "car"
        hero_cls = "failed"
        hero_title = f"Delivery Failed — {kind.title()} Collision"
        hero_sub = f"Robot collided with a {kind} at <b>{html.escape(env.node_names[d['current_location']])}</b>."
        accent_col = "#dc2626"
    else:
        hero_cls = "failed"
        hero_title = "Time Limit Reached"
        hero_sub = f"Simulation ended after reaching the {env.max_steps}-step horizon."
        accent_col = "#d97706"

    rain_status = "100% Dry" if d["rain_violations"] == 0 else f"{d['rain_violations']} Open-Air"

    st.markdown(f"""
    <div class="hero-result-card {hero_cls}">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:18px;">
            <div>
                <div style="color:{accent_col}; font-size:21px; font-weight:900; margin-bottom:4px;">{hero_title}</div>
                <div style="color:#475569; font-size:14px;">{hero_sub}</div>
            </div>
            <div style="display:flex; gap:26px; flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="color:{accent_col}; font-size:23px; font-weight:900;">{d['total_score']} pts</div>
                    <div style="color:#64748b; font-size:10.5px; font-weight:700; letter-spacing:0.5px;">FINAL SCORE</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:{accent_col}; font-size:23px; font-weight:900;">{d['step_num']}</div>
                    <div style="color:#64748b; font-size:10.5px; font-weight:700; letter-spacing:0.5px;">STEPS</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:{accent_col}; font-size:23px; font-weight:900;">{d['retreat_count']}</div>
                    <div style="color:#64748b; font-size:10.5px; font-weight:700; letter-spacing:0.5px;">RETREATS</div>
                </div>
                <div style="text-align:center;">
                    <div style="color:{accent_col}; font-size:23px; font-weight:900;">{rain_status}</div>
                    <div style="color:#64748b; font-size:10.5px; font-weight:700; letter-spacing:0.5px;">RAIN SAFETY</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    csv_bytes = pd.DataFrame(d["log_records"]).to_csv(index=False).encode("utf-8")
    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        st.button("🏠 Back to Order Selection", on_click=go_to_order, use_container_width=True)
    with btn_c2:
        st.download_button(
            label="📥 Export Delivery Log (CSV)",
            data=csv_bytes,
            file_name=f"delivery_log_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ---- Open the centered success pop-up (only once, right after the delivery finishes) ----
    if st.session_state.pop("show_result_popup", False) and show_result_dialog is not None:
        show_result_dialog(d, selected_model)

# ==========================================
# VIEW 3: DELIVERY SUMMARY PAGE
# ==========================================

elif st.session_state.page == "summary":

    if not st.session_state.get("sim_data"):
        st.session_state.page = "order"
        st.rerun()

    d = st.session_state.sim_data
    selected_model = st.session_state.get("selected_model", "Q-Learning")
    time_labels = ["Day", "Evening", "Night"]
    weather_txt = "Rainy" if d["current_weather"] == 1 else "Sunny"
    waits = sum(1 for r in d["log_records"] if r["Outcome"] == "wait")
    sidesteps = sum(1 for e in d["log_entries"] if e.get("sidestepped"))
    rain_txt = "100% Dry" if d["rain_violations"] == 0 else f"{d['rain_violations']} Open-Air"
    start_name = env.node_names[d["visited_nodes"][0]]
    target_name = env.node_names[d["target_destination"]]
    route_str = " ➜ ".join(env.node_names[n] for n in d["visited_nodes"])

    # Top Title & Back to Map
    top_c1, top_c2 = st.columns([4.2, 1])
    with top_c1:
        st.markdown("""
            <div class="section-header" style="margin:4px 0 8px 0;">
                <div class="section-header-left">
                    <span class="dot"></span>
                    <span class="label" style="font-size:20px;">Delivery Summary</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with top_c2:
        if st.button("← Back to Map", use_container_width=True):
            st.session_state.page = "tracking"
            st.rerun()

    # Hero Score Banner
    st.markdown(f"""
        <div class="hero-result-card">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:18px;">
                <div>
                    <div style="color:#0d9488; font-size:24px; font-weight:900; margin-bottom:4px;">
                        Delivery Successful
                    </div>
                    <div style="color:#475569; font-size:14px;">
                        <b>{html.escape(d['selected_food'])}</b> delivered from
                        <b>{html.escape(start_name)}</b> to <b>{html.escape(target_name)}</b>
                        via <b>{html.escape(selected_model)}</b>.
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="color:#0d9488; font-size:34px; font-weight:900; line-height:1;">
                        {d['total_score']} pts
                    </div>
                    <div style="color:#64748b; font-size:11px; font-weight:700; letter-spacing:0.8px; margin-top:4px;">
                        FINAL SCORE
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 5 Key Metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Steps Taken", d["step_num"])
    k2.metric("Sidesteps", sidesteps)
    k3.metric("Retreats", d["retreat_count"])
    k4.metric("Waits", waits)
    k5.metric("Rain Safety", rain_txt)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Mission Details & Route Bento Card
    with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="section-header">
                <div class="section-header-left">
                    <span class="dot"></span>
                    <span class="label">Mission Breakdown</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="snapshot-grid">
                <div class="snapshot-item">
                    <div class="snap-label">Order Item</div>
                    <div class="snap-val">{html.escape(d["selected_food"])}</div>
                </div>
                <div class="snapshot-item">
                    <div class="snap-label">AI Model</div>
                    <div class="snap-val">{html.escape(selected_model)}</div>
                </div>
                <div class="snapshot-item">
                    <div class="snap-label">Mission Route</div>
                    <div class="snap-val">{html.escape(start_name)} ➜ {html.escape(target_name)}</div>
                </div>
                <div class="snapshot-item">
                    <div class="snap-label">Environment Conditions</div>
                    <div class="snap-val">{weather_txt} · {time_labels[d["current_time_of_day"]]}</div>
                </div>
            </div>
            <div class="advisor-strip" style="margin-bottom:6px;">
                <b>Full Path Taken:</b> {html.escape(route_str)}
            </div>
        """, unsafe_allow_html=True)

    # Step-by-Step Table
    with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="section-header">
                <div class="section-header-left">
                    <span class="dot"></span>
                    <span class="label">Step-by-Step Delivery Log</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(d["log_records"]), use_container_width=True, hide_index=True)

    # Bottom Action Buttons
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    csv_bytes = pd.DataFrame(d["log_records"]).to_csv(index=False).encode("utf-8")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔁 Replay Delivery", use_container_width=True):
            st.session_state.page = "tracking"
            replay_order()
            st.rerun()
    with b2:
        st.button("🔄 New Order", on_click=go_to_order, use_container_width=True)
    with b3:
        st.download_button(
            label="📥 Export Log (CSV)",
            data=csv_bytes,
            file_name=f"delivery_log_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
