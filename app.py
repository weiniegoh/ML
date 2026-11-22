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

INTERP_FRAMES = 4                # sub-frames per step for smooth gliding
FRAME_DELAY = 0.14               # seconds per sub-frame (increase to slow down)
STEP_PAUSE = 0.35                # brief pause after arriving at each node
USE_ANTI_BACKTRACKING = False    # set False to match evaluate_agent() in .ipynb
OBSTACLE_MODE = "routes"         # change to "random" if your .ipynb trained on random
Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.npy")


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TAR UMT AI Food Delivery",
    page_icon="🤖",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "order"


# ==========================================
# 2. SMART STICKY HEADER + RIGHT DRAWER JS
# - Hide sticky header on scroll down, reveal on scroll up
# - Inject floating right-edge handle for collapsible Simulation Log drawer
#   (stays open/closed without interrupting the live animation loop)
# ==========================================

is_tracking_js = "true" if st.session_state.page == "tracking" else "false"

interactive_ui_js = f"""
<script>
const parentWin = window.parent;
const parentDoc = window.parent.document;
const isTrackingPage = {is_tracking_js};

// 1. Smart Sticky Header scroll listener (init once)
if (!parentWin._campusSmartHeaderInit) {{
    let lastScrollY = 0;
    let ticking = false;

    const scrollHandler = function(e) {{
        if (!ticking) {{
            parentWin.requestAnimationFrame(function() {{
                let currentScrollY = parentWin.scrollY;
                if (e.target && e.target.scrollTop !== undefined && e.target.tagName !== 'IFRAME') {{
                    currentScrollY = e.target.scrollTop;
                }}

                if (currentScrollY <= 70) {{
                    parentDoc.body.classList.remove('hide-smart-header');
                }} else if (currentScrollY > lastScrollY + 15) {{
                    parentDoc.body.classList.add('hide-smart-header');
                }} else if (currentScrollY < lastScrollY - 15) {{
                    parentDoc.body.classList.remove('hide-smart-header');
                }}
                lastScrollY = currentScrollY;
                ticking = false;
            }});
            ticking = true;
        }}
    }};

    parentDoc.addEventListener('scroll', scrollHandler, true);
    parentWin._campusSmartHeaderInit = true;
}}

// 2. Floating Right Sidebar Toggle Handle for Simulation Log
let btn = parentDoc.getElementById('campus-log-drawer-btn');
if (!btn) {{
    btn = parentDoc.createElement('button');
    btn.id = 'campus-log-drawer-btn';
    btn.type = 'button';
    btn.innerHTML = '◀&nbsp; Live Log';
    btn.onclick = function() {{
        const isOpen = parentDoc.body.classList.toggle('log-drawer-open');
        btn.innerHTML = isOpen ? '▶&nbsp; Hide Log' : '◀&nbsp; Live Log';
    }};
    parentDoc.body.appendChild(btn);
}}

// Show handle only on View 2 (Live Tracking); auto-close drawer on View 1
if (isTrackingPage) {{
    btn.style.display = 'flex';
    const isOpen = parentDoc.body.classList.contains('log-drawer-open');
    btn.innerHTML = isOpen ? '▶&nbsp; Hide Log' : '◀&nbsp; Live Log';
}} else {{
    btn.style.display = 'none';
    parentDoc.body.classList.remove('log-drawer-open');
}}
</script>
"""
components.html(interactive_ui_js, height=0, width=0)


# ==========================================
# 3. ADVANCED CSS (LIGHT MODE + BENTO + RIGHT DRAWER + MODAL)
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
        padding-top: 1.05rem !important;
        padding-bottom: 1.8rem !important;
        max-width: 1460px;
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

    /* ---- Force Light Theme inside Centered Modal Dialog ---- */
    [data-testid="stDialog"] [role="dialog"],
    div[role="dialog"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-radius: 20px !important;
        border: 1px solid #bae6fd !important;
        box-shadow: 0 24px 60px rgba(11, 25, 44, 0.28) !important;
        max-width: 640px !important;
        width: 92vw !important;
    }
    div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3,
    div[role="dialog"] p, div[role="dialog"] span, div[role="dialog"] label {
        color: #0f172a !important;
    }
    div[role="dialog"] button[aria-label="Close"] svg {
        fill: #475569 !important;
        color: #475569 !important;
    }

    /* ---- Smart Sticky Centered Banner (No Emoji) ---- */
    div.element-container:has(.campus-header) {
        position: sticky !important;
        top: 0.7rem !important;
        z-index: 999 !important;
        transition: transform 0.38s cubic-bezier(0.3, 0, 0.2, 1) !important;
    }
    body.hide-smart-header div.element-container:has(.campus-header) {
        transform: translateY(-190px) !important;
    }

    .campus-header {
        width: 100%;
        padding: 20px 32px;
        margin-bottom: 10px;
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
        font-size: 29px;
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
        padding: 10px 16px 14px 16px !important;
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
        margin: 2px 0 10px 0;
        flex-wrap: wrap;
        gap: 8px;
    }
    .section-header-left {
        display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
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
    div.stButton > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {
        width: 100%;
        background: linear-gradient(180deg, #0d9488 0%, #0891b2 55%, #0e7490 100%) !important;
        border: none !important;
        border-bottom: 3px solid #155e75 !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        font-weight: 800 !important;
        box-shadow: 0 5px 0 #155e75, 0 8px 18px rgba(8, 145, 178, 0.28) !important;
        transform: translateY(0);
        transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.2s ease !important;
    }
    div.stButton > button,
    div.stButton > button *,
    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
    }
    div.stButton > button:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(180deg, #14b8a6 0%, #06b6d4 55%, #0891b2 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 7px 0 #155e75, 0 14px 24px rgba(8, 145, 178, 0.36) !important;
    }
    div.stButton > button:active,
    [data-testid="stBaseButton-secondary"]:active {
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
        padding: 10px 18px !important;
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

    /* ---- Live Obstacle Pills (Below Full-Width Map) ---- */
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

    /* ==========================================================
       COLLAPSIBLE RIGHT SIDEBAR DRAWER (SIMULATION LOG)
       ========================================================== */
    #campus-log-drawer-btn {
        position: fixed;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        z-index: 999995;
        display: none;
        align-items: center;
        justify-content: center;
        background: linear-gradient(180deg, #0b192c 0%, #0e7490 100%);
        color: #ffffff;
        border: 1px solid rgba(103, 232, 249, 0.5);
        border-right: none;
        border-radius: 14px 0 0 14px;
        padding: 13px 15px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.4px;
        cursor: pointer;
        box-shadow: -4px 6px 20px rgba(11, 25, 44, 0.25);
        transition: right 0.32s cubic-bezier(0.25, 0.8, 0.25, 1), background 0.2s ease;
    }
    #campus-log-drawer-btn:hover {
        background: linear-gradient(180deg, #0e4a66 0%, #06b6d4 100%);
    }
    body.log-drawer-open #campus-log-drawer-btn {
        right: 360px;
    }

    .right-log-drawer {
        position: fixed;
        top: 0;
        right: 0;
        width: 360px;
        height: 100vh;
        background: #ffffff;
        border-left: 3px solid #0891b2;
        box-shadow: -12px 0 36px rgba(11, 25, 44, 0.18);
        z-index: 999990;
        transform: translateX(100%);
        transition: transform 0.32s cubic-bezier(0.25, 0.8, 0.25, 1);
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
    }
    body.log-drawer-open .right-log-drawer {
        transform: translateX(0);
    }
    .drawer-header {
        background: linear-gradient(95deg, #0b192c 0%, #0e4a66 100%);
        color: #ffffff !important;
        padding: 18px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-shrink: 0;
    }
    .drawer-title {
        color: #ffffff !important;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.3px;
    }
    .drawer-count {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(186, 230, 253, 0.4);
        color: #e0f2fe !important;
        font-size: 11.5px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
    }
    .drawer-body {
        flex: 1;
        overflow-y: auto;
        padding: 14px;
        background: #f8fafc;
        display: flex;
        flex-direction: column;
        gap: 9px;
    }
    .drawer-body::-webkit-scrollbar { width: 6px; }
    .drawer-body::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 6px; }

    .log-entry {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0891b2;
        border-radius: 10px;
        padding: 9px 12px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .log-entry.goal { border-left-color: #16a34a; background: #f0fdf4; }
    .log-entry.wait, .log-entry.retreat { border-left-color: #f59e0b; background: #fffbeb; }
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
        background: #dcfce7; padding: 1px 8px; border-radius: 999px;
    }
    .log-reward-neg {
        font-size: 11px; font-weight: 800; color: #475569;
        background: #f1f5f9; padding: 1px 8px; border-radius: 999px;
    }
    .log-reward-bad {
        font-size: 11px; font-weight: 800; color: #b91c1c;
        background: #fee2e2; padding: 1px 8px; border-radius: 999px;
    }
    .log-msg {
        font-size: 12.8px; font-weight: 600; color: #1e293b; line-height: 1.38;
    }
    .log-empty {
        margin: auto; text-align: center; color: #64748b;
        font-size: 13px; padding: 20px;
    }

    /* ==========================================================
       CENTERED MISSION SUMMARY POPUP CARD STYLING
       ========================================================== */
    .modal-hero {
        text-align: center;
        padding: 18px 16px 14px 16px;
        border-radius: 16px;
        background: radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.14), transparent 70%),
                    linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
        border: 1px solid #a5f3fc;
        margin-bottom: 14px;
    }
    .modal-hero.failed {
        background: radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.12), transparent 70%),
                    linear-gradient(180deg, #fef2f2 0%, #ffffff 100%);
        border-color: #fecaca;
    }
    .modal-status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 800;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .modal-title {
        font-size: 24px;
        font-weight: 900;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }
    .modal-sub {
        font-size: 13.8px;
        color: #475569 !important;
        margin: 0;
    }
    .modal-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 14px 0 14px 0;
    }
    .modal-kpi {
        background: #f8fdff;
        border: 1px solid #dbeafe;
        border-top: 3px solid #0891b2;
        border-radius: 12px;
        padding: 12px 8px;
        text-align: center;
    }
    .modal-kpi-val {
        font-size: 20px;
        font-weight: 900;
        color: #0f172a !important;
        line-height: 1.2;
    }
    .modal-kpi-lbl {
        font-size: 10.5px;
        font-weight: 700;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-top: 4px;
    }
    .modal-meta-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 8px;
        margin-bottom: 16px;
    }
    .modal-meta-pill {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 700;
        color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. ENVIRONMENT, HELPERS & Q-TABLE
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
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


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


def render_right_log_drawer_html(log_entries):
    """Render the collapsible right-sidebar drawer containing newest-first logs."""
    count = len(log_entries)
    if not log_entries:
        body_html = (
            '<div class="log-empty">Waiting for robot dispatch...<br>'
            '<span style="font-size:11.5px;color:#94a3b8;">'
            'Step-by-step robot actions will stream here live.</span></div>'
        )
    else:
        cards = []
        for entry in reversed(log_entries):
            outcome = entry["outcome"]
            if outcome == "goal":
                cls = "goal"
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
        body_html = "".join(cards)

    return (
        f'<div class="right-log-drawer">'
        f'  <div class="drawer-header">'
        f'    <span class="drawer-title">Simulation Log</span>'
        f'    <span class="drawer-count">{count} steps</span>'
        f'  </div>'
        f'  <div class="drawer-body">{body_html}</div>'
        f'</div>'
    )


@st.cache_resource
def load_q_table(path=Q_TABLE_PATH):
    """Load the trained 5D Q-table if it exists and has the expected shape."""
    if not os.path.exists(path):
        return None

    table = np.load(path)
    expected_shape = (36, 36, 2, 3, 3, 37)

    if table.shape != expected_shape:
        st.warning(
            f"Q-table shape mismatch. Expected {expected_shape}, "
            f"but received {table.shape}."
        )
        return None
    return table


# ==========================================
# 5. 2D CAMPUS MAP RENDERER (FULL-WIDTH + SMOOTH GLIDE)
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
                      extend_trail=True, figsize=(13.5, 7.6)):

    is_rainy = env.current_weather == 1
    bg_color = "#e0f2fe" if is_rainy else "#f8fafc"

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
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
            ha="left", va="center", fontsize=6.8, fontweight="bold",
            color=label_color, zorder=8
        )

    # Moving obstacles
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

    # Robot icon
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

def choose_action(env, state, model_name, q_table, previous_node):
    """Trained Q-table for 'Q-Learning'; random valid action otherwise."""
    valid = env.valid_actions()

    if model_name == "Q-Learning" and q_table is not None:
        candidates = valid
        if USE_ANTI_BACKTRACKING and previous_node is not None:
            filtered = [a for a in valid if a != previous_node]
            if filtered:
                candidates = filtered
        q_values = q_table[state][candidates]
        best = np.flatnonzero(q_values == q_values.max())
        return int(candidates[int(random.choice(best.tolist()))])

    return random.choice(valid)


def go_to_order():
    st.session_state.page = "order"
    st.session_state.sim_finished = False
    st.session_state.sim_data = None
    st.session_state.show_result_modal = False


def replay_order():
    st.session_state.sim_finished = False
    st.session_state.sim_data = None
    st.session_state.show_result_modal = False


def open_result_modal():
    st.session_state.show_result_modal = True


# ==========================================
# 7. CENTERED MISSION SUMMARY POPUP MODAL
# ==========================================

def render_mission_summary_content(d, selected_model, in_dialog=True):
    """Render the centered result summary card and action buttons."""
    final_outcome = d["final_outcome"]
    time_labels = ["Day", "Evening", "Night"]
    weather_str = "Rainy" if d["current_weather"] == 1 else "Sunny"
    tod_str = time_labels[d["current_time_of_day"]]
    rain_status = "100% Dry" if d["rain_violations"] == 0 else f"{d['rain_violations']} Open-Air"

    if final_outcome == "goal":
        hero_cls = ""
        badge_bg, badge_fg = "#dcfce7", "#15803d"
        badge_txt = "DELIVERY SUCCESSFUL"
        title_col = "#0d9488"
        title_txt = "Order Delivered Successfully!"
        sub_txt = (
            f"<b>{html.escape(d['selected_food'])}</b> safely arrived at "
            f"<b>{html.escape(env.node_names[d['target_destination']])}</b>."
        )
    elif final_outcome in ("ped_collision", "car_collision"):
        kind = "pedestrian" if final_outcome == "ped_collision" else "car"
        hero_cls = "failed"
        badge_bg, badge_fg = "#fee2e2", "#b91c1c"
        badge_txt = "MISSION FAILED"
        title_col = "#dc2626"
        title_txt = f"Collision with {kind.title()}"
        sub_txt = (
            f"The delivery robot collided with a {kind} at "
            f"<b>{html.escape(env.node_names[d['current_location']])}</b>."
        )
    else:
        hero_cls = "failed"
        badge_bg, badge_fg = "#fef3c7", "#92400e"
        badge_txt = "TIME LIMIT REACHED"
        title_col = "#d97706"
        title_txt = "Delivery Timed Out"
        sub_txt = f"Reached the maximum horizon of {env.max_steps} steps before arrival."

    st.markdown(f"""
        <div class="modal-hero {hero_cls}">
            <div class="modal-status-badge" style="background:{badge_bg};color:{badge_fg} !important;">
                {badge_txt}
            </div>
            <div class="modal-title" style="color:{title_col} !important;">{title_txt}</div>
            <p class="modal-sub">{sub_txt}</p>
        </div>

        <div class="modal-meta-row">
            <span class="modal-meta-pill">Route: {html.escape(env.node_names[d['start_location']])} ➜ {html.escape(env.node_names[d['target_destination']])}</span>
            <span class="modal-meta-pill">Model: {html.escape(selected_model)}</span>
            <span class="modal-meta-pill">{weather_str} · {tod_str}</span>
        </div>

        <div class="modal-kpi-grid">
            <div class="modal-kpi">
                <div class="modal-kpi-val" style="color:{title_col} !important;">{d['total_score']} pts</div>
                <div class="modal-kpi-lbl">Final Score</div>
            </div>
            <div class="modal-kpi">
                <div class="modal-kpi-val">{d['step_num']}</div>
                <div class="modal-kpi-lbl">Steps Taken</div>
            </div>
            <div class="modal-kpi">
                <div class="modal-kpi-val">{d['retreat_count']}</div>
                <div class="modal-kpi-lbl">Retreats</div>
            </div>
            <div class="modal-kpi">
                <div class="modal-kpi-val">{rain_status}</div>
                <div class="modal-kpi-lbl">Rain Safety</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    csv_bytes = pd.DataFrame(d["log_records"]).to_csv(index=False).encode("utf-8")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔁 Replay", key=f"dlg_replay_{in_dialog}", use_container_width=True):
            replay_order()
            st.rerun()
    with c2:
        if st.button("🔄 New Order", key=f"dlg_new_{in_dialog}", use_container_width=True):
            go_to_order()
            st.rerun()
    with c3:
        st.download_button(
            label="📥 Export CSV",
            data=csv_bytes,
            file_name=f"delivery_log_{selected_model.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"dlg_csv_{in_dialog}",
            use_container_width=True
        )


if hasattr(st, "dialog"):
    @st.dialog("Mission Result Summary")
    def show_mission_dialog(d, selected_model):
        render_mission_summary_content(d, selected_model, in_dialog=True)
else:
    show_mission_dialog = None


# ==========================================
# 8. SMART STICKY CENTERED HEADER (NO EMOJI)
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

    with col_form:
        with st.container(border=True):
            st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
            st.markdown("""
                <div class="section-header">
                    <div class="section-header-left">
                        <span class="dot"></span>
                        <span class="label">Order & Simulation Setup</span>
                    </div>
                    <span class="section-chip">All-in-One Control</span>
                </div>
            """, unsafe_allow_html=True)

            target_options = {env.node_names[n]: n for n in env.target_destinations}
            start_options = {env.node_names[n]: n for n in env.start_locations}

            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                selected_start_name = st.selectbox("🏪 1. Restaurant", list(start_options.keys()))
                chosen_start = start_options[selected_start_name]
            with r1_c2:
                selected_target_name = st.selectbox("🎯 2. Destination", list(target_options.keys()))
                chosen_target = target_options[selected_target_name]

            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                chosen_food = st.selectbox("🍱 3. Food Item", env.food_menus[chosen_start])
            with r2_c2:
                selected_model = st.selectbox(
                    "🤖 4. RL Navigation Model",
                    ["Q-Learning", "SARSA", "A2C", "REINFORCE", "DQN"]
                )

            r3_c1, r3_c2 = st.columns(2)
            with r3_c1:
                weather_mode = st.selectbox(
                    "🌦️ 5. Weather Condition",
                    ["🎲 Auto Random", "☀️ Force Sunny", "🌧️ Force Rainy"]
                )
            with r3_c2:
                time_mode = st.selectbox(
                    "🕒 6. Time of Day",
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
                <div class="snapshot-grid">
                    <div class="snapshot-item">
                        <div class="snap-label">Selected Route</div>
                        <div class="snap-val">{html.escape(selected_start_name)} ➜ {html.escape(selected_target_name)}</div>
                    </div>
                    <div class="snapshot-item">
                        <div class="snap-label">Order & Agent</div>
                        <div class="snap-val">{html.escape(chosen_food)} · {html.escape(selected_model)}</div>
                    </div>
                    <div class="snapshot-item">
                        <div class="snap-label">Min Graph Distance</div>
                        <div class="snap-val">{est_hops} hops {"(Sheltered)" if is_preview_rain else "(Direct)"}</div>
                    </div>
                    <div class="snapshot-item">
                        <div class="snap-label">Pre-Dispatch Estimate</div>
                        <div class="snap-val">{est_steps_str}</div>
                    </div>
                </div>
                <div class="advisor-strip">
                    <b>Dispatch Advisor:</b>
                    {"Rainy mode active — robot prioritises covered walkways (-40 open-air penalty)." if is_preview_rain else "Standard routing active — optimising shortest safe path across campus."}
                    <span class="risk-badge {risk_cls}">{risk_label}</span>
                </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Confirm Order & Launch Live Tracking", use_container_width=True):
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
                st.session_state.show_result_modal = False
                st.session_state.trigger_balloons = False
                st.session_state.page = "tracking"
                st.rerun()

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
                show_obstacles=False, figsize=(13, 8.0)
            )
            st.pyplot(preview_fig, use_container_width=True)
            plt.close(preview_fig)


# ==========================================
# VIEW 2: FULL-WIDTH LIVE MAP + COLLAPSIBLE RIGHT LOG DRAWER
# ==========================================

elif st.session_state.page == "tracking":

    if "order_config" not in st.session_state:
        st.session_state.page = "order"
        st.rerun()

    cfg = st.session_state.order_config
    selected_model = st.session_state.get("selected_model", "Q-Learning")
    q_table = load_q_table()

    # Right Slide-Out Log Drawer placeholder (MUST sit at root level for fixed viewport positioning)
    log_drawer_placeholder = st.empty()

    # Model Status Pill
    if selected_model == "Q-Learning" and q_table is not None:
        model_pill_html = '<span class="status-pill-ok">Model: Q-Learning (Trained 5D Table)</span>'
    elif selected_model == "Q-Learning":
        model_pill_html = '<span class="status-pill-warn">q_table.npy missing — Random Fallback</span>'
    else:
        model_pill_html = f'<span class="status-pill-warn">{html.escape(selected_model)} not linked — Random Fallback</span>'

    # ---- 1. COMPACT MISSION TOP BAR ----
    top_col1, top_col2, top_col3 = st.columns([3.9, 0.9, 1.0])
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
        st.button("🔁 Replay", on_click=replay_order, use_container_width=True)
    with top_col3:
        st.button("🔄 New Order", on_click=go_to_order, use_container_width=True)

    # ---- 2. FULL-WIDTH LIVE CAMPUS MAP BENTO CARD ----
    with st.container(border=True):
        st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div class="section-header" style="margin-bottom:6px;">
                <div class="section-header-left">
                    <span class="dot"></span>
                    <span class="label">Live 2D Campus Map</span>
                </div>
                <span class="section-chip">Click "◀ Live Log" on right edge to inspect steps</span>
            </div>
        """, unsafe_allow_html=True)
        map_placeholder = st.empty()
        intent_placeholder = st.empty()

    # ---- 3. LIVE SUMMARY METRICS STRIP (BELOW FULL-WIDTH MAP) ----
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

        start_location = env.current_location
        total_score = 0
        step_num = 0
        terminated = truncated = False
        previous_node = None
        visited_nodes = [env.current_location]
        log_entries = []
        log_records = []
        retreat_count = 0
        rain_violations = 0

        log_drawer_placeholder.markdown(render_right_log_drawer_html(log_entries), unsafe_allow_html=True)
        render_summary_cards(total_score, 0, step_num, env.current_weather, env.current_time_of_day,
                             len(env.pedestrian_positions) + len(env.car_positions))
        fig = render_campus_map(env, visited_nodes, env.current_location,
                                env.target_destination, total_score, step_num, "Order Dispatched")
        map_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        intent_placeholder.markdown(render_obstacle_pills_html(env), unsafe_allow_html=True)
        time.sleep(0.75)

        reward = 0
        status_msg = "Order Dispatched"

        while not (terminated or truncated):
            step_num += 1
            action = choose_action(env, state, selected_model, q_table, previous_node)
            previous_location = env.current_location
            prev_positions = {"ped": list(env.pedestrian_positions), "car": list(env.car_positions)}

            state, reward, terminated, truncated, info = env.step(action)
            total_score += reward

            outcome = info.get("outcome")
            if info.get("retreated"):
                retreat_count += 1
            if info.get("rain_violation"):
                rain_violations += 1

            moved = env.current_location != previous_location
            here = env.node_names[env.current_location]

            if outcome == "wait":
                status_msg, log_line = "Robot pulled over (WAIT)", "WAIT (Safe pull-over)"
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
                status_msg, log_line = f"Delivered at {here}!", f"MOVE ➜ {here} (Delivered!)"
            else:
                status_msg, log_line = f"Moved to {here}", f"MOVE ➜ {here}"

            log_entries.append({"step": step_num, "text": log_line, "reward": reward, "outcome": outcome})
            log_records.append({
                "Step": step_num,
                "From": env.node_names[previous_location],
                "To": here,
                "Outcome": outcome,
                "Reward": reward,
                "Cumulative_Score": total_score,
                "Weather": info["weather_str"],
                "Time_of_Day": info["time_of_day_str"],
                "Traffic": info["traffic_str"],
                "Rain_Violation": info["rain_violation"],
            })

            # Stream updated logs into the collapsible right drawer & refresh summary strip
            log_drawer_placeholder.markdown(render_right_log_drawer_html(log_entries), unsafe_allow_html=True)
            render_summary_cards(total_score, reward, step_num, env.current_weather, env.current_time_of_day,
                                 len(env.pedestrian_positions) + len(env.car_positions))

            # ---- SMOOTH GLIDING ANIMATION ----
            start_xy = env.node_coords[previous_location]
            end_xy = env.node_coords[env.current_location]
            blocked_idx = info.get("blocked_node")
            blocked_xy = env.node_coords[blocked_idx] if (outcome == "retreat" and blocked_idx is not None) else None

            for f in range(1, INTERP_FRAMES + 1):
                t = f / INTERP_FRAMES

                if moved:
                    curr_robot_xy = lerp_coord(start_xy, end_xy, t)
                    extend_trail = True
                elif blocked_xy is not None:
                    bounce_t = 0.42 * (1.0 - abs(2.0 * t - 1.0))
                    curr_robot_xy = lerp_coord(start_xy, blocked_xy, bounce_t)
                    extend_trail = False
                else:
                    curr_robot_xy = end_xy
                    extend_trail = False

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

                frame_status = (f"Moving to {here}..." if (moved and f < INTERP_FRAMES) else status_msg)
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

        # Save final snapshot, trigger centered popup modal on clean rerun
        st.session_state.sim_data = {
            "start_location": start_location,
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
        st.session_state.show_result_modal = True
        st.session_state.trigger_balloons = (info.get("outcome") == "goal")
        st.rerun()

    else:
        # ---- RESTORE FINISHED STATE ----
        d = st.session_state.sim_data
        env.current_location = d["current_location"]
        env.target_destination = d["target_destination"]
        env.current_weather = d["current_weather"]
        env.current_time_of_day = d["current_time_of_day"]
        env.selected_food = d["selected_food"]
        env.pedestrian_positions = list(d["pedestrian_positions"])
        env.car_positions = list(d["car_positions"])

        log_drawer_placeholder.markdown(render_right_log_drawer_html(d["log_entries"]), unsafe_allow_html=True)
        fig = render_campus_map(env, d["visited_nodes"], d["current_location"],
                                d["target_destination"], d["total_score"],
                                d["step_num"], d["status_msg"])
        map_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        intent_placeholder.markdown(d["obstacle_pills_html"], unsafe_allow_html=True)
        render_summary_cards(d["total_score"], d["last_reward"], d["step_num"],
                             d["current_weather"], d["current_time_of_day"],
                             len(d["pedestrian_positions"]) + len(d["car_positions"]))

        # Play celebratory balloons once upon successful arrival
        if st.session_state.pop("trigger_balloons", False):
            st.balloons()

        # Bottom action bar (reopen popup / export / go back anytime)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        csv_bytes = pd.DataFrame(d["log_records"]).to_csv(index=False).encode("utf-8")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.button("📊 Show Result Summary Popup", on_click=open_result_modal, use_container_width=True)
        with b2:
            st.button("🏠 Back to Order Selection", on_click=go_to_order, use_container_width=True)
        with b3:
            st.download_button(
                label="📥 Export Delivery Log (CSV)",
                data=csv_bytes,
                file_name=f"delivery_log_{selected_model.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Trigger centered popup automatically once delivery finishes
        if st.session_state.get("show_result_modal", False):
            st.session_state.show_result_modal = False
            if show_mission_dialog is not None:
                show_mission_dialog(d, selected_model)
            else:
                # Fallback for older Streamlit versions without st.dialog
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown('<div class="bento-marker"></div>', unsafe_allow_html=True)
                    render_mission_summary_content(d, selected_model, in_dialog=False)