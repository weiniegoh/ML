import random
import time
from html import escape
from pathlib import Path
from zipfile import BadZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
import streamlit as st

from campus_environment import make_env


MODEL_FILES = {
    "Q-Learning": "q_table_battery.npz",
    "SARSA": "sarsa_table.npy",
    "A2C": "a2c_table.npy",
    "REINFORCE": "reinforce_table.npy",
}
RANDOM_BASELINE = "Random baseline"
BASE_DIR = Path(__file__).resolve().parent
FRAMES_PER_STEP = 4
FRAME_DELAY = 0.14
STEP_PAUSE = 0.35

st.set_page_config(page_title="TAR UMT Campus Delivery", page_icon="🤖", layout="wide")

st.markdown("""
<style>
:root {
    color-scheme: light;
    --ink:#142c3c; --muted:#536b78; --teal:#08786d; --teal-dark:#075c54;
    --paper:#fff; --wash:#f3f7f7; --line:#dce8e8; --font:Inter,system-ui,sans-serif;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stMain"], [data-testid="stBottom"] {background:var(--wash);color:var(--ink);}
[data-testid="stAppViewContainer"] {font-family:var(--font);}
[data-testid="stMainBlockContainer"], .block-container {
    max-width:1260px;padding:4.15rem 2rem 3rem;
}
[data-testid="stDecoration"] {display:none;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stWidgetLabel"] p {color:var(--ink);letter-spacing:-.025em;}
[data-testid="stCaptionContainer"] p {color:var(--muted);}
[data-testid="stVerticalBlockBorderWrapper"],
.st-key-order_form, .st-key-route_card, .st-key-tracking_map, .st-key-tracking_status,
.st-key-summary_route, .st-key-summary_log {
    background:var(--paper);border-color:var(--line)!important;
    border-radius:18px!important;
}
.st-key-order_form,.st-key-route_card,.st-key-tracking_map,.st-key-tracking_status,
.st-key-summary_route,.st-key-summary_log {
    box-shadow:0 7px 25px #183d4210;
}
[data-baseweb="select"] > div, [data-baseweb="select"] input {
    background:#fff!important;color:var(--ink)!important;border-color:var(--line);
    border-radius:10px!important;
}
[data-baseweb="popover"], [role="listbox"], [role="option"] {
    background:#fff!important;color:var(--ink)!important;
}
[data-baseweb="popover"] * {color:var(--ink)!important;}
.stApp [data-testid="stBaseButton-primary"] {
    background:#08786d;border-color:#08786d;color:#fff;border-radius:10px;font-weight:700;
}
.stApp [data-testid="stBaseButton-primary"]:hover {
    background:#075c54;border-color:#075c54;color:#fff;
}
.stApp [data-testid="stBaseButton-primary"]:disabled {
    background:#dbe6e6;border-color:#dbe6e6;color:#536b78;opacity:1;
}
.stApp [data-testid="stBaseButton-secondary"],
.stApp [data-testid="stBaseButton-secondaryFormSubmit"] {
    background:#fff;color:var(--ink);border-color:var(--line);border-radius:10px;font-weight:650;
}
.stApp button:focus-visible {outline:3px solid #54b8ad;outline-offset:3px;}
.st-key-campus_nav_order button,.st-key-campus_nav_tracking button,
.st-key-campus_nav_summary button {
    min-height:2.65rem;font-weight:650;
}
.st-key-campus_nav_order [data-testid="stBaseButton-primary"],
.st-key-campus_nav_tracking [data-testid="stBaseButton-primary"],
.st-key-campus_nav_summary [data-testid="stBaseButton-primary"] {
    background:#ddf2ee;color:#075c54;border-color:#85c8bc;
}
.st-key-campus_nav_order button:disabled,
.st-key-campus_nav_tracking button:disabled,
.st-key-campus_nav_summary button:disabled {
    background:#f3f7f7;color:#687c86;border-color:#dce8e8;
}
.app-brand {display:flex;align-items:center;gap:.72rem;padding:.1rem 0 .2rem;}
.brand-mark {
    display:grid;place-content:center;width:37px;height:37px;flex:none;border-radius:11px;
    background:#0b7469;color:#fff;font-size:1.15rem;font-weight:800;
}
.brand-main {color:#162e3e;font-weight:770;line-height:1.15;font-size:1.04rem;}
.brand-campus {color:#536b78;font-size:.75rem;}
.hero {
    position:relative;overflow:hidden;isolation:isolate;margin:.5rem 0 .8rem;
    padding:clamp(1.25rem,2.7vw,2.05rem);
    border-radius:20px;background:linear-gradient(110deg,#112e3b 5%,#104d4c 100%);color:#fff;
}
.hero::before,.hero::after {
    content:"";position:absolute;z-index:-1;border:1px solid #9ce5d750;border-radius:50%;
    pointer-events:none;
}
.hero::before {width:23rem;height:23rem;right:-7rem;top:-13rem;}
.hero::after {width:30rem;height:30rem;right:-9rem;top:-15rem;}
.hero h1 {
    margin:.35rem 0 .35rem;padding:0;color:#fff!important;
    font-weight:760;font-size:clamp(1.8rem,3.8vw,2.8rem);line-height:1.1;letter-spacing:-.04em;
}
.hero p {margin:0;color:#d5ebe8;max-width:42ch;font-size:.96rem;line-height:1.5;}
.hero-tag {
    display:inline-block;padding:.23rem .62rem;border-radius:999px;border:1px solid #91d4c57a;
    color:#e6f7f3;font-size:.72rem;font-weight:650;
}
.section-heading {display:flex;align-items:baseline;justify-content:space-between;gap:.6rem;}
.section-heading h2 {
    margin:0;padding:0;color:var(--ink);font-size:1.14rem;letter-spacing:-.025em;font-weight:730;
}
.section-heading .aside {color:var(--muted);font-size:.78rem;}
.route-line {
    display:flex;align-items:center;gap:.85rem;flex-wrap:wrap;
    background:#f0f7f6;border-radius:12px;padding:.75rem .85rem;margin:.2rem 0 .48rem;
}
.route-label {display:block;color:var(--muted);font-size:.7rem;line-height:1.3;}
.route-value {display:block;color:var(--ink);font-size:.91rem;font-weight:700;line-height:1.38;}
.route-end {display:flex;align-items:center;gap:.52rem;}
.route-pin {width:9px;height:9px;border-radius:50%;background:#b46627;flex:none;}
.route-pin.drop {background:#08786d;}
.route-arrow {color:#65828b;font-size:1rem;}
.battery-card {
    background:#f0f7f6;border:1px solid #d8eae6;border-radius:13px;
    padding:.9rem 1rem;color:var(--ink);
}
.battery-heading {display:flex;align-items:center;justify-content:space-between;gap:.55rem;}
.battery-heading span {font-size:.86rem;font-weight:720;}
.battery-status {
    padding:.18rem .48rem;border-radius:999px;background:#dfefec;font-size:.72rem;font-weight:700;
}
.battery-status.low {background:#f8ebd7;color:#815411;}
.battery-status.critical {background:#fae2dd;color:#9a352a;}
.battery-reading {display:flex;align-items:baseline;justify-content:space-between;margin-top:.4rem;}
.battery-reading strong {font-size:1.56rem;letter-spacing:-.04em;line-height:1.2;}
.battery-reading small {color:var(--muted);font-size:.8rem;}
.battery-track {height:8px;background:#d9e8e5;border-radius:999px;overflow:hidden;margin-top:.6rem;}
.battery-fill {height:100%;border-radius:inherit;}
.delivery-state {padding:.65rem .05rem .2rem;}
.delivery-state .place {color:var(--ink);font-weight:740;line-height:1.3;font-size:1.15rem;}
.delivery-state .meta {color:var(--muted);font-size:.79rem;margin-top:.16rem;}
.delivery-state .event {color:var(--teal-dark);font-size:.86rem;font-weight:680;margin:.28rem 0 0;}
.activity-list {margin:0;padding:0;list-style:none;}
.activity-row {display:flex;gap:.62rem;align-items:flex-start;padding:.58rem .04rem;border-top:1px solid var(--line);}
.activity-dot {width:8px;height:8px;flex:none;border-radius:50%;background:#0a8878;margin-top:.36rem;}
.activity-dot.warn {background:#b47022;}
.activity-dot.stop {background:#a83d30;}
.activity-row .activity-desc {flex:1;min-width:0;}
.activity-row strong {display:block;color:var(--ink);font-size:.81rem;font-weight:690;}
.activity-row small {display:block;color:var(--muted);font-size:.73rem;}
.activity-energy {font-size:.72rem;color:var(--muted);white-space:nowrap;padding:.15rem .38rem;background:#f0f5f5;border-radius:5px;}
.result-panel {
    padding:1.05rem 1.2rem;border:1px solid #d0e8e1;border-left:4px solid #08786d;
    border-radius:13px;background:#eaf7f3;color:#174b3d;
}
.result-panel.warn {border-color:#f0dfbf;border-left-color:#a6382c;background:#fdf2e7;color:#733e22;}
.result-panel strong {display:block;font-size:1.2rem;line-height:1.3;letter-spacing:-.025em;}
.result-panel span {display:block;margin-top:.2rem;font-size:.86rem;line-height:1.4;}
[data-testid="stMetric"] {
    background:#fff;border:1px solid var(--line);border-radius:12px;padding:.72rem .84rem;
}
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] {color:var(--ink)!important;}
[data-testid="stMetricValue"] {font-weight:720;letter-spacing:-.03em;}
[data-testid="stImage"] img {border-radius:12px;}
@media (max-width:640px) {
    [data-testid="stMainBlockContainer"], .block-container {padding:4.1rem .9rem 2rem;}
    .hero {margin-top:.25rem;}
    .hero p {font-size:.85rem;}
    .route-line {gap:.55rem;}
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"] :is(.st-key-order_form, .st-key-tracking_map, .st-key-summary_route)
    ) {flex-direction:column;}
    [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"] :is(.st-key-order_form, .st-key-tracking_map, .st-key-summary_route)
    ) > [data-testid="stColumn"] {width:100%!important;flex:1 1 auto!important;}
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-campus_nav_order) {
        display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.4rem;
    }
    [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-campus_nav_order)
    > [data-testid="stColumn"] {width:100%!important;min-width:0!important;}
    .st-key-campus_nav_order button,.st-key-campus_nav_tracking button,
    .st-key-campus_nav_summary button {font-size:.75rem;padding:.25rem .34rem;}
}
</style>
""", unsafe_allow_html=True)


def hero(title, description, tag=None):
    badge = f'<span class="hero-tag">{escape(tag)}</span>' if tag else ""
    st.markdown(
        f'<div class="hero">{badge}<h1>{escape(title)}</h1><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def section_heading(container, title, aside=""):
    detail = f'<span class="aside">{escape(aside)}</span>' if aside else ""
    container.markdown(
        f'<div class="section-heading"><h2>{escape(title)}</h2>{detail}</div>',
        unsafe_allow_html=True,
    )


def route_strip(container, pickup, destination):
    container.markdown(
        f"""
        <div class="route-line">
            <div class="route-end"><span class="route-pin" aria-hidden="true"></span>
                <div><span class="route-label">Pickup</span>
                    <strong class="route-value">{escape(pickup)}</strong></div>
            </div>
            <div class="route-arrow" aria-hidden="true">→</div>
            <div class="route-end"><span class="route-pin drop" aria-hidden="true"></span>
                <div><span class="route-label">Destination</span>
                    <strong class="route-value">{escape(destination)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_battery(container, remaining, capacity, heading="Robot battery"):
    percent = 100 * remaining / capacity
    if not remaining:
        status, tone, color = "Depleted", "critical", "#b42318"
    elif percent <= 20:
        status, tone, color = "Critical", "critical", "#b42318"
    elif percent <= 40:
        status, tone, color = "Low charge", "low", "#ad6a20"
    else:
        status, tone, color = ("Fully charged" if remaining == capacity else "Available"), "", "#08786d"

    container.markdown(
        f"""
        <div class="battery-card">
            <div class="battery-heading"><span>{escape(heading)}</span>
                <span class="battery-status {tone}">{status}</span>
            </div>
            <div class="battery-reading">
                <strong>{remaining} / {capacity} units</strong><small>{percent:.0f}%</small>
            </div>
            <div class="battery-track" role="progressbar" aria-label="{escape(heading)}"
                 aria-valuemin="0" aria-valuemax="{capacity}" aria-valuenow="{remaining}">
                <div class="battery-fill" style="width:{percent:.2f}%;background:{color}"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_delivery_status(container, location, status, weather, time_of_day, step, score):
    container.markdown(
        f"""
        <div class="delivery-state">
            <div class="meta">Robot location</div>
            <div class="place">{escape(location)}</div>
            <div class="event">{escape(status)}</div>
            <div class="meta">Step {step} · {score:+} pts · {escape(weather)} · {escape(time_of_day)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_table(path, modified_ns):
    """Cache a model until its file is replaced."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            return loaded["table"]
    return loaded


def model_table(model, env):
    if model == RANDOM_BASELINE:
        return None

    path = BASE_DIR / MODEL_FILES[model]
    try:
        table = load_table(str(path), path.stat().st_mtime_ns)
    except FileNotFoundError:
        raise ValueError(f"Place {path.name} next to app.py, or select Random baseline.") from None
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
        raise ValueError(f"Could not load {path.name}: {exc}") from exc

    shape = tuple(int(space.n) for space in env.observation_space.spaces) + (
        int(env.action_space.n),
    )
    if table.shape != shape or table.dtype.kind not in "fiu":
        raise ValueError(
            f"{path.name} must contain a numeric table of shape {shape}. "
            f"Found {table.shape} ({table.dtype}). Use a model trained with this battery environment."
        )
    return table


def choose_action(env, state, table, rng):
    valid = env.valid_actions(state[0])
    if table is None:
        return rng.choice(valid)
    values = table[state][valid]
    best = np.flatnonzero(values == values.max())
    if best.size == 0:
        raise ValueError("The policy contains invalid scores for this state.")
    return valid[int(rng.choice(best.tolist()))]


def interpolate(a, b, progress):
    return tuple(start + (end - start) * progress for start, end in zip(a, b))


def round_box(x, y, width, height, radius, color, border, line_width=1.2):
    return FancyBboxPatch(
        (x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
        fc=color, ec=border, lw=line_width,
    )


def robot_icon():
    icon = DrawingArea(30, 38, 0, 0)
    for x in (9, 21):
        icon.add_artist(Circle((x, 4), 3.6, fc="#334155", ec="white", lw=0.8))
    icon.add_artist(round_box(6, 6, 18, 11, 2.5, "#0e7490", "#164e63"))
    icon.add_artist(Rectangle((8, 9), 14, 3, fc="#67e8f9", ec="none"))
    icon.add_artist(Rectangle((14, 29), 2, 5, fc="#164e63", ec="none"))
    icon.add_artist(Circle((15, 35), 2.4, fc="#f97316", ec="#9a3412", lw=0.8))
    icon.add_artist(round_box(4, 17, 22, 12, 3.5, "#06b6d4", "#164e63", 1.5))
    for x in (11, 19):
        icon.add_artist(Circle((x, 23), 3, fc="white", ec="#164e63", lw=0.8))
        icon.add_artist(Circle((x, 23), 1.3, fc="#0f172a", ec="none"))
    return icon


def pedestrian_icon():
    icon = DrawingArea(22, 33, 0, 0)
    for x in (6, 12):
        icon.add_artist(Rectangle((x, 1), 4, 10, fc="#a16207", ec="#713f12", lw=0.8))
    icon.add_artist(round_box(3.5, 9, 15, 13, 5, "#facc15", "#854d0e", 1.3))
    icon.add_artist(Circle((11, 27.5), 5, fc="#fde68a", ec="#854d0e", lw=1.3))
    return icon


def car_icon():
    icon = DrawingArea(38, 23, 0, 0)
    icon.add_artist(Polygon([(10, 13), (13, 21), (27, 21), (30, 13)],
                            fc="#ef4444", ec="#7f1d1d", lw=1.3))
    icon.add_artist(Polygon([(13, 14), (15, 19.5), (26, 19.5), (28, 14)],
                            fc="#bae6fd", ec="none"))
    icon.add_artist(Rectangle((20, 14), 1, 5.5, fc="#b91c1c", ec="none"))
    icon.add_artist(round_box(2, 6, 34, 8, 3, "#ef4444", "#7f1d1d", 1.3))
    icon.add_artist(Circle((33.5, 10), 1.5, fc="#fef08a", ec="none"))
    for x in (11, 27):
        icon.add_artist(Circle((x, 6), 4.2, fc="#1f2937", ec="white", lw=0.8))
        icon.add_artist(Circle((x, 6), 1.6, fc="#9ca3af", ec="none"))
    return icon


def waypoint_icon(name, pickup):
    name = name.lower()
    if name.startswith("roof top"):
        return None

    icon = DrawingArea(30, 26, 0, 0)
    if pickup:
        icon.add_artist(Rectangle((5, 5), 20, 10, fc="#d97706", ec="#92400e", lw=0.8))
        icon.add_artist(Polygon([(3, 15), (27, 15), (24, 23), (6, 23)],
                                fc="#fb923c", ec="#9a3412", lw=0.8))
        for x in (8, 14, 20):
            icon.add_artist(Rectangle((x, 15), 3, 7, fc="#fff7ed", ec="none"))
    elif "swimming pool" in name:
        icon.add_artist(round_box(3, 5, 24, 17, 3, "#38bdf8", "#0369a1"))
        for y in (10, 15):
            icon.add_artist(Line2D([7, 12, 17, 22], [y, y + 2, y, y + 2], color="white", lw=1))
    elif "car park" in name or "carpark" in name:
        icon.add_artist(round_box(5, 4, 20, 21, 2, "#2563eb", "#1e40af"))
        icon.add_artist(Text(15, 14, "P", ha="center", va="center",
                             fontsize=15, fontweight="bold", color="white"))
    elif "bus stop" in name:
        icon.add_artist(Rectangle((14, 3), 2, 15, fc="#475569", ec="none"))
        icon.add_artist(round_box(4, 16, 22, 9, 1, "#0f766e", "#134e4a"))
        icon.add_artist(Text(15, 20, "BUS", ha="center", va="center",
                             fontsize=6, fontweight="bold", color="white"))
    elif "basketball" in name:
        icon.add_artist(Rectangle((3, 5), 24, 17, fc="#e9a15b", ec="#92400e", lw=1))
        icon.add_artist(Line2D([15, 15], [5, 22], color="white", lw=0.8))
        icon.add_artist(Circle((15, 13.5), 3.5, fc="none", ec="white", lw=0.8))
    elif "library" in name:
        for x, color in ((5, "#2563eb"), (13, "#16a34a"), (21, "#f59e0b")):
            icon.add_artist(Rectangle((x, 5), 5, 18, fc=color, ec="#334155", lw=0.8))
    else:
        icon.add_artist(Rectangle((5, 5), 20, 15, fc="#c4b49a", ec="#78664d", lw=0.9))
        icon.add_artist(Polygon([(3, 20), (15, 25), (27, 20)], fc="#8b7355", ec="#78664d"))
        for x in (9, 18):
            icon.add_artist(Rectangle((x, 9), 5, 7, fc="#e0f2fe", ec="#64748b", lw=0.6))
    return icon


def place_icon(ax, icon, xy, offset, order):
    marker = AnnotationBbox(
        icon, xy, xybox=offset, xycoords="data", boxcoords="offset points",
        frameon=False, pad=0, zorder=order,
    )
    ax.add_artist(marker)


def draw_map(env, start, target, path, robot_xy, pedestrians=(), cars=(), rainy=False):
    fig, ax = plt.subplots(figsize=(12.4, 7.6), facecolor="#f0f7f6")
    ground = "#dcebf0" if rainy else "#dcead1"
    ax.set_facecolor(ground)
    coords = env.node_coords

    for x, y, width, height in (
        (5, 88, 13, 7), (39, 82, 10, 5), (47, 49, 8, 6),
        (70, 12, 12, 6), (4, 5, 9, 5), (88, 72, 9, 5),
    ):
        ax.add_patch(round_box(x, y, width, height, 1.5, "#c6dcb5", "none"))
    for x, y in (
        (4, 83), (20, 65), (25, 57), (47, 75), (58, 83), (88, 88), (96, 78),
        (5, 40), (16, 15), (35, 10), (68, 10), (76, 18), (94, 17), (96, 56),
    ):
        ax.add_patch(Circle((x, y), 1.15, fc="#5d9a54", ec="#3b7134", lw=0.6, zorder=0.9))

    for node, neighbors in env.graph.items():
        for neighbor in neighbors:
            if node >= neighbor:
                continue
            uncovered = frozenset((node, neighbor)) in env.uncovered_paths
            ax.plot(
                *zip(coords[node], coords[neighbor]),
                color="#ad7411" if uncovered else "#64748b",
                linestyle="--" if uncovered else "-", linewidth=1.9, zorder=1,
            )

    trail = [coords[node] for node in path]
    if robot_xy != trail[-1]:
        trail.append(robot_xy)
    if len(trail) > 1:
        ax.plot(*zip(*trail), color="#0891b2", linewidth=4.5, zorder=2)

    ax.scatter(*zip(*coords.values()), s=9, c="#50666e", zorder=3)
    for node, color in ((start, "#c35b17"), (target, "#087e73")):
        ax.scatter(*coords[node], c=color, s=110, edgecolors="white", linewidths=1.2, zorder=4)

    label_outline = [patheffects.withStroke(linewidth=2.6, foreground=ground)]
    offsets = {0: (-10, -18), 1: (0, 14), 2: (-5, 15), 3: (0, -17), 4: (3, 16),
               5: (6, 15), 17: (10, -14), 18: (-2, -15), 19: (9, -12), 26: (-3, 15),
               31: (9, 14), 33: (-12, 17)}
    for node, xy in coords.items():
        name = env.node_names[node]
        icon = waypoint_icon(name, node in env.start_locations)
        if icon is None:
            continue
        dx, dy = offsets.get(node, (4, 15))
        place_icon(ax, icon, xy, (dx, dy), 5)
        if "bus stop" in name.lower():
            continue
        align = "right" if xy[0] > 76 else "left"
        horizontal = dx - 15 if align == "right" else dx + 15
        label = ax.annotate(
            name, xy, xytext=(horizontal, dy), textcoords="offset points",
            ha=align, va="center", fontsize=7,
            fontweight="bold" if node in (start, target) else "normal",
            color="#143c42", zorder=6,
        )
        label.set_path_effects(label_outline)

    for positions, icon_factory, tag, offset in (
        (pedestrians, pedestrian_icon, "P", -17), (cars, car_icon, "C", 16),
    ):
        for index, xy in enumerate(positions, start=1):
            dx = offset + 9 * (index - 1)
            place_icon(ax, icon_factory(), xy, (dx, -8), 7)
            ax.annotate(
                f"{tag}{index}", xy, xytext=(dx, -26), textcoords="offset points",
                ha="center", va="center", color="#334155", fontsize=7,
                fontweight="bold", zorder=8, bbox={"fc": "white", "ec": "none", "alpha": .85},
            )

    place_icon(ax, robot_icon(), robot_xy, (0, 9), 9)
    legend = [
        Line2D([], [], color="#0891b2", lw=3, label="Travelled route"),
        Line2D([], [], color="#64748b", label="Covered path"),
        Line2D([], [], color="#ad7411", ls="--", label="Uncovered path"),
        Line2D([], [], marker="o", color="#c35b17", ls="", label="Pickup"),
        Line2D([], [], marker="o", color="#087e73", ls="", label="Destination"),
    ]
    if pedestrians or cars:
        legend.extend([
            Line2D([], [], marker="$P$", color="#9a5300", ls="", label="Pedestrian"),
            Line2D([], [], marker="$C$", color="#b42318", ls="", label="Car"),
        ])
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.02),
              ncol=4, frameon=False, fontsize=8.5)
    ax.set(xlim=(-7, 104), ylim=(0, 106), xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.8)
    return fig


def show_map(slot, *args, **kwargs):
    figure = draw_map(*args, **kwargs)
    try:
        slot.pyplot(figure, use_container_width=True)
    finally:
        plt.close(figure)


OUTCOME_LABELS = {
    "goal": "Delivery completed",
    "ped_collision": "Delivery failed · pedestrian collision",
    "car_collision": "Delivery failed · vehicle collision",
    "battery_depleted": "Delivery failed · battery depleted",
    "timeout": "Delivery timed out",
    "wait": "Robot waited",
    "retreat": "Robot retreated",
    "invalid": "Invalid action",
    "sidestep": "Robot sidestepped",
    "sidestep_failed": "Sidestep failed",
    "move": "Robot moved",
}


def obstacle_coordinates(env):
    return tuple(env.node_coords[n] for n in env.pedestrian_positions), tuple(
        env.node_coords[n] for n in env.car_positions
    )


def obstacle_status(env):
    intents = env.obstacle_intents()
    labels = []
    for kind, prefix in (("ped", "P"), ("car", "C")):
        for index, item in enumerate(intents[kind], start=1):
            here = env.node_names[item["node"]]
            goal = item["goal"]
            if goal is not None:
                detail = f" → {env.node_names[goal]}"
            else:
                detail = " · waiting" if item["waiting"] else ""
            labels.append(f"{prefix}{index} at {here}{detail}")
    return "  •  ".join(labels)


def step_label(row):
    outcome = row["Outcome"]
    if outcome == "retreat":
        return "Retreated after moving" if row["Energy Used"] else "Next junction blocked"
    return OUTCOME_LABELS.get(outcome, outcome)


def show_log(slot, rows, compact=False):
    if not rows:
        slot.caption("Waiting for the first movement decision.")
        return
    if not compact:
        slot.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        return

    events = []
    for row in reversed(rows[-4:]):
        outcome = row["Outcome"]
        if outcome in ("timeout", "battery_depleted", "ped_collision", "car_collision"):
            tone = "stop"
        elif outcome in ("wait", "retreat", "invalid", "sidestep_failed"):
            tone = "warn"
        else:
            tone = ""
        events.append(
            f'<li class="activity-row"><span class="activity-dot {tone}" aria-hidden="true"></span>'
            f'<div class="activity-desc"><strong>{escape(step_label(row))}</strong>'
            f'<small>{escape(row["To"])} · Step {row["Step"]}</small></div>'
            f'<span class="activity-energy">{row["Energy Used"]} units</span></li>'
        )
    slot.markdown(f'<ul class="activity-list">{"".join(events)}</ul>', unsafe_allow_html=True)


def simulate(env, mission, table, map_slot, battery_slot, log_slot, detail_slot, obstacle_slot):
    order = mission["order"]
    start, target = order["start"], order["target"]
    state, info = env.reset(seed=mission["seed"], options=order)
    rng = random.Random(mission["seed"])
    initial_battery = info["battery_remaining"]
    capacity = info["battery_capacity"]
    battery = initial_battery
    path = [start]
    records = []
    score = 0
    rain_violations = 0
    retreats = 0
    terminated = truncated = False
    rainy = env.current_weather == 1
    pedestrians, cars = obstacle_coordinates(env)
    show_map(map_slot, env, start, target, path, env.node_coords[start], pedestrians, cars, rainy)
    render_battery(battery_slot, battery, capacity)
    show_log(log_slot, records)
    obstacle_slot.caption(obstacle_status(env))
    render_delivery_status(
        detail_slot, env.node_names[start], "Dispatched", info["weather_str"],
        info["time_of_day_str"], 0, score,
    )

    while not (terminated or truncated):
        previous_node, previous_battery = state[0], battery
        old_pedestrians, old_cars = pedestrians, cars
        action = choose_action(env, state, table, rng)
        state, reward, terminated, truncated, info = env.step(action)
        current_node = state[0]
        battery = info["battery_remaining"]
        score += reward
        rain_violations += int(info["rain_violation"])
        retreats += int(info["retreated"])
        outcome = "timeout" if truncated else info["outcome"]
        records.append({
            "Step": env.steps,
            "From": env.node_names[previous_node],
            "To": env.node_names[current_node],
            "Outcome": outcome,
            "Reward": reward,
            "Cumulative Score": score,
            "Battery Remaining": battery,
            "Battery Capacity": capacity,
            "Energy Used": previous_battery - battery,
            "Weather": info["weather_str"],
            "Time of Day": info["time_of_day_str"],
            "Traffic": info["traffic_str"],
            "Rain Violation": info["rain_violation"],
        })
        pedestrians, cars = obstacle_coordinates(env)
        frames = FRAMES_PER_STEP if mission["animate"] else 1
        for frame in range(1, frames + 1):
            progress = frame / frames
            robot_xy = interpolate(env.node_coords[previous_node], env.node_coords[current_node], progress)
            ped_xy = [interpolate(a, b, progress) for a, b in zip(old_pedestrians, pedestrians)]
            car_xy = [interpolate(a, b, progress) for a, b in zip(old_cars, cars)]
            show_map(map_slot, env, start, target, path, robot_xy, ped_xy, car_xy, rainy)
            if frame < frames:
                time.sleep(FRAME_DELAY)

        if current_node != previous_node:
            path.append(current_node)

        render_delivery_status(
            detail_slot, env.node_names[current_node], step_label(records[-1]),
            info["weather_str"], info["time_of_day_str"], env.steps, score,
        )
        render_battery(battery_slot, battery, capacity)
        show_log(log_slot, records, compact=True)
        obstacle_slot.caption(obstacle_status(env))
        if mission["animate"] and not (terminated or truncated):
            time.sleep(STEP_PAUSE)

    return {
        "outcome": outcome, "score": score, "records": records, "path": path,
        "battery": battery, "capacity": capacity, "initial_battery": initial_battery,
        "pedestrians": pedestrians, "cars": cars, "rainy": rainy,
        "weather": info["weather_str"], "time_of_day": info["time_of_day_str"],
        "rain_violations": rain_violations, "retreats": retreats,
        "obstacle_status": obstacle_status(env),
    }


def navigate(page):
    st.session_state.campus_page = page


def dispatch(start, target, food, weather, time_of_day, model, animate):
    order = {"start": start, "target": target, "food": food}
    if weather != "Random":
        order["weather"] = ("Sunny", "Rainy").index(weather)
    if time_of_day != "Random":
        order["time_of_day"] = ("Day", "Evening", "Night").index(time_of_day)
    st.session_state.campus_mission = {
        "order": order, "model": model, "animate": animate,
        "seed": random.randrange(2**32),
    }
    st.session_state.campus_result = None
    st.session_state.campus_success_popup = False
    navigate("tracking")


def replay():
    st.session_state.campus_result = None
    st.session_state.campus_success_popup = False
    navigate("tracking")


def result_callout(result):
    outcome = result["outcome"]
    title = OUTCOME_LABELS.get(outcome, outcome)
    if outcome == "goal":
        description = "The robot reached the selected destination."
        tone = ""
    else:
        description = f'The robot ended this delivery at {result["records"][-1]["To"]}.'
        tone = "warn"
    st.markdown(
        f'<div class="result-panel {tone}" role="status"><strong>{escape(title)}</strong>'
        f'<span>{escape(description)}</span></div>',
        unsafe_allow_html=True,
    )


def export_log(result, model, key):
    st.download_button(
        "Download delivery log",
        data=pd.DataFrame(result["records"]).to_csv(index=False).encode("utf-8"),
        file_name=f"delivery_log_{model.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


st.session_state.setdefault("campus_page", "order")
st.session_state.setdefault("campus_mission", None)
st.session_state.setdefault("campus_result", None)
page = st.session_state.campus_page
mission = st.session_state.campus_mission
result = st.session_state.campus_result
env = make_env()

st.markdown("""
<div class="app-brand">
    <div class="brand-mark" aria-hidden="true">↗</div>
    <div><div class="brand-main">Campus Delivery</div>
        <div class="brand-campus">TAR UMT</div></div>
</div>
""", unsafe_allow_html=True)

for column, (label, name, disabled) in zip(st.columns(3), (
    ("Order", "order", False),
    ("Live Tracking", "tracking", mission is None),
    ("Delivery Summary", "summary", result is None),
)):
    column.button(
        label, type="primary" if page == name else "secondary",
        on_click=navigate, args=(name,), disabled=disabled,
        key=f"campus_nav_{name}", use_container_width=True,
    )

if page == "order":
    hero("Order campus delivery", "Choose your food and destination, then follow the robot across campus.")
    left, right = st.columns([.85, 1.38], gap="large")
    with left:
        with st.container(border=True, key="order_form"):
            section_heading(st, "New order")
            a, b = st.columns(2)
            start = a.selectbox("Pickup restaurant", env.start_locations,
                                format_func=env.node_names.__getitem__)
            target = b.selectbox("Delivery destination", env.target_destinations,
                                 format_func=env.node_names.__getitem__)
            food = st.selectbox("Food item", env.food_menus[start])

            with st.expander("Simulation settings", expanded=True):
                model = st.selectbox("Navigation model", [*MODEL_FILES, RANDOM_BASELINE])
                a, b = st.columns(2)
                weather = a.selectbox("Weather", ["Random", "Sunny", "Rainy"])
                time_of_day = b.selectbox("Time of day", ["Random", "Day", "Evening", "Night"])
                animate = st.toggle("Animate movement", value=True)

            missing_file = model != RANDOM_BASELINE and not (BASE_DIR / MODEL_FILES[model]).is_file()
            if missing_file:
                st.caption(f"{model} needs `{MODEL_FILES[model]}`. Choose Random baseline to try tracking.")
            st.button(
                "Dispatch order", type="primary", on_click=dispatch,
                args=(start, target, food, weather, time_of_day, model, animate),
                disabled=missing_file, use_container_width=True,
            )

    with right:
        with st.container(border=True, key="route_card"):
            section_heading(st, "Campus map", "Order preview")
            route_strip(st, env.node_names[start], env.node_names[target])
            show_map(st, env, start, target, [start], env.node_coords[start], rainy=weather == "Rainy")
            render_battery(st, env.battery_capacity, env.battery_capacity, "Dispatch battery")

elif page == "tracking":
    order = mission["order"]
    model = mission["model"]
    start, target = order["start"], order["target"]
    order_id = f"Order CD-{mission['seed']:08X}"

    hero("Track your delivery", f'{order["food"]} · {model}', order_id)
    route_strip(st, env.node_names[start], env.node_names[target])
    map_col, status_col = st.columns([1.55, 1], gap="large")
    with map_col:
        with st.container(border=True, key="tracking_map"):
            section_heading(st, "Campus map")
            map_slot = st.empty()
            with st.expander("Campus traffic"):
                obstacle_slot = st.empty()
    with status_col:
        with st.container(border=True, key="tracking_status"):
            section_heading(st, "Delivery status")
            detail_slot = st.empty()
            battery_slot = st.empty()
            section_heading(st, "Recent activity")
            log_slot = st.empty()

    if result is None:
        try:
            table = model_table(model, env)
            result = simulate(
                env, mission, table, map_slot, battery_slot, log_slot, detail_slot, obstacle_slot
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.session_state.campus_result = result
        st.session_state.campus_success_popup = result["outcome"] == "goal"
        st.rerun()

    last = result["records"][-1]
    show_map(
        map_slot, env, start, target, result["path"], env.node_coords[result["path"][-1]],
        result["pedestrians"], result["cars"], result["rainy"],
    )
    render_battery(battery_slot, result["battery"], result["capacity"], "Battery at finish")
    render_delivery_status(
        detail_slot, last["To"], OUTCOME_LABELS[result["outcome"]],
        result["weather"], result["time_of_day"], len(result["records"]), result["score"],
    )
    obstacle_slot.caption(result.get("obstacle_status", ""))
    show_log(log_slot, result["records"], compact=True)
    result_callout(result)

    first, second, third = st.columns(3)
    first.button("View delivery summary", type="primary", on_click=navigate,
                 args=("summary",), use_container_width=True)
    second.button("Replay order", on_click=replay, key="tracking_replay", use_container_width=True)
    with third:
        export_log(result, model, "tracking_csv")

    if st.session_state.pop("campus_success_popup", False) and hasattr(st, "dialog"):
        @st.dialog("Delivery complete")
        def show_delivery_success():
            st.write(f'{order["food"]} arrived at {env.node_names[target]}.')
            render_battery(st, result["battery"], result["capacity"], "Battery at finish")
            if st.button(
                "Open delivery summary", key="success_summary", on_click=navigate,
                args=("summary",), use_container_width=True,
            ):
                st.rerun()

        show_delivery_success()

elif page == "summary":
    order = mission["order"]
    model = mission["model"]
    order_id = f"Order CD-{mission['seed']:08X}"

    hero("Delivery summary", f'{order["food"]} · {model}', order_id)
    result_callout(result)

    a, b, c, d = st.columns(4)
    a.metric("Final score", f'{result["score"]:+} pts')
    b.metric("Delivery steps", len(result["records"]))
    c.metric("Energy used", f'{result["initial_battery"] - result["battery"]} units')
    d.metric("Retreats", result["retreats"])

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True, key="summary_route"):
            section_heading(st, "Route recap")
            route_strip(st, env.node_names[order["start"]], env.node_names[order["target"]])
            st.write(" → ".join(env.node_names[node] for node in result["path"]))
    with right:
        with st.container(border=True):
            section_heading(st, "Battery and conditions")
            render_battery(st, result["battery"], result["capacity"], "Battery at finish")
            conditions = f'{result["weather"]} · {result["time_of_day"]}'
            if result["rainy"]:
                conditions += f' · {result["rain_violations"]} uncovered moves'
            st.caption(conditions)

    with st.container(border=True, key="summary_log"):
        section_heading(st, "Delivery log", "Energy and decisions by step")
        show_log(st, result["records"])

    a, b, c = st.columns(3)
    a.button("Back to live map", on_click=navigate, args=("tracking",), use_container_width=True)
    b.button("Replay order", on_click=replay, key="summary_replay", use_container_width=True)
    with c:
        export_log(result, model, "summary_csv")
