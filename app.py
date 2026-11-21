import html
import os
import random
import time

import numpy as np
import streamlit as st
import matplotlib

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
# 0. SETTINGS
# ==========================================

STEP_DELAY = 0.6                 # seconds between animation frames
USE_ANTI_BACKTRACKING = True     # must match the setting used in training
OBSTACLE_MODE = "routes"         # must match OBSTACLE_MODE in q_learning_delivery.py
Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.npy")


# ==========================================
# 1. PAGE CONFIGURATION & LIGHT-MODE STYLING
# (the theme itself is forced to light by .streamlit/config.toml)
# ==========================================

st.set_page_config(
    page_title="TAR UMT AI Food Delivery",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    /* ---- Force a complete light look, independent of the user's/browser's
       Streamlit theme (so no white-on-white text that only shows on hover) ---- */
    :root { color-scheme: light; }

    html, body, .stApp,
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stBottom"], .main {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #f1f5f9 !important;
    }

    /* ---- Normal text ---- */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stHeader"] *,
    .stApp [data-testid="stToolbar"] * {
        color: #1e293b !important;
    }

    /* ---- Dropdowns (selectbox) ---- */
    .stApp [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stApp [data-baseweb="select"] *,
    .stApp [data-baseweb="select"] input {
        color: #1e293b !important;
        -webkit-text-fill-color: #1e293b !important;
    }
    .stApp [data-baseweb="select"] svg { fill: #475569 !important; }

    /* the opened list lives in a popover outside .stApp */
    [data-baseweb="popover"], [data-baseweb="popover"] > div,
    [data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li, [data-baseweb="menu"] li,
    ul[role="listbox"] li {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li *, [data-baseweb="menu"] li *,
    ul[role="listbox"] li * {
        color: #1e293b !important;
    }
    [data-baseweb="popover"] li:hover, ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {
        background-color: #e0f2fe !important;
    }

    /* ---- Radio buttons ---- */
    .stApp [data-testid="stRadio"] label,
    .stApp [data-testid="stRadio"] label *,
    .stApp [data-testid="stRadio"] p {
        color: #1e293b !important;
    }

    /* ---- Expander ---- */
    .stApp [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px;
    }
    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary *,
    .stApp [data-testid="stExpander"] details,
    .stApp [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #1e293b !important;
        background-color: transparent !important;
    }
    .stApp [data-testid="stExpander"] summary:hover {
        background-color: #f1f5f9 !important;
    }

    /* ---- Info / warning / success / error boxes ----
       The inner containers carry their own (dark) background, so every
       layer is made white/transparent, otherwise dark text sits on dark. */
    .stApp [data-testid="stAlert"],
    .stApp [data-testid="stAlertContainer"],
    .stApp [data-testid="stNotification"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-left: 5px solid #06b6d4 !important;
        border-radius: 10px;
    }
    .stApp [data-testid="stAlert"] *,
    .stApp [data-testid="stAlertContainer"] *,
    .stApp [data-testid="stNotification"] * {
        background: transparent !important;
        background-color: transparent !important;
        color: #1e293b !important;
        -webkit-text-fill-color: #1e293b !important;
    }
    /* avoid a double border when the outer and inner box both match */
    .stApp [data-testid="stAlert"] [data-testid="stAlertContainer"] {
        border: none !important;
    }

    /* ---- Metric cards ---- */
    .stApp [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #06b6d4;
        border-radius: 10px;
        padding: 10px 14px;
    }
    
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] *,
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] * {
        color: #1e293b !important;
    }
    
    /* ---- Metric delta text (+/- points) ---- */
    .stApp [data-testid="stMetricDelta"],
    .stApp [data-testid="stMetricDelta"] * {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
    }

    /* ---- Buttons: white text on the teal gradient (always visible) ---- */
    .stApp div.stButton > button,
    .stApp [data-testid="stBaseButton-secondary"],
    .stApp [data-testid="stBaseButton-primary"] {
        width: 100%;
        background: linear-gradient(90deg, #0d9488 0%, #06b6d4 100%) !important;
        border: none !important;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 700;
    }
    .stApp div.stButton > button,
    .stApp div.stButton > button *,
    .stApp div.stButton button [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stBaseButton-secondary"] *,
    .stApp [data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
    }
    .stApp div.stButton > button:hover,
    .stApp [data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(90deg, #0f766e 0%, #0891b2 100%) !important;
    }
    
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. ENVIRONMENT & Q-TABLE
# ==========================================

env = CampusDeliveryEnv(obstacle_mode=OBSTACLE_MODE)


NOTE_STYLES = {   # background, border, text
    "info":    ("#e0f2fe", "#0ea5e9", "#0c4a6e"),
    "success": ("#dcfce7", "#16a34a", "#14532d"),
    "warning": ("#fef3c7", "#f59e0b", "#78350f"),
    "error":   ("#fee2e2", "#dc2626", "#7f1d1d"),
}


def note(kind, text):
    """Coloured message box (plain HTML, so it never depends on Streamlit internals)."""
    bg, border, fg = NOTE_STYLES[kind]
    st.markdown(
        f'<div style="background:{bg};border:1px solid {border};'
        f'border-left:6px solid {border};border-radius:10px;padding:14px 18px;'
        f'margin:8px 0;color:{fg};font-weight:600;line-height:1.5;">'
        f'{html.escape(text)}</div>',
        unsafe_allow_html=True)


def describe_obstacles(env):
    """One line per obstacle: where it is and what it is doing (routes mode)."""
    if env.obstacle_mode != "routes":
        return ""
    intents = env.obstacle_intents()
    lines = []
    for kind, label, icon in (("ped", "P", "🚶"), ("car", "C", "🚗")):
        for i, it in enumerate(intents[kind]):
            here = env.node_names[it["node"]]
            if it["goal"] is not None:
                what = f"heading to {env.node_names[it['goal']]}"
            elif it["waiting"]:
                what = "waiting"
            else:
                what = "choosing next stop"
            lines.append(f"{icon} **{label}{i + 1}** at {here}: {what}")
    return "  \n".join(lines)


@st.cache_resource
def load_q_table(path=Q_TABLE_PATH):
    """Load the trained Q-table if it exists and has the right shape."""
    if not os.path.exists(path):
        return None
    table = np.load(path)
    if table.shape != (36, 36, 2, 3, 37):
        return None
    return table


# ==========================================
# 3. 2D CAMPUS MAP RENDERER
# ==========================================

# ---- Vector icons (drawn in points, so they never get stretched by the map) ----

def _round_box(x, y, w, h, r, fc, ec, lw=1.2):
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                          fc=fc, ec=ec, lw=lw)


def icon_robot(scale=1.0):
    k = scale
    da = DrawingArea(30 * k, 38 * k, 0, 0)
    for cx in (9, 21):                                          # wheels
        da.add_artist(Circle((cx * k, 4 * k), 3.6 * k, fc="#334155", ec="white", lw=0.8))
    da.add_artist(_round_box(6 * k, 6 * k, 18 * k, 11 * k, 2.5 * k, "#0e7490", "#164e63"))   # body
    da.add_artist(Rectangle((8 * k, 9 * k), 14 * k, 3 * k, fc="#67e8f9", ec="none"))          # chest light
    da.add_artist(Rectangle((14 * k, 29 * k), 2 * k, 5 * k, fc="#164e63", ec="none"))         # antenna
    da.add_artist(Circle((15 * k, 35 * k), 2.4 * k, fc="#f97316", ec="#9a3412", lw=0.8))
    da.add_artist(_round_box(4 * k, 17 * k, 22 * k, 12 * k, 3.5 * k, "#06b6d4", "#164e63", 1.5))  # head
    for cx in (11, 19):                                         # eyes
        da.add_artist(Circle((cx * k, 23 * k), 3 * k, fc="white", ec="#164e63", lw=0.8))
        da.add_artist(Circle((cx * k, 23 * k), 1.3 * k, fc="#0f172a", ec="none"))
    return da


def icon_person(scale=1.0):
    k = scale
    da = DrawingArea(22 * k, 33 * k, 0, 0)
    for lx in (6, 12):                                          # legs
        da.add_artist(Rectangle((lx * k, 1 * k), 4 * k, 10 * k, fc="#a16207", ec="#713f12", lw=0.8))
    da.add_artist(_round_box(3.5 * k, 9 * k, 15 * k, 13 * k, 5 * k, "#facc15", "#854d0e", 1.3))  # torso
    da.add_artist(Circle((11 * k, 27.5 * k), 5 * k, fc="#fde68a", ec="#854d0e", lw=1.3))         # head
    return da


def icon_car(scale=1.0):
    k = scale
    da = DrawingArea(38 * k, 23 * k, 0, 0)
    da.add_artist(Polygon([(10 * k, 13 * k), (13 * k, 21 * k), (27 * k, 21 * k), (30 * k, 13 * k)],
                          closed=True, fc="#ef4444", ec="#7f1d1d", lw=1.3))                       # cabin
    da.add_artist(Polygon([(13 * k, 14 * k), (15 * k, 19.5 * k), (19.5 * k, 19.5 * k), (19.5 * k, 14 * k)],
                          closed=True, fc="#bae6fd", ec="none"))                                   # windows
    da.add_artist(Polygon([(20.5 * k, 14 * k), (20.5 * k, 19.5 * k), (26 * k, 19.5 * k), (28 * k, 14 * k)],
                          closed=True, fc="#bae6fd", ec="none"))
    da.add_artist(_round_box(2 * k, 6 * k, 34 * k, 8 * k, 3 * k, "#ef4444", "#7f1d1d", 1.3))      # body
    da.add_artist(Circle((33.5 * k, 10 * k), 1.5 * k, fc="#fef08a", ec="none"))                    # headlight
    for cx in (11, 27):                                         # wheels
        da.add_artist(Circle((cx * k, 6 * k), 4.2 * k, fc="#1f2937", ec="white", lw=0.8))
        da.add_artist(Circle((cx * k, 6 * k), 1.6 * k, fc="#9ca3af", ec="none"))
    return da


def place_icon(ax, da, x, y, dx=0, dy=0, zorder=10):
    """Put an icon at data point (x, y), shifted by (dx, dy) points."""
    ab = AnnotationBbox(da, (x, y), xybox=(dx, dy), xycoords="data",
                        boxcoords="offset points", frameon=False,
                        box_alignment=(0.5, 0.5), pad=0)
    ab.set_zorder(zorder)
    ax.add_artist(ab)


def render_campus_map(env, visited_nodes, current_node, target_node,
                      score, step_num, status_text, show_obstacles=True,
                      prev_positions=None):

    is_rainy = env.current_weather == 1
    bg_color = "#e0f2fe" if is_rainy else "#f8fafc"

    fig, ax = plt.subplots(figsize=(13, 8.2), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # 3.1 Road connections
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
                ax.plot([x1, x2], [y1, y2], color="#eab308", linestyle="--",
                        linewidth=2.6, alpha=0.9, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], color="#64748b", linestyle="-",
                        linewidth=2.2, alpha=0.8, zorder=1)

    # 3.2 Robot's travelled route
    for i in range(len(visited_nodes) - 1):
        x1, y1 = env.node_coords[visited_nodes[i]]
        x2, y2 = env.node_coords[visited_nodes[i + 1]]
        ax.plot([x1, x2], [y1, y2], color="#06b6d4", linewidth=5,
                alpha=0.95, zorder=3)

    # 3.3 Nodes + labels
    for node_id, (x, y) in env.node_coords.items():
        if node_id == current_node:
            color, size, edge_color, z_order = "#06b6d4", 120, "#0e7490", 6
        elif node_id == target_node:
            color, size, edge_color, z_order = "#10b981", 260, "#047857", 5
        elif node_id in env.start_locations:
            color, size, edge_color, z_order = "#f97316", 150, "#c2410c", 4
        elif node_id in visited_nodes:
            color, size, edge_color, z_order = "#0ea5e9", 120, "#0369a1", 4
        else:
            color, size, edge_color, z_order = "#cbd5e1", 85, "#64748b", 2

        ax.scatter(x, y, s=size, c=color, edgecolors=edge_color,
                   linewidths=1.8, zorder=z_order)

        hazard_tag = ""
        if node_id in env.pedestrian_zones and node_id in env.car_zones:
            hazard_tag = " [P/C]"
        elif node_id in env.pedestrian_zones:
            hazard_tag = " [P]"
        elif node_id in env.car_zones:
            hazard_tag = " [C]"

        # the robot icon sits on its node, so lift that node's label above it
        label_y = y + (5.2 if node_id == current_node else 2.2)
        ax.text(x, label_y, f"{env.node_names[node_id]}{hazard_tag}",
                fontsize=7.5, ha="center", va="bottom", color="#1e293b",
                bbox=dict(boxstyle="round,pad=0.18", fc="#ffffff",
                          ec="none", alpha=0.85),
                zorder=7)

    # 3.4 Moving obstacles: person and car icons, plus an arrow for the last move
    if show_obstacles:
        for kind, positions, make_icon, base_dx, base_dy, arrow_color, tag in (
            ("ped", env.pedestrian_positions, icon_person, -15, -12, "#a16207", "P"),
            ("car", env.car_positions, icon_car, 16, -10, "#b91c1c", "C"),
        ):
            for i, node in enumerate(positions):
                x, y = env.node_coords[node]
                dx, dy = base_dx + 11 * i, base_dy
                if prev_positions and prev_positions[kind][i] != node:
                    px, py = env.node_coords[prev_positions[kind][i]]
                    ax.annotate("", xy=(x, y), xytext=(px, py),
                                arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                                lw=2.2, shrinkA=7, shrinkB=9),
                                zorder=8)
                place_icon(ax, make_icon(), x, y, dx, dy, zorder=9)
                ax.annotate(f"{tag}{i + 1}", (x, y), xytext=(dx, dy - 22),
                            textcoords="offset points", ha="center", va="center",
                            fontsize=7, fontweight="bold", color="#0f172a",
                            bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff",
                                      ec="none", alpha=0.85),
                            zorder=10)

    # 3.5 Robot icon on its current node
    rx, ry = env.node_coords[current_node]
    place_icon(ax, icon_robot(), rx, ry, 0, 9, zorder=12)

    # 3.6 HUD (dark text on a white box so it is readable in light mode)
    weather_banner = "RAINY" if is_rainy else "SUNNY"
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

    # 3.7 Icon key (left margin, below the HUD)
    ax.add_patch(FancyBboxPatch((-27.5, 50), 21, 22, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc="#ffffff", ec="#cbd5e1", lw=1.2, alpha=0.95, zorder=11))
    ax.text(-26.5, 70.2, "ICON KEY", fontsize=8, fontweight="bold",
            color="#0f172a", va="center", zorder=12)
    for row_y, make_icon, label in ((63.5, icon_robot, "Robot"),
                                    (57.5, icon_person, "Pedestrian"),
                                    (52.5, icon_car, "Car")):
        place_icon(ax, make_icon(0.55), -23.5, row_y, 0, 0, zorder=12)
        ax.text(-19, row_y, label, fontsize=8, color="#1e293b",
                va="center", ha="left", zorder=12)

    # 3.8 Legend for nodes and paths
    legend_elements = [
        patches.Patch(facecolor="#10b981", edgecolor="#047857", label="Destination"),
        patches.Patch(facecolor="#f97316", edgecolor="#c2410c", label="Restaurant"),
        Line2D([0], [0], color="#06b6d4", lw=4, label="Robot Travelled Path"),
        Line2D([0], [0], color="#64748b", lw=2, label="Covered Walkway"),
        Line2D([0], [0], color="#eab308", lw=2, ls="--", label="Uncovered Path"),
        Line2D([0], [0], color="#b91c1c", lw=2, marker=">", markersize=6,
               label="Obstacle's last move"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", facecolor="#ffffff",
              edgecolor="#cbd5e1", labelcolor="#1e293b", fontsize=8)

    ax.set_xlim(-28, 102)
    ax.set_ylim(2, 103)
    ax.axis("off")
    plt.tight_layout()
    return fig


# ==========================================
# 4. ACTION POLICY
# ==========================================

def choose_action(env, state, model_name, q_table, previous_node):
    """Trained Q-table for 'Q-Learning'; random valid action otherwise
    (the other algorithms are not integrated yet)."""
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


# ==========================================
# 5. NAVIGATION CALLBACKS
# (callbacks run BEFORE the rerun, so a button never replays the simulation)
# ==========================================

def go_to_order():
    st.session_state.page = "order"


def go_to_tracking():
    st.session_state.page = "tracking"


if "page" not in st.session_state:
    st.session_state.page = "order"


# ==========================================
# VIEW 1: CUSTOMER ORDER SELECTION
# ==========================================

if st.session_state.page == "order":

    st.title("🍔 TAR UMT Autonomous Campus Food Delivery")
    st.markdown(
        "Select your delivery destination, restaurant, food and weather. "
        "The simulation shows the robot moving among pedestrians and cars "
        "on the campus map."
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📋 Step 1 – Configure Your Order")

        target_options = {env.node_names[n]: n for n in env.target_destinations}
        selected_target_name = st.selectbox(
            "🎯 Choose Delivery Destination:", list(target_options.keys()))
        chosen_target = target_options[selected_target_name]

        start_options = {env.node_names[n]: n for n in env.start_locations}
        selected_start_name = st.selectbox(
            "🏪 Choose Restaurant:", list(start_options.keys()))
        chosen_start = start_options[selected_start_name]

        chosen_food = st.selectbox(
            f"🍱 Select Food from {selected_start_name}:",
            env.food_menus[chosen_start])

        st.markdown("---")
        st.subheader("🤖 Step 2 – Select Reinforcement Learning Model")
        selected_model = st.selectbox(
            "Choose an AI model:",
            ["Q-Learning", "SARSA", "DQN", "A2C", "REINFORCE"])

        st.markdown("---")
        st.subheader("🌦️ Step 3 – Weather")
        weather_mode = st.radio(
            "Weather Condition:",
            ["🎲 Auto Random Weather", "☀️ Force Sunny", "🌧️ Force Rainy"],
            horizontal=True)

        st.write("")

        if st.button("🚀 Confirm Order & Launch Live Tracking"):
            if weather_mode == "☀️ Force Sunny":
                weather_value = 0
            elif weather_mode == "🌧️ Force Rainy":
                weather_value = 1
            else:
                weather_value = random.choice([0, 1])

            st.session_state.order_config = {
                "start": chosen_start,
                "target": chosen_target,
                "food": chosen_food,
                "weather": weather_value,
            }
            st.session_state.selected_model = selected_model
            st.session_state.page = "tracking"
            st.rerun()

    with col2:
        st.subheader("🗺️ Campus Map Preview")

        preview_env = CampusDeliveryEnv(obstacle_mode=OBSTACLE_MODE)
        preview_env.current_location = chosen_start
        preview_env.target_destination = chosen_target
        preview_env.current_weather = 1 if weather_mode == "🌧️ Force Rainy" else 0
        preview_env.selected_food = chosen_food

        preview_fig = render_campus_map(
            preview_env, [chosen_start], chosen_start, chosen_target,
            score=0, step_num=0, status_text="Waiting for Order",
            show_obstacles=False)
        st.pyplot(preview_fig)
        plt.close(preview_fig)


# ==========================================
# VIEW 2: LIVE ROBOT TRACKING
# ==========================================

elif st.session_state.page == "tracking":

    if "order_config" not in st.session_state:
        st.session_state.page = "order"
        st.rerun()

    cfg = st.session_state.order_config
    selected_model = st.session_state.get("selected_model", "Q-Learning")
    q_table = load_q_table()

    try:
        state, info = env.reset(options=cfg)
    except ValueError as err:
        note("error", f"Invalid order: {err}")
        st.button("🏠 Back to Order Selection", on_click=go_to_order)
        st.stop()

    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.subheader(
            f"📡 Live Robot Tracking: {env.node_names[env.current_location]} "
            f"➜ {env.node_names[env.target_destination]}")
    with top_col2:
        st.button("🔄 New Order", on_click=go_to_order)

    if selected_model == "Q-Learning" and q_table is not None:
        note("info", "🤖 Model: Q-Learning (trained Q-table loaded from q_table.npy)")
    elif selected_model == "Q-Learning":
        note("warning", "🤖 q_table.npy not found next to app.py, so a random valid-action "
                        "policy is used. Run the training script first.")
    else:
        note("warning", f"🤖 {selected_model} is not integrated yet, so a random "
                        "valid-action policy is used.")

    hud_placeholder = st.empty()
    map_placeholder = st.empty()
    intent_placeholder = st.empty()
    with st.expander("📜 Environment Simulation Log", expanded=True):
        log_placeholder = st.empty()

    total_score = 0
    step_num = 0
    terminated = truncated = False
    previous_node = None
    visited_nodes = [env.current_location]
    logs = []

    fig = render_campus_map(env, visited_nodes, env.current_location,
                            env.target_destination, total_score, step_num,
                            "Order Dispatched")
    map_placeholder.pyplot(fig)
    plt.close(fig)
    intent_placeholder.markdown(describe_obstacles(env))
    time.sleep(0.8)

    reward = 0
    while not (terminated or truncated):
        step_num += 1

        action = choose_action(env, state, selected_model, q_table, previous_node)
        previous_location = env.current_location
        prev_positions = {"ped": list(env.pedestrian_positions),
                          "car": list(env.car_positions)}
        state, reward, terminated, truncated, info = env.step(action)
        total_score += reward

        outcome = info.get("outcome")
        moved = env.current_location != previous_location
        if moved:
            previous_node = previous_location
            visited_nodes.append(env.current_location)

        here = env.node_names[env.current_location]
        if outcome == "wait":
            status_msg, log_line = "Robot waiting", "WAIT"
        elif outcome == "retreat":
            blocked = env.node_names[info["blocked_node"]]
            status_msg = f"Blocked at {blocked}, holding position"
            log_line = f"RETREAT (blocked at {blocked})"
        elif outcome == "invalid":
            status_msg, log_line = "Invalid action", "INVALID action"
        elif outcome == "ped_collision":
            status_msg, log_line = f"Pedestrian collision at {here}", f"MOVE to {here} -> PEDESTRIAN COLLISION"
        elif outcome == "car_collision":
            status_msg, log_line = f"Car collision at {here}", f"MOVE to {here} -> CAR COLLISION"
        elif outcome == "goal":
            status_msg, log_line = f"Delivered at {here}", f"MOVE to {here} -> DELIVERED"
        else:
            status_msg, log_line = f"Moved to {here}", f"MOVE to {here}"

        logs.append(f"Step {step_num}: {log_line} | Reward: {reward:+} pts")

        with hud_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Current Score", f"{total_score} pts", f"{reward:+} pts")
            m2.metric("👣 Steps Taken", f"{step_num}")
            m3.metric("🌦️ Weather",
                      "Rainy 🌧️" if env.current_weather == 1 else "Sunny ☀️")
            m4.metric("🚧 Moving Obstacles",
                      f"{len(env.pedestrian_positions) + len(env.car_positions)}")

        fig = render_campus_map(env, visited_nodes, env.current_location,
                                env.target_destination, total_score,
                                step_num, status_msg,
                                prev_positions=prev_positions)
        map_placeholder.pyplot(fig)
        plt.close(fig)
        intent_placeholder.markdown(describe_obstacles(env))

        log_placeholder.markdown("\n\n".join(logs[-10:]))
        time.sleep(STEP_DELAY)

    # ======================================
    # FINAL RESULT (based on the real outcome, not just the location)
    # ======================================

    final_outcome = info.get("outcome")

    if final_outcome == "goal":
        note("success",
             f"🎉 DELIVERY SUCCESSFUL! Delivered {env.selected_food} to "
             f"{env.node_names[env.target_destination]} in {step_num} steps. "
             f"Final Score: {total_score} pts")
        st.balloons()

    elif final_outcome in ("ped_collision", "car_collision"):
        kind = "pedestrian" if final_outcome == "ped_collision" else "car"
        note("error",
             f"🚨 DELIVERY FAILED! The robot collided with a {kind} at "
             f"{env.node_names[env.current_location]}. "
             f"Final Score: {total_score} pts")

    elif truncated:
        note("warning", f"⏰ TIME LIMIT REACHED after {step_num} steps. "
                        f"Final Score: {total_score} pts")

    st.button("🏠 Back to Order Selection", on_click=go_to_order)
