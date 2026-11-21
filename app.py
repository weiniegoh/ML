
import streamlit as st
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from campus_environment import CampusDeliveryEnv


# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================

st.set_page_config(
    page_title="TAR UMT AI Food Delivery",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }

    [data-testid="stHeader"] {
        background-color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }

    .stMarkdown, .stText, label, p, h1, h2, h3 {
        color: #1e293b;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0d9488 0%, #06b6d4 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        border: none;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #0f766e 0%, #0891b2 100%);
        color: white;
    }

    .hud-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #06b6d4;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. INITIALIZE NEW ENVIRONMENT
# ==========================================

env = CampusDeliveryEnv()

# Add the 2D coordinates if they are not already
# included in your campus_environment.py file.
#
# IMPORTANT:
# The coordinates must be inside the environment
# class's __init__() method, not here.


# ==========================================
# 3. 2D CAMPUS MAP RENDERER
# ==========================================

def render_campus_map(
    env,
    visited_nodes,
    current_node,
    target_node,
    score,
    step_num,
    status_text
):

    is_rainy = env.current_weather == 1

    bg_color = "#e0f2fe" if is_rainy else "#f8fafc"

    fig, ax = plt.subplots(
        figsize=(13, 8.2),
        facecolor=bg_color
    )

    ax.set_facecolor(bg_color)

    # --------------------------------------
    # 3.1 Draw road connections
    # --------------------------------------

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

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="#facc15",
                    linestyle="--",
                    linewidth=2.6,
                    alpha=0.85,
                    zorder=1
                )

            else:

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="#94a3b8",
                    linestyle="-",
                    linewidth=2.2,
                    alpha=0.8,
                    zorder=1
                )

    # --------------------------------------
    # 3.2 Draw robot's travelled route
    # --------------------------------------

    if len(visited_nodes) > 1:

        for i in range(len(visited_nodes) - 1):

            u = visited_nodes[i]
            v = visited_nodes[i + 1]

            x1, y1 = env.node_coords[u]
            x2, y2 = env.node_coords[v]

            ax.plot(
                [x1, x2],
                [y1, y2],
                color="#06b6d4",
                linewidth=5,
                alpha=0.95,
                zorder=3
            )

    # --------------------------------------
    # 3.3 Draw all 36 campus nodes
    # --------------------------------------

    for node_id, (x, y) in env.node_coords.items():

        if node_id == current_node:

            color = "#22d3ee"
            size = 260
            edge_color = "#ffffff"
            z_order = 6

        elif node_id == target_node:

            color = "#10b981"
            size = 230
            edge_color = "#ffffff"
            z_order = 5

        elif node_id in env.start_locations:

            color = "#f97316"
            size = 150
            edge_color = "#fed7aa"
            z_order = 4

        elif node_id in visited_nodes:

            color = "#0ea5e9"
            size = 120
            edge_color = "#bae6fd"
            z_order = 4

        else:

            color = "#cbd5e1"
            size = 85
            edge_color = "#64748b"
            z_order = 2

        ax.scatter(
            x, y,
            s=size,
            c=color,
            edgecolors=edge_color,
            linewidths=1.8,
            zorder=z_order
        )

        # Hazard zone labels

        hazard_tag = ""

        if node_id in env.pedestrian_zones and node_id in env.car_zones:
            hazard_tag = " [P/C]"

        elif node_id in env.pedestrian_zones:
            hazard_tag = " [P]"

        elif node_id in env.car_zones:
            hazard_tag = " [C]"

        label_color = (
            "#ffffff"
            if node_id in [current_node, target_node]
            else "#334155"
        )

        ax.text(
            x,
            y + 2.2,
            f"{env.node_names[node_id]}{hazard_tag}",
            fontsize=7.5,
            ha="center",
            va="bottom",
            color=label_color,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="#ffffff",
                ec="none",
                alpha=0.75
            ),
            zorder=7
        )

    # --------------------------------------
    # 3.4 Draw moving pedestrians
    # --------------------------------------

    pedestrian_positions = env.pedestrian_positions

    for i, node in enumerate(pedestrian_positions):

        x, y = env.node_coords[node]

        # Small offsets help distinguish obstacles
        # when two obstacles occupy the same node.

        offset = (i - (len(pedestrian_positions) - 1) / 2) * 1.5

        ax.scatter(
            x + offset,
            y + 0.8,
            s=240,
            marker="o",
            color="#facc15",
            edgecolors="white",
            linewidths=1.8,
            zorder=9
        )

        ax.text(
            x + offset,
            y + 0.8,
            f"P{i + 1}",
            color="#111827",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10
        )

    # --------------------------------------
    # 3.5 Draw moving cars
    # --------------------------------------

    car_positions = env.car_positions

    for i, node in enumerate(car_positions):

        x, y = env.node_coords[node]

        offset = (i - (len(car_positions) - 1) / 2) * 1.5

        ax.scatter(
            x + offset,
            y - 0.8,
            s=250,
            marker="s",
            color="#ef4444",
            edgecolors="white",
            linewidths=1.8,
            zorder=9
        )

        ax.text(
            x + offset,
            y - 0.8,
            f"C{i + 1}",
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10
        )

    # --------------------------------------
    # 3.6 Live information HUD
    # --------------------------------------

    weather_banner = (
        "RAINY WEATHER"
        if is_rainy
        else "SUNNY WEATHER"
    )

    obstacle_text = (
        f"Pedestrians : {pedestrian_positions}\n"
        f"Cars        : {car_positions}"
    )

    hud_text = (
        f"LIVE DELIVERY ENVIRONMENT\n"
        f"──────────────────────────\n"
        f"Score       : {score} pts\n"
        f"Steps Taken : {step_num}\n"
        f"Weather     : {weather_banner}\n"
        f"Current Loc : {env.node_names[current_node]}\n"
        f"Target Dest : {env.node_names[target_node]}\n"
        f"Order Item  : {env.selected_food}\n"
        f"{obstacle_text}\n"
        f"Status      : {status_text}"
    )

    ax.text(
        1.5,
        98.5,
        hud_text,
        fontsize=8.5,
        color="#f8fafc",
        family="monospace",
        va="top",
        ha="left",
        zorder=11,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#ffffff",
            edgecolor="#06b6d4",
            linewidth=2,
            alpha=0.92
        )
    )

    # --------------------------------------
    # 3.7 Map legend
    # --------------------------------------

    legend_elements = [

        patches.Patch(
            facecolor="#22d3ee",
            edgecolor="white",
            label="Robot"
        ),

        patches.Patch(
            facecolor="#10b981",
            edgecolor="white",
            label="Destination"
        ),

        patches.Patch(
            facecolor="#f97316",
            edgecolor="white",
            label="Restaurant"
        ),

        patches.Patch(
            facecolor="#facc15",
            edgecolor="white",
            label="Pedestrian"
        ),

        patches.Patch(
            facecolor="#ef4444",
            edgecolor="white",
            label="Car"
        ),

        plt.Line2D(
            [0], [0],
            color="#06b6d4",
            lw=4,
            label="Robot Travelled Path"
        ),

        plt.Line2D(
            [0], [0],
            color="#475569",
            lw=2,
            label="Covered Walkway"
        ),

        plt.Line2D(
            [0], [0],
            color="#facc15",
            lw=2,
            ls="--",
            label="Uncovered Path"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower right",
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        labelcolor="#1e293b",
        fontsize=8
    )

    ax.set_xlim(0, 102)
    ax.set_ylim(2, 103)
    ax.axis("off")

    plt.tight_layout()

    return fig


# ==========================================
# 4. TEMPORARY DEMONSTRATION POLICY
# ==========================================

def choose_demo_action(env):

    """
    Temporary random valid-action policy.

    This is NOT an RL algorithm.
    It is only used to demonstrate the environment
    before integrating the five RL algorithms.
    """

    valid_actions = env.graph[env.current_location] + [36]

    return random.choice(valid_actions)


# ==========================================
# 5. STREAMLIT APPLICATION FLOW
# ==========================================

if "page" not in st.session_state:
    st.session_state.page = "order"


# ==========================================
# VIEW 1: CUSTOMER ORDER SELECTION
# ==========================================

if st.session_state.page == "order":

    st.title("🍔 TAR UMT Autonomous Campus Food Delivery")

    st.markdown(
        """
        Select your delivery destination, restaurant,
        food and weather condition.

        The simulation will display the robot's movement
        and the four moving obstacles on the campus map.
        """
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.subheader("📋 Step 1 – Configure Your Order")

        # Destination selection

        target_options = {
            env.node_names[nid]: nid
            for nid in env.target_destinations
        }

        selected_target_name = st.selectbox(
            "🎯 1. Choose Delivery Destination:",
            list(target_options.keys())
        )

        chosen_target = target_options[selected_target_name]

        # Restaurant selection

        start_options = {
            env.node_names[nid]: nid
            for nid in env.start_locations
        }

        selected_start_name = st.selectbox(
            "🏪 2. Choose Restaurant:",
            list(start_options.keys())
        )

        chosen_start = start_options[selected_start_name]

        # Food selection

        available_menu = env.food_menus[chosen_start]

        chosen_food = st.selectbox(
            f"🍱 3. Select Food from {selected_start_name}:",
            available_menu
        )

        # Weather selection

        # RL Model Selection

        st.markdown("---")
        
        st.subheader("🤖 Step 2 – Select Reinforcement Learning Model")
        
        selected_model = st.selectbox(
            "Choose an AI model:",
            [
                "Q-Learning",
                "SARSA",
                "DQN",
                "A2C",
                "REINFORCE"
            ]
        )
        
        # Weather selection
        
        weather_mode = st.radio(
            "🌦️ 3. Weather Condition:",
            [
                "🎲 Auto Random Weather",
                "☀️ Force Sunny",
                "🌧️ Force Rainy"
            ],
            horizontal=True
        )

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
                "model": selected_model
            }

            st.session_state.page = "tracking"

            st.rerun()

    # --------------------------------------
    # Campus map preview
    # --------------------------------------

    with col2:

        st.subheader("🗺️ Campus Map Preview")

        preview_env = CampusDeliveryEnv()

        preview_env.current_location = chosen_start
        preview_env.target_destination = chosen_target
        preview_env.current_weather = 0
        preview_env.selected_food = chosen_food

        preview_fig = render_campus_map(
            preview_env,
            visited_nodes=[chosen_start],
            current_node=chosen_start,
            target_node=chosen_target,
            score=0,
            step_num=0,
            status_text="Waiting for Order"
        )

        st.pyplot(preview_fig)

        plt.close(preview_fig)


# ==========================================
# VIEW 2: LIVE ROBOT TRACKING
# ==========================================

elif st.session_state.page == "tracking":

    cfg = st.session_state.order_config
    selected_model = cfg["model"]
    # Reset the latest environment using the
    # customer's selected order.

    state, info = env.reset(options=cfg)

    top_col1, top_col2 = st.columns([4, 1])

    with top_col1:

        st.subheader(
            f"📡 Live Robot Tracking: "
            f"{env.node_names[env.current_location]} "
            f"➜ "
            f"{env.node_names[env.target_destination]}"
        )

st.info(f"🤖 Selected RL Model: {selected_model}")

    with top_col2:

        if st.button("🔄 New Order"):

            st.session_state.page = "order"

            st.rerun()

    hud_placeholder = st.empty()

    map_placeholder = st.empty()

    log_expander = st.expander(
        "📜 View Environment Simulation Log",
        expanded=True
    )

    total_score = 0
    step_num = 0

    terminated = False
    truncated = False

    visited_nodes = [env.current_location]

    logs = []

    # --------------------------------------
    # Initial map frame
    # --------------------------------------

    fig = render_campus_map(
        env,
        visited_nodes,
        env.current_location,
        env.target_destination,
        total_score,
        step_num,
        "Order Dispatched"
    )

    map_placeholder.pyplot(fig)

    plt.close(fig)

    time.sleep(0.8)

    # --------------------------------------
    # Environment simulation loop
    # --------------------------------------

    while not terminated and not truncated:

        if step_num >= env.max_steps:
            break

        step_num += 1

        # Select a temporary valid action.
        # No RL algorithm is used at this stage.

        action = choose_demo_action(env)

        previous_location = env.current_location

        # Execute one environment step.

        next_state, reward, terminated, truncated, info = env.step(action)

        total_score += reward

        state = next_state

        # Determine movement status.

        if action == 36:

            status_msg = "Robot Waiting"

            logs.append(
                f"Step {step_num}: "
                f"Robot selected WAIT | "
                f"Reward: {reward} pts"
            )

        else:

            new_location = env.current_location

            if new_location != previous_location:

                visited_nodes.append(new_location)

            status_msg = (
                f"Moved to {env.node_names[env.current_location]}"
            )

            logs.append(
                f"Step {step_num}: "
                f"Moved to {env.node_names[env.current_location]} | "
                f"Reward: {reward:+} pts"
            )

        # ----------------------------------
        # Update metric cards
        # ----------------------------------

        with hud_placeholder.container():

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "🏆 Current Score",
                f"{total_score} pts",
                f"{reward:+} pts"
            )

            m2.metric(
                "👣 Steps Taken",
                f"{step_num}"
            )

            m3.metric(
                "🌦️ Weather",
                "Rainy 🌧️"
                if env.current_weather == 1
                else "Sunny ☀️"
            )

            m4.metric(
                "🚧 Moving Obstacles",
                f"{len(env.pedestrian_positions) + len(env.car_positions)}"
            )

        # ----------------------------------
        # Update 2D campus map
        # ----------------------------------

        fig = render_campus_map(
            env,
            visited_nodes,
            env.current_location,
            env.target_destination,
            total_score,
            step_num,
            status_msg
        )

        map_placeholder.pyplot(fig)

        plt.close(fig)

        # ----------------------------------
        # Update simulation log
        # ----------------------------------

        with log_expander:

            st.markdown(
                "<br>".join(logs[-10:]),
                unsafe_allow_html=True
            )

        time.sleep(0.6)

    # ======================================
    # 6. FINAL SIMULATION RESULT
    # ======================================

    if terminated and env.current_location == env.target_destination:

        st.success(
            f"🎉 DELIVERY SUCCESSFUL! "
            f"Delivered {env.selected_food} to "
            f"{env.node_names[env.target_destination]} "
            f"in {step_num} steps. "
            f"Final Score: {total_score} pts"
        )

        st.balloons()

    elif terminated:

        st.error(
            f"🚨 DELIVERY TERMINATED! "
            f"The robot encountered a collision or "
            f"another terminal condition. "
            f"Final Score: {total_score} pts"
        )

    elif truncated:

        st.warning(
            f"⏰ SIMULATION TIME LIMIT REACHED! "
            f"Final Score: {total_score} pts"
        )

    else:

        st.warning(
            f"Simulation stopped. "
            f"Final Score: {total_score} pts"
        )

    if st.button("🏠 Back to Order Selection"):

        st.session_state.page = "order"

        st.rerun()
