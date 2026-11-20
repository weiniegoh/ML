
# ==========================================
# 1. IMPORT REQUIRED LIBRARIES
# ==========================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import heapq


# ==========================================
# 2. CAMPUS DELIVERY ENVIRONMENT
# ==========================================

class CampusDeliveryEnv(gym.Env):

    def __init__(self, max_steps=80):
        super().__init__()

        self.max_steps = max_steps
        self.steps = 0

        # --------------------------------------
        # 2.1 NODE NAMES
        # --------------------------------------

        self.all_nodes = list(range(36))

        self.node_names = {
            0: "Red Bricks",
            1: "DK4",
            2: "Rimba",
            3: "Block C",
            4: "Yum Yum",
            5: "Arena",
            6: "Block M",
            7: "Library",
            8: "Roof Top D",
            9: "Hostel",
            10: "Block Q (West Slope)",
            11: "DK W",
            12: "Carpark C",
            13: "Roof Top C",
            14: "DKABE",
            15: "Roof Top E",
            16: "Clubhouse",
            17: "Block W",
            18: "Block V",
            19: "DTAR",
            20: "Swimming Pool",
            21: "Sport Complex",
            22: "FM (CITC)",
            23: "Roof Top A",
            24: "Bus Stop A",
            25: "Carpark A",
            26: "DSA",
            27: "Basketball Court",
            28: "Roof Top B",
            29: "Car Park B",
            30: "Casaurina Cafe",
            31: "East Slope",
            32: "Bus Stop B",
            33: "Roof Top F",
            34: "Roof Top G",
            35: "Carpark D"
        }

        # --------------------------------------
        # 2.2 NODE COORDINATES
        # --------------------------------------

        self.node_coords = {
            0: (6, 48),   1: (15, 48),  2: (21, 38),  3: (32, 38),
            4: (43, 38),  5: (58, 25),  6: (43, 47),  7: (43, 56),
            8: (60, 90),  9: (80, 92),  10: (22, 80), 11: (35, 83),
            12: (71, 25), 13: (69, 34), 14: (73, 43), 15: (76, 68),
            16: (64, 57), 17: (67, 50), 18: (67, 43), 19: (63, 37),
            20: (63, 64), 21: (63, 72), 22: (15, 23), 23: (12, 12),
            24: (25, 9),  25: (26, 23), 26: (35, 24), 27: (45, 16),
            28: (51, 33), 29: (54, 56), 30: (94, 61), 31: (93, 52),
            32: (87, 45), 33: (82, 52), 34: (81, 41), 35: (84, 76)
        }

        # --------------------------------------
        # 2.3 BIDIRECTIONAL CAMPUS GRAPH
        # --------------------------------------

        self.graph = {
            0: [1],
            1: [0, 2, 10],
            2: [1, 3, 26],
            3: [2, 4],
            4: [3, 6, 26, 27, 28],
            5: [12, 13, 19, 27, 28],
            6: [4, 7],
            7: [6, 8, 29],
            8: [7, 9, 11, 15, 21, 35],
            9: [8, 15, 35],
            10: [1, 11],
            11: [10, 8],
            12: [5, 14],
            13: [5, 14],
            14: [12, 13, 15, 33, 34],
            15: [8, 9, 14, 33],
            16: [17, 20, 29],
            17: [16, 18],
            18: [17, 19],
            19: [18, 5],
            20: [16, 21],
            21: [20, 8],
            22: [23, 25],
            23: [22, 24],
            24: [23, 26, 27],
            25: [22, 26],
            26: [2, 4, 24, 25, 27],
            27: [4, 5, 24, 26],
            28: [4, 5],
            29: [7, 16],
            30: [31, 35],
            31: [30, 32, 33],
            32: [31, 34],
            33: [14, 15, 31],
            34: [14, 32],
            35: [8, 9, 30]
        }

        # Validate graph connections
        for node, neighbours in self.graph.items():
            for neighbour in neighbours:
                assert node in self.graph[neighbour], (
                    f"Graph is not bidirectional: {node} -> {neighbour}"
                )

        assert set(self.graph.keys()) == set(self.all_nodes)
        assert set(self.node_coords.keys()) == set(self.all_nodes)

        # --------------------------------------
        # 2.4 RESTAURANTS AND DESTINATIONS
        # --------------------------------------

        self.start_locations = [0, 4, 16, 22, 30]

        self.target_destinations = [5, 9, 2]

        self.food_menus = {
            0: ["Nasi Lemak", "Roti Canai", "Teh Tarik"],
            4: ["Chicken Rice", "Wantan Mee", "Mixed Rice"],
            16: ["Club Sandwich", "Burger & Fries", "Iced Lemon Tea"],
            22: ["Japanese Bento", "Onigiri", "Oden"],
            30: ["Western Chicken Chop", "Fried Kuey Teow", "Iced Coffee"]
        }

        # --------------------------------------
        # 2.5 WEATHER AND UNCOVERED PATHS
        # --------------------------------------

        self.uncovered_paths = {
            frozenset({1, 10}),
            frozenset({10, 11}),
            frozenset({27, 5}),
            frozenset({5, 12}),
            frozenset({12, 14}),
            frozenset({22, 25}),
            frozenset({25, 26}),
            frozenset({7, 29}),
            frozenset({29, 16}),
            frozenset({30, 35}),
            frozenset({35, 9}),
            frozenset({35, 8})
        }

        # --------------------------------------
        # 2.6 PEDESTRIAN AND CAR ZONES
        # --------------------------------------

        self.crowd_level = {
            7: 1.5,
            6: 1.5,
            3: 1.5,
            26: 1.5,
            14: 1.5,
            4: 1.8
        }

        self.pedestrian_zones = {
            0, 1, 3, 4, 5, 6, 7, 14, 16,
            20, 21, 24, 26, 27, 28, 30, 31, 32
        }

        self.car_zones = {
            3, 4, 5, 8, 10, 12, 21, 23,
            25, 26, 27, 28, 29, 33, 34, 35
        }

        # --------------------------------------
        # 2.7 MOVING OBSTACLES
        # --------------------------------------

        self.pedestrian_routes = [
            [3, 4]
        ]

        self.car_routes = [
            [4, 27, 5, 28]
        ]

        self.pedestrian_positions = [3, 6]
        self.car_positions = [4, 12]

        self.pedestrian_route_indices = [0]
        self.car_route_indices = [0]

        # --------------------------------------
        # 2.8 DISTANCE-BASED STEP COSTS
        # --------------------------------------

        self.longer_paths = {
            frozenset({23, 24}): -3,
            frozenset({24, 27}): -2,
            frozenset({1, 10}): -2,
            frozenset({7, 8}): -2,
            frozenset({14, 15}): -2,
            frozenset({30, 35}): -2
        }

        # --------------------------------------
        # 2.9 ACTION AND OBSERVATION SPACES
        # --------------------------------------

        # Actions 0-35: move to a neighbouring node
        # Action 36: wait

        self.action_space = spaces.Discrete(37)

        # Default mode for Q-learning and SARSA
        self.observation_mode = "tabular"

        self.observation_space = spaces.Tuple((
            spaces.Discrete(36),
            spaces.Discrete(36),
            spaces.Discrete(2),
            spaces.Discrete(3)
        ))

        # --------------------------------------
        # 2.10 INITIAL VARIABLES
        # --------------------------------------

        self.current_location = None
        self.target_destination = None
        self.current_weather = None
        self.current_traffic = None
        self.selected_food = None

    # ==========================================
    # 3. HELPER FUNCTIONS
    # ==========================================

    def _choice(self, seq):
        return seq[int(self.np_random.integers(len(seq)))]

    def valid_actions(self, location=None):
        loc = self.current_location if location is None else location

        if loc not in self.graph:
            raise ValueError("Invalid location.")

        return self.graph[loc] + [36]

    def obstacle_prob(self, node):
        base = 0.25 if self.current_weather == 0 else 0.35

        return min(
            0.6,
            base * self.crowd_level.get(node, 1.0)
        )

    def edge_cost(self, u, v):
        edge = frozenset({u, v})

        cost = -self.longer_paths.get(edge, -1)

        if self.current_weather == 1 and edge in self.uncovered_paths:
            cost += 15

        if (
            v != self.target_destination
            and (v in self.pedestrian_zones or v in self.car_zones)
        ):
            p = self.obstacle_prob(v)
            cost += 2 * p / (1 - p)

        return cost

    def optimal_action(self):
        """Dijkstra-based reference policy, not an RL agent."""

        dist = {
            node: float("inf")
            for node in self.all_nodes
        }

        dist[self.target_destination] = 0
        pq = [(0, self.target_destination)]

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v in self.graph[u]:
                nd = d + self.edge_cost(v, u)

                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        loc = self.current_location

        return min(
            self.graph[loc],
            key=lambda n: self.edge_cost(loc, n) + dist[n]
        )

    # ==========================================
    # 4. OBSERVATION FUNCTIONS
    # ==========================================

    def set_observation_mode(self, mode):
        if mode not in ["tabular", "full"]:
            raise ValueError("Mode must be 'tabular' or 'full'.")

        self.observation_mode = mode

        if mode == "tabular":
            self.observation_space = spaces.Tuple((
                spaces.Discrete(36),
                spaces.Discrete(36),
                spaces.Discrete(2),
                spaces.Discrete(3)
            ))

        else:
            self.observation_space = spaces.Tuple((
                spaces.Discrete(36),
                spaces.Discrete(36),
                spaces.Discrete(2),
                spaces.Discrete(36),
                spaces.Discrete(36),
                spaces.Discrete(36),
                spaces.Discrete(36)
            ))

    def _get_tabular_obs(self):
        nearby_nodes = (
            [self.current_location]
            + self.graph[self.current_location]
        )

        pedestrian_nearby = any(
            pos in nearby_nodes
            for pos in self.pedestrian_positions
        )

        car_nearby = any(
            pos in nearby_nodes
            for pos in self.car_positions
        )

        if car_nearby:
            traffic = 2
        elif pedestrian_nearby:
            traffic = 1
        else:
            traffic = 0

        return (
            self.current_location,
            self.target_destination,
            self.current_weather,
            traffic
        )

    def _get_full_obs(self):
        return (
            self.current_location,
            self.target_destination,
            self.current_weather,
            self.pedestrian_positions[0],
            self.pedestrian_positions[1],
            self.car_positions[0],
            self.car_positions[1]
        )

    def _get_obs(self):
        if self.observation_mode == "tabular":
            return self._get_tabular_obs()

        return self._get_full_obs()

    def _get_info(self, outcome="reset", rain_violation=False):
        traffic_map = {
            0: "Clear",
            1: "Pedestrian Detected",
            2: "Car Detected"
        }

        return {
            "location_str": self.node_names[self.current_location],
            "target_str": self.node_names[self.target_destination],
            "weather_str": (
                "Sunny" if self.current_weather == 0 else "Rainy"
            ),
            "traffic_str": traffic_map[self.current_traffic],
            "food_item": self.selected_food,
            "outcome": outcome,
            "rain_violation": rain_violation,
            "steps": self.steps
        }


    # ======================================
    # OBSTACLE FUNCTIONS
    # ======================================

    # Copy your latest:
    def _move_fixed_obstacles(self):
        """Move fixed-route obstacles to their next positions."""

        # Move fixed-route pedestrian
        pedestrian_route = self.pedestrian_routes[0]

        self.pedestrian_route_indices[0] = (
            self.pedestrian_route_indices[0] + 1
        ) % len(pedestrian_route)

        self.pedestrian_positions[0] = pedestrian_route[
            self.pedestrian_route_indices[0]
        ]

        # Move fixed-route car
        car_route = self.car_routes[0]

        self.car_route_indices[0] = (
            self.car_route_indices[0] + 1
        ) % len(car_route)

        self.car_positions[0] = car_route[
            self.car_route_indices[0]
        ]


    def _move_random_obstacles(self):
        """Move random obstacles without overlapping fixed obstacles."""

        # Random-moving pedestrian
        pedestrian_neighbours = [
            node for node in self.graph[self.pedestrian_positions[1]]
            if (
                node in self.pedestrian_zones
                and node != self.pedestrian_positions[0]
            )
        ]

        if pedestrian_neighbours:
            self.pedestrian_positions[1] = self._choice(
                pedestrian_neighbours
            )

        # Random-moving car
        car_neighbours = [
            node for node in self.graph[self.car_positions[1]]
            if (
                node in self.car_zones
                and node != self.car_positions[0]
            )
        ]

        if car_neighbours:
            self.car_positions[1] = self._choice(car_neighbours)
    # Any other obstacle-related functions

        # ======================================
    # ENVIRONMENT FUNCTIONS
    # ======================================

    def reset(self, seed=None, options=None):
        """
        Resets the campus delivery environment.
        """

        super().reset(seed=seed)

        self.steps = 0

        # Reset moving obstacle positions
        self.pedestrian_positions = [3, 6]
        self.car_positions = [4, 12]

        # Reset fixed-route obstacle indices
        self.pedestrian_route_indices = [0]
        self.car_route_indices = [0]

        options = {} if options is None else options

        if not isinstance(options, dict):
            raise TypeError("Options must be a dictionary.")

        allowed_options = {"start", "target", "food", "weather"}

        unknown_options = set(options.keys()) - allowed_options

        if unknown_options:
            raise ValueError(
                f"Unknown reset options: {unknown_options}"
            )

        # Validate starting restaurant
        if "start" in options:
            if options["start"] not in self.start_locations:
                raise ValueError("Invalid starting restaurant.")

        # Validate destination
        if "target" in options:
            if options["target"] not in self.target_destinations:
                raise ValueError("Invalid target destination.")

        # Validate weather
        if "weather" in options:
            if options["weather"] not in (0, 1):
                raise ValueError(
                    "Weather must be 0 (Sunny) or 1 (Rainy)."
                )

        # Select destination
        self.target_destination = (
            options["target"]
            if "target" in options
            else self._choice(self.target_destinations)
        )

        # Select restaurant
        self.current_location = (
            options["start"]
            if "start" in options
            else self._choice(self.start_locations)
        )

        # Select weather
        self.current_weather = (
            options["weather"]
            if "weather" in options
            else self._choice([0, 1])
        )

        # Select food
        if "food" in options:
            self.selected_food = options["food"]

            if self.selected_food not in self.food_menus[self.current_location]:
                raise ValueError(
                    "Food item is not available at the selected restaurant."
                )
        else:
            self.selected_food = self._choice(
                self.food_menus[self.current_location]
            )

        # Initial traffic detection
        self.current_traffic = self._get_tabular_obs()[3]

        return self._get_obs(), self._get_info("reset")

    def step(self, action):
        """Executes one action and updates moving obstacles."""

        reward = 0
        terminated = False
        truncated = False

        outcome = "move"
        rain_violation = False

        action = int(action)
        self.steps += 1

        # Save obstacle positions before they move
        old_pedestrian_positions = self.pedestrian_positions.copy()
        old_car_positions = self.car_positions.copy()

        previous_location = self.current_location

        # ------------------------------------------
        # PROCESS ROBOT ACTION
        # ------------------------------------------

        if action == 36:
            reward = -2
            outcome = "wait"

        elif action not in self.graph[self.current_location]:
            reward = -5
            outcome = "invalid"

        else:
            path_edge = frozenset({
                self.current_location,
                action
            })

            reward += self.longer_paths.get(path_edge, -1)

            if (
                self.current_weather == 1
                and path_edge in self.uncovered_paths
            ):
                reward -= 15
                rain_violation = True

            self.current_location = action

        # ------------------------------------------
        # MOVE ALL FOUR OBSTACLES
        # ------------------------------------------

        self._move_fixed_obstacles()
        self._move_random_obstacles()

        # ------------------------------------------
        # CHECK NODE COLLISIONS
        # ------------------------------------------

        if self.current_location in self.pedestrian_positions:
            reward -= 25
            terminated = True
            outcome = "ped_collision"

        elif self.current_location in self.car_positions:
            reward -= 50
            terminated = True
            outcome = "car_collision"

        # ------------------------------------------
        # CHECK EDGE-SWAP COLLISIONS
        # ------------------------------------------

        if not terminated:

            for old_pos, new_pos in zip(
                old_pedestrian_positions,
                self.pedestrian_positions
            ):
                if (
                    previous_location == new_pos
                    and self.current_location == old_pos
                ):
                    reward -= 25
                    terminated = True
                    outcome = "ped_collision"
                    break

            if not terminated:
                for old_pos, new_pos in zip(
                    old_car_positions,
                    self.car_positions
                ):
                    if (
                        previous_location == new_pos
                        and self.current_location == old_pos
                    ):
                        reward -= 50
                        terminated = True
                        outcome = "car_collision"
                        break

        # ------------------------------------------
        # CHECK DESTINATION
        # ------------------------------------------

        if (
            not terminated
            and self.current_location == self.target_destination
        ):
            reward += 100
            terminated = True
            outcome = "goal"

        # ------------------------------------------
        # UPDATE TRAFFIC INFORMATION
        # ------------------------------------------

        self.current_traffic = self._get_tabular_obs()[3]

        # ------------------------------------------
        # CHECK MAXIMUM STEP LIMIT
        # ------------------------------------------

        if not terminated and self.steps >= self.max_steps:
            truncated = True

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(outcome, rain_violation)
        )
