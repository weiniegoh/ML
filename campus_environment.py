import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import heapq


class CampusDeliveryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=80, obstacle_mode="random", contact_mode="retreat",
                 allow_wait=True, stay_prob=0.2, retreat_penalty=-3):
        super().__init__()
        if obstacle_mode not in ("random", "mixed"):
            raise ValueError("obstacle_mode must be 'random' or 'mixed'.")
        if contact_mode not in ("retreat", "collision"):
            raise ValueError("contact_mode must be 'retreat' or 'collision'.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer.")
        if isinstance(stay_prob, bool) or not isinstance(stay_prob, (int, float)) or not 0 <= stay_prob <= 1:
            raise ValueError("stay_prob must be between 0 and 1.")
        if isinstance(retreat_penalty, bool) or not isinstance(retreat_penalty, (int, float)):
            raise ValueError("retreat_penalty must be numeric.")
        if not isinstance(allow_wait, bool):
            raise ValueError("allow_wait must be True or False.")

        self.max_steps = max_steps
        self.obstacle_mode = obstacle_mode
        self.contact_mode = contact_mode
        self.allow_wait = allow_wait
        self.stay_prob = float(stay_prob)
        self.retreat_penalty = retreat_penalty

        self.steps = 0
        self._episode_over = False
        self.all_nodes = list(range(36))
        self.node_names = {0:"Red Bricks",1:"DK4",2:"Rimba",3:"Block C",4:"Yum Yum",5:"Arena",6:"Block M",7:"Library",8:"Roof Top D",9:"Hostel",10:"Block Q (West Slope)",11:"DK W",12:"Carpark C",13:"Roof Top C",14:"DKABE",15:"Roof Top E",16:"Clubhouse",17:"Block W",18:"Block V",19:"DTAR",20:"Swimming Pool",21:"Sport Complex",22:"FM (CITC)",23:"Roof Top A",24:"Bus Stop A",25:"Carpark A",26:"DSA",27:"Basketball Court",28:"Roof Top B",29:"Car Park B",30:"Casaurina Cafe",31:"East Slope",32:"Bus Stop B",33:"Roof Top F",34:"Roof Top G",35:"Carpark D"}
        self.node_coords = {
            0: (6, 48), 1: (15, 48), 2: (21, 38), 3: (32, 38),
            4: (43, 38), 5: (58, 25), 6: (43, 47), 7: (43, 56),
            8: (60, 90), 9: (80, 92), 10: (22, 80), 11: (35, 83),
            12: (71, 25), 13: (69, 34), 14: (73, 43), 15: (76, 68),
            16: (64, 57), 17: (67, 50), 18: (67, 43), 19: (63, 37),
            20: (63, 64), 21: (63, 72), 22: (15, 23), 23: (12, 12),
            24: (25, 9), 25: (26, 23), 26: (35, 24), 27: (45, 16),
            28: (51, 33), 29: (54, 56), 30: (94, 61), 31: (93, 52),
            32: (87, 45), 33: (82, 52), 34: (81, 41), 35: (84, 76)
        }
        self.graph = {0:[1],1:[0,2,10],2:[1,3,26],3:[2,4],4:[3,6,26,27,28],5:[12,13,19,27,28],6:[4,7],7:[6,8,29],8:[7,9,11,15,21,35],9:[8,15,35],10:[1,11],11:[10,8],12:[5,14],13:[5,14],14:[12,13,15,33,34],15:[8,9,14,33],16:[17,20,29],17:[16,18],18:[17,19],19:[18,5],20:[16,21],21:[20,8],22:[23,25],23:[22,24],24:[23,26,27],25:[22,26],26:[2,4,24,25,27],27:[4,5,24,26],28:[4,5],29:[7,16],30:[31,35],31:[30,32,33],32:[31,34],33:[14,15,31],34:[14,32],35:[8,9,30]}
        for node, neighbours in self.graph.items():
            for neighbour in neighbours:
                assert node in self.graph[neighbour], f"Graph is not bidirectional: {node} -> {neighbour}"
        assert set(self.graph.keys()) == set(self.all_nodes)

        self.start_locations = [0, 4, 16, 22, 30]
        self.target_destinations = [5, 9, 2]
        self.food_menus = {0:["Nasi Lemak","Roti Canai","Teh Tarik"],4:["Chicken Rice","Wantan Mee","Mixed Rice"],16:["Club Sandwich","Burger & Fries","Iced Lemon Tea"],22:["Japanese Bento","Onigiri","Oden"],30:["Western Chicken Chop","Fried Kuey Teow","Iced Coffee"]}
        self.uncovered_paths = {frozenset({1,10}),frozenset({10,11}),frozenset({27,5}),frozenset({5,12}),frozenset({12,14}),frozenset({22,25}),frozenset({25,26}),frozenset({7,29}),frozenset({29,16}),frozenset({30,35}),frozenset({35,9}),frozenset({35,8})}
        self.crowd_level = {7:1.5,6:1.5,3:1.5,26:1.5,14:1.5,4:1.8}
        self.pedestrian_zones = {0,1,3,4,5,6,7,14,16,20,21,24,26,27,28,30,31,32}
        self.car_zones = {3,4,5,8,10,12,21,23,25,26,27,28,29,33,34,35}
        self.pedestrian_routes = [[3, 4]]
        self.car_routes = [[4, 27, 5, 28]]
        self.pedestrian_positions = [3, 6]
        self.car_positions = [4, 12]
        self.pedestrian_route_indices = [0]
        self.car_route_indices = [0]
        self.pedestrian_prev = [None, None]
        self.car_prev = [None, None]

        self.longer_paths = {frozenset({23,24}):-3,frozenset({24,27}):-2,frozenset({1,10}):-2,frozenset({7,8}):-2,frozenset({14,15}):-2,frozenset({30,35}):-2}
        self.action_space = spaces.Discrete(37)
        self.observation_mode = "tabular"
        self.observation_space = spaces.Tuple((spaces.Discrete(36),spaces.Discrete(36),spaces.Discrete(2),spaces.Discrete(3)))
        self.current_location = None
        self.target_destination = None
        self.current_weather = None
        self.current_traffic = None
        self.selected_food = None

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _choice(self, seq):
        return seq[int(self.np_random.integers(len(seq)))]

    def valid_actions(self, location=None):
        loc = self.current_location if location is None else location
        if loc is None or loc not in self.graph:
            raise ValueError("Invalid location (did you call reset()?).")
        actions = list(self.graph[loc])
        if self.allow_wait:
            actions.append(36)
        return actions

    def obstacle_prob(self, node):
        base = 0.25 if self.current_weather == 0 else 0.35
        return min(0.6, base * self.crowd_level.get(node, 1.0))

    def edge_cost(self, u, v):
        edge = frozenset({u, v})
        cost = -self.longer_paths.get(edge, -1)
        if self.current_weather == 1 and edge in self.uncovered_paths:
            cost += 15
        if v != self.target_destination and (v in self.pedestrian_zones or v in self.car_zones):
            p = self.obstacle_prob(v)
            cost += 2 * p / (1 - p)
        return cost

    def dijkstra_reference_action(self, safety_margin=1):
        """Map-based benchmark (NOT an RL agent, and not truly optimal).
        Uses Dijkstra costs, then picks the cheapest neighbour that is not
        blocked. A neighbour is blocked if someone stands on it or, with
        safety_margin=1, next to it."""
        if self.current_location is None:
            raise RuntimeError("Call reset() first.")
        dist = {node: float("inf") for node in self.all_nodes}
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
        occupied = self.occupied_nodes()
        danger = set(occupied)
        if safety_margin >= 1:
            for n in occupied:
                danger.update(self.graph[n])

        ranked = sorted(self.graph[loc], key=lambda n: self.edge_cost(loc, n) + dist[n])
        for n in ranked:
            if (n not in danger) or (n == self.target_destination and n not in occupied):
                return n
        for n in ranked:
            if n not in occupied:
                return n
        return 36 if self.allow_wait else ranked[0]

    optimal_action = dijkstra_reference_action  # old name kept as an alias

    # ------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------

    def set_observation_mode(self, mode):
        if mode not in ["tabular", "full"]:
            raise ValueError("Mode must be 'tabular' or 'full'.")
        self.observation_mode = mode
        if mode == "tabular":
            self.observation_space = spaces.Tuple((spaces.Discrete(36),spaces.Discrete(36),spaces.Discrete(2),spaces.Discrete(3)))
        else:
            self.observation_space = spaces.Tuple((spaces.Discrete(36),spaces.Discrete(36),spaces.Discrete(2),spaces.Discrete(36),spaces.Discrete(36),spaces.Discrete(36),spaces.Discrete(36)))

    def _get_tabular_obs(self):
        nearby_nodes = [self.current_location] + self.graph[self.current_location]
        pedestrian_nearby = any(pos in nearby_nodes for pos in self.pedestrian_positions)
        car_nearby = any(pos in nearby_nodes for pos in self.car_positions)
        if car_nearby:
            traffic = 2
        elif pedestrian_nearby:
            traffic = 1
        else:
            traffic = 0
        return (int(self.current_location), int(self.target_destination), int(self.current_weather), traffic)

    def _get_full_obs(self):
        return (int(self.current_location), int(self.target_destination), int(self.current_weather),
                int(self.pedestrian_positions[0]), int(self.pedestrian_positions[1]),
                int(self.car_positions[0]), int(self.car_positions[1]))

    def _get_obs(self):
        if self.observation_mode == "tabular":
            return self._get_tabular_obs()
        return self._get_full_obs()

    def _get_info(self, outcome="reset", rain_violation=False):
        traffic_map = {0: "Clear", 1: "Pedestrian Detected", 2: "Car Detected"}
        return {"location_str": self.node_names[self.current_location],
                "target_str": self.node_names[self.target_destination],
                "weather_str": ("Sunny" if self.current_weather == 0 else "Rainy"),
                "traffic_str": traffic_map[self.current_traffic],
                "food_item": self.selected_food,
                "outcome": outcome,
                "rain_violation": rain_violation,
                "steps": self.steps}

    # ------------------------------------------------------------
    # Obstacle helpers
    # ------------------------------------------------------------

    def _hops_from(self, start):
        hops = {start: 0}
        queue = [start]
        for u in queue:
            for v in self.graph[u]:
                if v not in hops:
                    hops[v] = hops[u] + 1
                    queue.append(v)
        return hops

    def _zone_component_size(self, zone):
        size, seen = {}, set()
        for n in zone:
            if n in seen:
                continue
            comp, queue = {n}, [n]
            for u in queue:
                for v in self.graph[u]:
                    if v in zone and v not in comp:
                        comp.add(v)
                        queue.append(v)
            seen |= comp
            for c in comp:
                size[c] = len(comp)
        return size

    def _spawn_walkers(self, zone):
        """Two random start nodes inside `zone`, at least 2 hops from the robot
        (so they never start on the robot) and inside a zone piece with >= 3
        connected nodes. Always returns exactly 2 plain ints."""
        hops = self._hops_from(self.current_location)
        sizes = self._zone_component_size(zone)
        eligible = sorted(n for n in zone if hops.get(n, 99) >= 2 and sizes[n] >= 3)
        if len(eligible) < 2:  # fallback: relax the movement requirement
            eligible = sorted(n for n in zone if hops.get(n, 99) >= 2)
        if len(eligible) < 2:  # last resort: anything except the robot's node
            eligible = sorted(n for n in zone if n != self.current_location)
        picks = self.np_random.choice(eligible, size=2, replace=False)
        return [int(n) for n in picks]

    def _walker_step(self, pos, prev, zone):
        """One random-walk step inside `zone`. Always uses exactly two random
        draws, so every agent sees identical obstacle movement per seed."""
        options = [n for n in self.graph[pos] if n in zone]
        pause = self.np_random.random() < self.stay_prob
        u = self.np_random.random()
        if not options:
            return pos, prev
        forward = [n for n in options if n != prev] or options
        weights = np.array([self.crowd_level.get(n, 1.0) for n in forward], dtype=float)
        cum = np.cumsum(weights / weights.sum())
        nxt = forward[min(int(np.searchsorted(cum, u, side="right")), len(forward) - 1)]
        return (pos, prev) if pause else (nxt, pos)

    def _move_walkers(self):
        for i in range(2):
            self.pedestrian_positions[i], self.pedestrian_prev[i] = self._walker_step(
                self.pedestrian_positions[i], self.pedestrian_prev[i], self.pedestrian_zones)
            self.car_positions[i], self.car_prev[i] = self._walker_step(
                self.car_positions[i], self.car_prev[i], self.car_zones)

    def _move_fixed_obstacles(self):
        pedestrian_route = self.pedestrian_routes[0]
        self.pedestrian_route_indices[0] = (self.pedestrian_route_indices[0] + 1) % len(pedestrian_route)
        self.pedestrian_positions[0] = pedestrian_route[self.pedestrian_route_indices[0]]
        car_route = self.car_routes[0]
        self.car_route_indices[0] = (self.car_route_indices[0] + 1) % len(car_route)
        self.car_positions[0] = car_route[self.car_route_indices[0]]

    def _move_random_obstacles(self):
        self.pedestrian_positions[1], self.pedestrian_prev[1] = self._walker_step(
            self.pedestrian_positions[1], self.pedestrian_prev[1], self.pedestrian_zones)
        self.car_positions[1], self.car_prev[1] = self._walker_step(
            self.car_positions[1], self.car_prev[1], self.car_zones)

    def _move_obstacles(self):
        if self.obstacle_mode == "random":
            self._move_walkers()
        else:
            self._move_fixed_obstacles()
            self._move_random_obstacles()

    def occupied_nodes(self):
        return set(self.pedestrian_positions) | set(self.car_positions)

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Optional customer order: {'start', 'target', 'food', 'weather'}."""
        super().reset(seed=seed)

        options = {} if options is None else options
        if not isinstance(options, dict):
            raise TypeError("Options must be a dictionary.")
        unknown = set(options) - {"start", "target", "food", "weather"}
        if unknown:
            raise ValueError(f"Unknown reset options: {unknown}")
        if "start" in options and options["start"] not in self.start_locations:
            raise ValueError("Invalid starting restaurant.")
        if "target" in options and options["target"] not in self.target_destinations:
            raise ValueError("Invalid target destination.")
        if "weather" in options and options["weather"] not in (0, 1):
            raise ValueError("Weather must be 0 (Sunny) or 1 (Rainy).")

        self.steps = 0
        self._episode_over = False

        # Destination
        self.target_destination = (options["target"] if "target" in options
                                   else self._choice(self.target_destinations))

        # Starting restaurant
        if "start" in options:
            self.current_location = options["start"]
        elif "food" in options:
            candidates = [r for r in self.start_locations if options["food"] in self.food_menus[r]]
            if not candidates:
                raise ValueError("No restaurant serves that food item.")
            self.current_location = self._choice(candidates)
        else:
            self.current_location = self._choice(self.start_locations)

        # Weather and food
        self.current_weather = (options["weather"] if "weather" in options
                                else self._choice([0, 1]))
        if "food" in options:
            self.selected_food = options["food"]
            if self.selected_food not in self.food_menus[self.current_location]:
                raise ValueError("Food item is not available at the selected restaurant.")
        else:
            self.selected_food = self._choice(self.food_menus[self.current_location])

        self.pedestrian_prev = [None, None]
        self.car_prev = [None, None]

        # Obstacles: always exactly 2 pedestrians and 2 cars
        if self.obstacle_mode == "random":
            self.pedestrian_positions = self._spawn_walkers(self.pedestrian_zones)
            self.car_positions = self._spawn_walkers(self.car_zones)
        else:
            pedestrian_route = self.pedestrian_routes[0]
            car_route = self.car_routes[0]
            ped_index = next((i for i, n in enumerate(pedestrian_route)
                              if n != self.current_location), 0)
            car_index = next((i for i, n in enumerate(car_route)
                              if n != self.current_location), 0)
            self.pedestrian_route_indices = [ped_index]
            self.car_route_indices = [car_index]
            self.pedestrian_positions = [pedestrian_route[ped_index], 6]
            self.car_positions = [car_route[car_index], 12]

        self.current_traffic = self._get_tabular_obs()[3]
        return self._get_obs(), self._get_info("reset")

    # ------------------------------------------------------------
    # Step
    # ------------------------------------------------------------

    def step(self, action):
        if self.current_location is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_over:
            raise RuntimeError("Episode has ended. Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        action = int(action)
        reward = 0
        terminated = False
        truncated = False
        outcome = "move"
        rain_violation = False
        blocked_node = None
        retreated = False
        contact = None  # obstacle standing on the node the robot moves into

        self.steps += 1

        old_pedestrian_positions = self.pedestrian_positions.copy()
        old_car_positions = self.car_positions.copy()
        previous_location = self.current_location

        # Robot action
        if action == 36 and self.allow_wait:
            reward = -2
            outcome = "wait"

        elif action not in self.graph[self.current_location]:
            reward = -5
            outcome = "invalid"

        elif self.contact_mode == "retreat" and action in self.occupied_nodes():
            reward = self.retreat_penalty
            outcome = "retreat"
            retreated = True
            blocked_node = action

        else:
            # Record contact BEFORE obstacles move, so walking into someone
            # cannot be avoided by them stepping away in the same tick.
            if action in self.pedestrian_positions:
                contact = "ped"
            elif action in self.car_positions:
                contact = "car"

            path_edge = frozenset({self.current_location, action})
            reward += self.longer_paths.get(path_edge, -1)
            if self.current_weather == 1 and path_edge in self.uncovered_paths:
                reward -= 15
                rain_violation = True
            self.current_location = action

        # Obstacles move
        self._move_obstacles()

        # Node collisions
        if self.current_location in self.pedestrian_positions or contact == "ped":
            reward -= 25
            terminated = True
            outcome = "ped_collision"
        elif self.current_location in self.car_positions or contact == "car":
            reward -= 50
            terminated = True
            outcome = "car_collision"

        # Edge-swap collisions
        if not terminated and self.current_location != previous_location:
            for old_pos, new_pos in zip(old_pedestrian_positions, self.pedestrian_positions):
                if previous_location == new_pos and self.current_location == old_pos:
                    reward -= 25
                    terminated = True
                    outcome = "ped_collision"
                    break
            if not terminated:
                for old_pos, new_pos in zip(old_car_positions, self.car_positions):
                    if previous_location == new_pos and self.current_location == old_pos:
                        reward -= 50
                        terminated = True
                        outcome = "car_collision"
                        break

        # Delivery success
        if not terminated and self.current_location == self.target_destination:
            reward += 100
            terminated = True
            outcome = "goal"

        self.current_traffic = self._get_tabular_obs()[3]

        if not terminated and self.steps >= self.max_steps:
            truncated = True

        self._episode_over = terminated or truncated

        info = self._get_info(outcome, rain_violation)
        info["blocked_node"] = blocked_node
        info["retreated"] = retreated
        info["previous_location"] = previous_location
        info["current_location"] = self.current_location
        info["robot_moved"] = self.current_location != previous_location

        return self._get_obs(), reward, terminated, truncated, info


# ============================================================
# EVALUATION
# ============================================================

TEST_SEED = 1000
NUM_TEST_EPISODES = 1000


def evaluate_agent(env, table=None, num_episodes=NUM_TEST_EPISODES, base_seed=TEST_SEED):
    """Evaluate a Q-table (shape (36, 36, 2, 3, 37)) or, if table=None, a random
    valid-action baseline. The same seeds give every agent the same orders,
    weather and obstacle movement."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be greater than zero.")
    if table is not None:
        env.set_observation_mode("tabular")
        table = np.asarray(table)
        if table.shape != (36, 36, 2, 3, 37):
            raise ValueError(f"Q-table must have shape (36, 36, 2, 3, 37), got {table.shape}.")

    rng = random.Random(base_seed)
    success = ped_crashes = car_crashes = timeouts = 0
    rainy_episodes = rain_compliant = retreats = 0
    total_rewards, steps_for_success = [], []

    for ep in range(num_episodes):
        state, info = env.reset(seed=base_seed + ep)
        terminated = truncated = False
        ep_reward = ep_steps = 0
        rain_violation = False
        is_rainy = env.current_weather == 1
        rainy_episodes += is_rainy

        while not (terminated or truncated):
            valid = env.valid_actions(state[0])
            if table is not None:
                values = table[state][valid]
                best = np.flatnonzero(values == values.max())
                action = valid[int(rng.choice(best))]
            else:
                action = rng.choice(valid)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            rain_violation |= info["rain_violation"]
            retreats += int(info["retreated"])

        if info["outcome"] == "goal":
            success += 1
            steps_for_success.append(ep_steps)
        elif info["outcome"] == "ped_collision":
            ped_crashes += 1
        elif info["outcome"] == "car_collision":
            car_crashes += 1
        elif truncated:
            timeouts += 1
        if is_rainy and not rain_violation:
            rain_compliant += 1
        total_rewards.append(ep_reward)

    return {
        "Success Rate (%)": success / num_episodes * 100,
        "Collision Rate (%)": (ped_crashes + car_crashes) / num_episodes * 100,
        "Pedestrian Crashes": ped_crashes,
        "Car Crashes": car_crashes,
        "Timeouts": timeouts,
        "Retreats per Episode": retreats / num_episodes,
        "Rain Roof Compliance (%)": (rain_compliant / rainy_episodes * 100) if rainy_episodes else None,
        "Avg Reward": float(np.mean(total_rewards)),
        "Avg Delivery Steps": float(np.mean(steps_for_success)) if steps_for_success else None,
    }


if __name__ == "__main__":
    env = CampusDeliveryEnv()
    state, info = env.reset(seed=42)
    print("Starting Restaurant:", info["location_str"])
    print("Destination:", info["target_str"])
    print("Selected Food:", info["food_item"])
    print("Weather:", info["weather_str"])
    print("Traffic:", info["traffic_str"])
    print("Initial State:", state)
    print("Valid Actions:", env.valid_actions())
    print("Obstacles (pedestrians, cars):", env.pedestrian_positions, env.car_positions)
