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

    def __init__(self):

        # Your latest __init__() code goes here.
        # Include:
        # - 36 campus nodes
        # - node_names
        # - node_coords
        # - graph
        # - restaurants and destinations
        # - food menus
        # - uncovered paths
        # - pedestrian and car zones
        # - moving obstacle routes and positions
        # - action_space and observation_space

        pass

    # ======================================
    # OBSERVATION FUNCTIONS
    # ======================================

    # Copy your latest:
    # set_observation_mode()
    # _get_tabular_obs()
    # _get_full_obs()
    # _get_obs()
    # _get_info()

    # ======================================
    # OBSTACLE FUNCTIONS
    # ======================================

    # Copy your latest:
    # _move_fixed_obstacles()
    # _move_random_obstacles()
    # Any other obstacle-related functions

    # ======================================
    # ENVIRONMENT FUNCTIONS
    # ======================================

    # Copy your latest:
    # reset()
    # step()
    # Other helper functions used by reset() and step()
