import persistent_env_info as pinfo
import numpy as np

# This tracks time-dependent information about the environment for determining more complex rewards

def init(H, W):
    global prev_agent_pos  # Grid index of previous agent position
    global prev_agent_positions_heatmap  # Dictionary mapping grid indices to visit frequency
    global path_of_least_resistance  # Grid index of the immediate cell of least resistance
    global prev_path_of_least_resistance  # Grid index of the previous cell of least resistance
    global enemy_positions  # List of (y, x) tuples of enemy positions
    global enemy_orientations  # Dictionary mapping (y, x) enemy position tuples to orientations of enemy positions (0-3 for left, down, right, up)
    global unknown_enemies  # Set of (y, x) enemy position tuples for which orientation has yet to be determined
    global unknown_enemy_possible_orientations # Dict mapping unknown enemies to a set of their possible orientations.
    global enemy_wall_map  # Simplified grid indicating enemy and wall locations as 2's and 3's respectively to save on repeated computation
    global possible_dangerous_cells  # Set of (y, x) tuples indicating potentially dangerous cells in the next time step

    prev_agent_pos = None

    prev_agent_positions_heatmap = {i: 0 for i in range(W * H)}
    path_of_least_resistance = None
    prev_path_of_least_resistance = None
    enemy_positions = []
    enemy_orientations = dict()
    unknown_enemies = set()
    unknown_enemy_possible_orientations = dict()
    enemy_wall_map = None
    possible_dangerous_cells = set()