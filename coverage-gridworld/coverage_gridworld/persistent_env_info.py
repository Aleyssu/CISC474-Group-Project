from collections import deque

# This tracks time-dependent information about the environment for determining more complex rewards and better observations

def init(H, W):
    global prev_agent_pos  # Grid index of previous agent position
    global path_of_least_resistance  # Grid index of the immediate cell of least resistance
    global prev_path_of_least_resistance  # Grid index of the previous cell of least resistance

    global enemy_positions  # List of (y, x) tuples of enemy positions
    global enemy_orientations  # Dictionary mapping (y, x) enemy position tuples to orientations of enemy positions (0-3 for left, down, right, up)
    global unknown_enemies  # Set of (y, x) enemy position tuples for which orientation has yet to be determined
    global unknown_enemy_possible_orientations # Dict mapping unknown enemies to a set of their possible orientations.
    global enemy_wall_map  # Simplified grid indicating enemy and wall locations as 2's and 3's respectively to save on repeated computation
    global enemy_wall_danger_map  # Dictionary mapping orientation indices (0-3) to a simplified grid indicating projected danger cells, enemy and wall locations as 1, 2 and 3 respectively to save on repeated computation
    global enemy_orientations_computed  # Indicates whether enemy_wall_danger_map has been fully computed
    global env_orientation  # Determines the current rotation of the environment

    global possible_dangerous_cells  # Set of (y, x) tuples indicating potentially dangerous cells in the next time step
    global trap_cells_prev  # Set of (y, x) tuples cells which will result in certain death if traveled to in the next time step
    global trap_cells  
    global bfs_cell_coords  # Set of (y, x) tuples indicating the cells coords for the moves leading towards the nearest empty cell
    global following_bfs  # Indicates whether the agent is following the current bfs policy
    global just_started_following_bfs  # Indicates whether the agent has just started following a search policy
    global push_traps  # Determines whether the agent will force its way into trap cells as a last resort
    global push_danger  # Determines whether the agent will try to kill itself in impossible situations
    global is_dying  # If true, floods the agent's senses with pheromones as they die so that they don't learn to associate anything else with reward

    prev_agent_pos = None

    path_of_least_resistance = None
    prev_path_of_least_resistance = None
    enemy_positions = []
    enemy_orientations = dict()
    unknown_enemies = set()
    unknown_enemy_possible_orientations = dict()
    enemy_wall_map = None
    possible_dangerous_cells = set()
    trap_cells = set()
    trap_cells_prev = set()
    bfs_cell_coords = deque()
    following_bfs = False
    just_started_following_bfs = False
    enemy_wall_danger_map = dict()
    enemy_orientations_computed = False
    env_orientation = 0
    push_traps = False
    push_danger = False
    is_dying = False