import numpy as np
import gymnasium as gym
import persistent_env_info as pinfo

# action IDs
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4

DIRECTIONS = [(0, -1), (-1, 0), (1, 0), (0, 1)]  # left, up, down, right

# Not to be confused with action IDs
# Can be incremented with mod operations to 
# keep track of enemy orientations
DIRECTION_MAP = {
    0: (-1, 0),  # Up
    1: (0, -1),  # Left
    2: (1, 0),   # Down
    3: (0, 1)    # Right
}
INV_DIRECTION_MAP = {
    (-1, 0): 0,  # Up
    (0, -1): 1,  # Left
    (1, 0): 2,   # Down
    (0, 1): 3    # Right
}

# rendering colors
BLACK = (0, 0, 0)            # unexplored cell
WHITE = (255, 255, 255)      # explored cell
BROWN = (101, 67, 33)        # wall
GREY = (160, 161, 161)       # agent
GREEN = (31, 198, 0)         # enemy
RED = (255, 0, 0)            # unexplored cell being observed by an enemy
LIGHT_RED = (255, 127, 127)  # explored cell being observed by an enemy

# These are not rendered, but used in the observation space
OUT_BOUNDS = "ob"                 # out of bounds
PATH_OF_LEAST_RESISTANCE = "plr"  # cell that is part of the path of least resistance to cover more cells
POSSIBLE_DANGER = "pd"            # cell that will possibly be observed by an enemy in the next time step 
DANGER = "d"                      # cell that will be observed by an enemy in the next time step

# rgb mapping to integer values for the observation space
COLOR_MAP = {
    BLACK: 0,      # unexplored cell
    WHITE: 1,      # explored cell
    BROWN: 2,      # wall
    GREY: 1,       # agent
    GREEN: 2,      # enemy
    RED: 0,        # unexplored cell being observed by an enemy
    LIGHT_RED: 1,  # explored cell being observed by an enemy
    OUT_BOUNDS: 2,                # out of bounds
    PATH_OF_LEAST_RESISTANCE: 3,  # cell that is part of the path of least resistance to cover more cells (not rendered, but can be used in the observation space)
    POSSIBLE_DANGER: 5,            # cell that will possibly be observed by an enemy in the next time step
    DANGER: 4
}

CELL_FEATURES = 6  # Number of features to encode for in any given cell

# Observation with Semi-OHE encoding for the 4 adjacent cells and the current cell (5 total)
def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    observation_space = gym.spaces.Box(
        low=0,
        high=1,
        shape=(5, CELL_FEATURES),
        dtype=np.uint8
    )
    return observation_space


def possibly_from_other_enemy(danger_grid: np.ndarray, danger_pos: tuple, origin_direction: tuple) -> int:
    """
    Determines if the given danger cell is possibly from an enemy other than the one in the origin direction

    Returns 1 if said direction is perpendicular to the original enemy's sightline, 2 if it is in front of
    the original enemy, and 0 otherwise.

    The return '2' scenario is important so that we know there's no point in scanning that sightline further
    as no more information can be derived from that direction. 
    """
    H, W = danger_grid.shape

    for dir in DIRECTIONS:
        if dir != origin_direction:
            for i in range(1, 5):
                proj_pos = (i * dir[0] + danger_pos[0], i * dir[1] + danger_pos[1])
                # Stop checking if we reach out of bounds
                if not (0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W):
                    break

                # Keep checking forward if there are more danger cells
                if danger_grid[proj_pos] == 1:
                    continue

                # If we reach another enemy, check if it's possible that they're looking this way
                elif danger_grid[proj_pos] == 2:
                    # Get the possible orientations of the enemy
                    if proj_pos not in pinfo.unknown_enemies:
                        orientations = {pinfo.enemy_orientations[proj_pos]}
                    else:
                        orientations = pinfo.unknown_enemy_possible_orientations[proj_pos]

                    # Check if any of the orientations face this way
                    for ori in orientations:
                        if DIRECTION_MAP[ori] == (dir[0] * -1, dir[1] * -1):
                            # Check if the other enemy is facing the original enemy
                            if dir == (origin_direction[0] * -1, origin_direction[1] * -1):
                                return 2
                            else:
                                return 1
                # Stop checking if there are no more danger cells ahead
                break
    return 0


# Note that enemies always rotate counterclockwise and project danger cells 4 cells out which are blocked by walls and other enemies.
def update_predicted_enemy_orientations(danger_grid: np.ndarray) -> None:
    """
    Given a integer matrix representing the game grid with the following mappings: 
    0: empty space
    1: danger cells
    2: enemy cells 
    3: walls
    Tries to predict where the enemies are oriented based on where the danger cells are being projected.

    Updates the persistent_env_info on the possible orientations for unknown enemies.
    """
    
    H, W = danger_grid.shape
    deduced_enemies = []
    for enemy_pos in pinfo.unknown_enemies.copy():
        deduced_enemy = False
        eliminated_dirs = []
        
        for dir in pinfo.unknown_enemy_possible_orientations[enemy_pos]:
            if deduced_enemy:
                break
            dir_yx = DIRECTION_MAP[dir]
            for i in range(1, 5):
                proj_pos = (i * dir_yx[0] + enemy_pos[0], i * dir_yx[1] + enemy_pos[1])
                # Check for projected sightline being inbounds
                if 0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W:
                    if danger_grid[proj_pos] == 1:
                        from_other_enemy = possibly_from_other_enemy(danger_grid, proj_pos, (dir_yx[0] * -1, dir_yx[1] * -1))
                        # The other enemy is possibly facing directly into our enemy - no more useful information can be gathered in this direction so stop searching
                        if from_other_enemy == 2:
                            break
                        elif from_other_enemy == 1:
                            continue
                        # The danger cell can only possibly be from our enemy - update the enemy's orientation as discovered
                        else:
                            pinfo.unknown_enemies.remove(enemy_pos)
                            deduced_enemies.append(enemy_pos)
                            pinfo.enemy_orientations[enemy_pos] = dir
                            deduced_enemy = True
                            break
                    # Wall or other enemy in the way - stop searching
                    elif danger_grid[proj_pos] == 2 or danger_grid[proj_pos] == 3:
                        break
                    # Empty space found so the enemy can't be facing this direction - eliminate it from the pool of possible directions and move on
                    else:
                        eliminated_dirs.append(dir)
                        break
        
        if not deduced_enemy:
            for dir in eliminated_dirs:
                pinfo.unknown_enemy_possible_orientations[enemy_pos].remove(dir)

            if len(pinfo.unknown_enemy_possible_orientations[enemy_pos]) == 1:
                pinfo.unknown_enemies.remove(enemy_pos)
                pinfo.enemy_orientations[enemy_pos] = pinfo.unknown_enemy_possible_orientations[enemy_pos].pop()
    
    # print("Unknowns: %s, Known Orientations: %s, Unknown Possible Orientations: %s" % (pinfo.unknown_enemies, pinfo.enemy_orientations, pinfo.unknown_enemy_possible_orientations))


def project_enemy_view(enemy_wall_map: np.ndarray, enemy_pos: tuple, dir: tuple):
    """
    Returns a list of (y, x) tuples indicating the cell coordinates which can be seen by the enemy
    if they're facing the given direction.
    """
    H, W = enemy_wall_map.shape
    viewed_cells = []
    for i in range(1, 5):
        proj_pos = (i * dir[0] + enemy_pos[0], i * dir[1] + enemy_pos[1])
        # Check for projected sightline being inbounds
        if 0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W and enemy_wall_map[proj_pos] == 0:
            viewed_cells.append(proj_pos)
        else:
            break

    return viewed_cells


def is_trap(enemy_wall_map: np.ndarray, pos: tuple, time_steps: int) -> bool:
    """
    Returns true if being in the given (y, x) position is guaranteed to get the agent caught
    at the given time step in the future, else false.
    """
    H, W = enemy_wall_map.shape

    # Compute predicted enemy orientations and projected danger cells in the given time step
    danger_cells = set()
    for enemy in pinfo.enemy_positions:
        if enemy not in pinfo.unknown_enemies:
            projected_ori = DIRECTION_MAP[(pinfo.enemy_orientations[enemy] + time_steps) % 4]
            for cell in project_enemy_view(enemy_wall_map, enemy, projected_ori):
                danger_cells.add(cell)
    
    # Check possible movement directions for a safe tile
    for dir in [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]:  # left, up, down, right, current position
        proj_pos = (pos[0] + dir[0], pos[1] + dir[1])
        # Position is a wall, enemy, or out of bounds (No movement)
        if not (0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W) or enemy_wall_map[proj_pos] in (2, 3):
            proj_pos = pos
        if proj_pos not in danger_cells:
            return False
    
    return True


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    H, W, _ = grid.shape

    # Reset persistent info if all cells are uncleared (new episode)
    cleared_cells = np.zeros((H, W), dtype=np.uint8)
    mask = np.all(grid == WHITE, axis=-1)
    cleared_cells[mask] = 1
    if np.sum(cleared_cells) == 0:
        # print("Clearing persistent info for new episode")
        pinfo.init(H, W)  

        # Get enemy positions and orientations
        enemy_mask = np.all(grid == GREEN, axis=-1)
        wall_mask = np.all(grid == BROWN, axis=-1)
        y, x  = np.where(enemy_mask)
        pinfo.enemy_positions = set(zip(y, x))
        pinfo.unknown_enemies = set(pinfo.enemy_positions)
        pinfo.unknown_enemy_possible_orientations = {enemy: {0, 1, 2, 3} for enemy in pinfo.enemy_positions}

        danger_mask = np.logical_or(np.all(grid == RED, axis=-1), np.all(grid == LIGHT_RED, axis=-1))
        pinfo.enemy_wall_map = 2 * enemy_mask + 3 * wall_mask 
        danger_grid = danger_mask + pinfo.enemy_wall_map  # Create a grid where 0 is safe, 1 is danger, 2 is enemy, and 3 is wall

        update_predicted_enemy_orientations(danger_grid)
    # Try to deduce unknown enemy orientations if there's any left
    elif len(pinfo.unknown_enemies) > 0:
        danger_mask = np.logical_or(np.all(grid == RED, axis=-1), np.all(grid == LIGHT_RED, axis=-1))
        danger_grid = danger_mask + pinfo.enemy_wall_map
        update_predicted_enemy_orientations(danger_grid)

    # Update enemy orientations for the next time step
    for enemy in pinfo.enemy_positions:
        if enemy not in pinfo.unknown_enemies:
            pinfo.enemy_orientations[enemy] = (pinfo.enemy_orientations[enemy] + 1) % 4
        else:
            possible_ori = set()
            for ori in pinfo.unknown_enemy_possible_orientations[enemy]:
                possible_ori.add((ori + 1) % 4)
            pinfo.unknown_enemy_possible_orientations[enemy] = possible_ori

    # Predict where the danger cells will be in the next time step
    possible_danger_cells = set()  # Contains (y, x) tuples for cells that might be dangerous in the next time step
    pinfo.possible_dangerous_cells = set()
    danger_cells = set()
    for enemy in pinfo.enemy_positions:
        # Cells that may be seen by enemies with yet to be determined orientations
        if enemy in pinfo.unknown_enemies:
            for dir in pinfo.unknown_enemy_possible_orientations[enemy]:
                for cell in project_enemy_view(pinfo.enemy_wall_map, enemy, DIRECTION_MAP[dir]):
                    possible_danger_cells.add(cell)
                    pinfo.possible_dangerous_cells.add(cell[0] * W + cell[1])
        # Cells that will certainly be seen by enemies
        else:
            for cell in project_enemy_view(pinfo.enemy_wall_map, enemy, DIRECTION_MAP[pinfo.enemy_orientations[enemy]]):
                danger_cells.add(cell)

    # For debugging
    # print_map = pinfo.enemy_wall_map.copy()
    # for cell in possible_danger_cells:
    #     print_map[cell] = 5
    # for cell in danger_cells:
    #     print_map[cell] = 4
    # print("Map:\n%s\nDanger: %s\nCaution: %s" % (print_map, danger_cells, possible_danger_cells))

    # Initialize the observation with zeros
    ohe = np.zeros((5, CELL_FEATURES), dtype=np.uint8)
    agent_pos = np.where(np.all(grid == GREY, axis=-1))
    # Return zeros if the agent position is not found (game over case)
    if(len(agent_pos[0]) == 0):
        return ohe
    agent_pos = (agent_pos[0][0], agent_pos[1][0])

    pinfo.trap_cells_prev = pinfo.trap_cells
    pinfo.trap_cells = set()
    # Update observation for adjacent cells and current cell with semi-OHE encoding
    observation_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # left, down, right, up, stay
    black_adjacent = False  # Indicates there's a safe unexplored tile in the immediate vicinity
    adjacent_explored_cells = []
    curr_cell_danger = 0  # Indicates if the agent's current cell will be dangerous in the next time step
    for i in range(4, -1, -1):
        action_offset = observation_offsets[i]

        # Check if adjacent cell is within bounds and get its color otherwise mark it as out of bounds
        pos = (agent_pos[0] + action_offset[0], agent_pos[1] + action_offset[1])
        pos_to_int = (pos[0] * W + pos[1])
        if 0 <= pos[0] < H and 0 <= pos[1] < W:
            if pos in danger_cells:
                color = COLOR_MAP[DANGER]
                ohe[i, color] = 1
                if i == 4: curr_cell_danger = 1
            elif is_trap(pinfo.enemy_wall_map, pos, 1):
                color = COLOR_MAP[DANGER]
                ohe[i, color] = 1
                pinfo.trap_cells.add(pos_to_int)
                if i == 4: curr_cell_danger = 1
            elif pos in possible_danger_cells:
                color = COLOR_MAP[POSSIBLE_DANGER]
                ohe[i, color] = 1
                if i == 4: curr_cell_danger = 2
            else:
                color = COLOR_MAP[tuple(grid[pos])]
                # Update explored cells with a semi-OHE encoding which indicates visit frequency
                if color == COLOR_MAP[WHITE]:
                    ohe[i, color] = 1 + pinfo.prev_agent_positions_heatmap[pos_to_int]
                    adjacent_explored_cells.append(i)
                elif color == COLOR_MAP[BLACK]:
                    black_adjacent = True
                    ohe[i, color] = 1
                # Treat walls and other obstacles as dangerous if the agent's current cell will be dangerous in the next time step
                elif color == COLOR_MAP[BROWN]:
                    if curr_cell_danger == 2:
                        color = COLOR_MAP[POSSIBLE_DANGER]
                    elif curr_cell_danger == 1:
                        color = COLOR_MAP[DANGER]
                    ohe[i, color] = 1
                else:
                    ohe[i, color] = 1

        else:
            color = COLOR_MAP[OUT_BOUNDS]
            ohe[i, color] = 1
    
    # Finding the cell of least resistance when there are no immediate unexplored tiles
    if not black_adjacent and len(adjacent_explored_cells) > 0:
        path_of_least_resistance = 0
        cell_of_least_resistance = agent_pos[0] * W + agent_pos[1]
        min_path = float('inf')

        for cell in adjacent_explored_cells:
            if ohe[cell, COLOR_MAP[WHITE]] < min_path:
                min_path = ohe[cell, COLOR_MAP[WHITE]]
                path_of_least_resistance = cell
                cell_yx = observation_offsets[cell]
                cell_of_least_resistance = (cell_yx[0] + agent_pos[0]) * W + cell_yx[1] + agent_pos[1]

        # Select the cell of least resistance
        ohe[path_of_least_resistance, COLOR_MAP[PATH_OF_LEAST_RESISTANCE]] = 1
        ohe[path_of_least_resistance, COLOR_MAP[WHITE]] = 0
        pinfo.prev_path_of_least_resistance = pinfo.path_of_least_resistance
        pinfo.path_of_least_resistance = cell_of_least_resistance


    # print(ohe)
    return ohe


# Global observation with OHE encoding for each type of tile

# def observation_space(env: gym.Env) -> gym.spaces.Space:
#     """
#     Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
#     """
#     observation_space = gym.spaces.Box(
#         low=0,
#         high=1,
#         shape=(env.grid.shape[0], env.grid.shape[1], len(COLOR_MAP)),
#         dtype=np.uint8
#     )

#     return observation_space


# def observation(grid: np.ndarray):
#     """
#     Function that returns the observation for the current state of the environment.
#     """
#     H, W, _ = grid.shape
#     ohe = np.zeros((H, W, len(COLOR_MAP)), dtype=np.uint8)

#     for color, idx in COLOR_MAP.items():
#         mask = np.all(grid == color, axis=-1)
#         ohe[mask, idx] = 1

#     return ohe


def reward(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using

    reward = 0

    if new_cell_covered:
        reward += 1
    elif pinfo.prev_agent_pos == agent_pos and not agent_pos == pinfo.prev_path_of_least_resistance:
        reward -= 0.1
    
    if agent_pos in pinfo.prev_agent_positions_heatmap.keys():
        if not agent_pos == pinfo.prev_path_of_least_resistance:
            reward -= 0.1 * pinfo.prev_agent_positions_heatmap[agent_pos]  # More penalty for visiting recently visited cells more often
        
        pinfo.prev_agent_positions_heatmap[agent_pos] += 0.5
    else:
        pinfo.prev_agent_positions_heatmap[agent_pos] = 0.5
    
    # Punish for taking unneccessary risks
    if agent_pos in pinfo.possible_dangerous_cells:
        return -100

    if agent_pos in pinfo.trap_cells_prev:
        return -200
    elif game_over:
        return -200

    pinfo.prev_agent_pos = agent_pos      
    return reward
