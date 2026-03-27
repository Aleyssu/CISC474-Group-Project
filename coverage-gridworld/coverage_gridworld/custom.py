import numpy as np
import gymnasium as gym
from coverage_gridworld import persistent_env_info as pinfo
from collections import deque

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

np.set_printoptions(legacy='1.25') 

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


def possibly_from_other_enemy(danger_pos: tuple, origin_direction: tuple) -> int:
    """
    Determines if the given danger cell is possibly from an enemy other than the one in the origin direction

    Returns 1 if said direction is perpendicular to the original enemy's sightline, 2 if it is in front of
    the original enemy, and 0 otherwise.

    The return '2' scenario is important so that we know there's no point in scanning that sightline further
    as no more information can be derived from that direction. 
    """
    danger_grid = pinfo.enemy_wall_danger_map[pinfo.env_orientation]
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
                        if DIRECTION_MAP[(ori + pinfo.env_orientation) % 4] == (dir[0] * -1, dir[1] * -1):
                            # Check if the other enemy is facing the original enemy
                            if dir == (origin_direction[0] * -1, origin_direction[1] * -1):
                                return 2
                            else:
                                return 1
                # Stop checking if there are no more danger cells ahead
                break
    return 0


# Note that enemies always rotate counterclockwise and project danger cells 4 cells out which are blocked by walls and other enemies.
def update_predicted_enemy_orientations() -> None:
    """
    Given a integer matrix representing the game grid with the following mappings: 
    0: empty space
    1: danger cells
    2: enemy cells 
    3: walls
    Tries to predict where the enemies are oriented based on where the danger cells are being projected.

    Updates the persistent_env_info on the possible orientations for unknown enemies.
    """
    danger_grid = pinfo.enemy_wall_danger_map[pinfo.env_orientation]
    H, W = danger_grid.shape
    deduced_enemies = []
    for enemy_pos in pinfo.unknown_enemies.copy():
        deduced_enemy = False
        eliminated_dirs = []
        
        for dir in pinfo.unknown_enemy_possible_orientations[enemy_pos]:
            if deduced_enemy:
                break
            dir_yx = DIRECTION_MAP[(dir + pinfo.env_orientation) % 4]
            for i in range(1, 5):
                proj_pos = (i * dir_yx[0] + enemy_pos[0], i * dir_yx[1] + enemy_pos[1])
                # Check for projected sightline being inbounds
                if 0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W:
                    if danger_grid[proj_pos] == 1:
                        from_other_enemy = possibly_from_other_enemy(proj_pos, (dir_yx[0] * -1, dir_yx[1] * -1))
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


def project_enemy_view(enemy_pos: tuple, dir: tuple):
    """
    Returns a list of (y, x) tuples indicating the cell coordinates which can be seen by the enemy
    if they're facing the given direction.
    """
    H, W = pinfo.enemy_wall_map.shape
    viewed_cells = []
    for i in range(1, 5):
        proj_pos = (i * dir[0] + enemy_pos[0], i * dir[1] + enemy_pos[1])
        # Check for projected sightline being inbounds and unobscured
        if 0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W and pinfo.enemy_wall_map[proj_pos] == 0:
            viewed_cells.append(proj_pos)
        else:
            break

    return viewed_cells


def is_trap(pos: tuple, env_ori: int, recurse=1) -> bool:
    """
    Returns true if being in the given (y, x) position at the given environment orientation 
    is guaranteed to get the agent caught in the future, else false.

    Recursively checks the given amount of times to look even deeper into the future for
    multi-step guaranteed fails.
    """
    H, W = pinfo.enemy_wall_map.shape
    
    # Don't bother checking the given position is a wall - return true for the recursion loop
    if pinfo.enemy_wall_map[pos] != 0:
        return True 

    # Check possible movement directions for a safe tile
    for dir in [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]:  # left, up, down, right, current position
        proj_pos = (pos[0] + dir[0], pos[1] + dir[1])
        # Position is out of bounds or an obstacle
        if not (0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W):
            proj_pos = pos
        # Position is clear
        if pinfo.enemy_wall_danger_map[(env_ori + 1) % 4][proj_pos] == 0:
            if recurse > 0:
                if is_trap(proj_pos, (env_ori + 1) % 4, recurse - 1):
                    continue
                else:
                    return False
            else:
                return False
    pinfo.enemy_wall_danger_map[env_ori][pos] = 5  # Update danger mappings to save on future computations
    # print("Trap found %s %s" % (pos, env_ori))
    # for x in pinfo.enemy_wall_danger_map:
    #     print(pinfo.enemy_wall_danger_map[x])
    return True


def bfs_empty_cell(enemy_wall_map: np.ndarray, explored_map: np.ndarray, agent_pos: tuple):
    """
    Using breadth first search, generates a list of (y, x) cell coords painting a greedy path leading towards the
    nearest empty cell for the agent. Saves the result in pinfo.bfs_cell_coords for the agent to use.

    explored_map is a binary matrix with 1 for explored cells and 0 for any other cells.

    Note: does not account for enemies
    """
    H, W = enemy_wall_map.shape
    q = deque([agent_pos])  # FIFO queue to track positions to search
    pos_trace = dict()  # Contains (y, x) : (y, x) pairs mapping a position back to the agent
    path = deque()  # Stack of (y, x) tuples leading the agent towards the nearest empty cell
    checked_cells = set()

    found_empty_cell = False
    while len(q) > 0 and not found_empty_cell:
        pos = q.popleft()

        # Skip if we've already checked this cell before
        if pos in checked_cells:
            continue

        for dir in DIRECTIONS:
            proj_pos = (dir[0] + pos[0], dir[1] + pos[1])

            # Skip if direction is out of bounds, obstructed, or already checked before
            if not (0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W) or enemy_wall_map[proj_pos] != 0 or proj_pos in checked_cells:
                continue
            # Found empty cell - construct path leading from the agent to the cell
            elif explored_map[proj_pos] == 0:
                found_empty_cell = True
                path.append(proj_pos)
                while pos != agent_pos:
                    path.append(pos)
                    pos = pos_trace[pos]
                break
            else:
                pos_trace[proj_pos] = pos
                q.append(proj_pos)

        checked_cells.add(pos)
    
    if not found_empty_cell:
        pinfo.following_bfs = False
    else:
        pinfo.bfs_cell_coords = path
        pinfo.following_bfs = True       
        pinfo.just_started_following_bfs = True


def advanced_bfs_empty_cell(explored_map: np.ndarray, agent_pos: tuple):
    """
    Using BFS search with consideration for predicted danger cells, generates a list of (y, x) 
    cell coords painting a safe path leading towards the nearest empty cell for the agent. 
    
    Saves the result in pinfo.bfs_cell_coords for the agent to use.

    Note: due to backtracking and standing still being exploration options to account for
    the 4 different possible enemy orientations, the search space is effectively quadrupled
    when using this algorithm instead of the above bfs algorithm. In a nutshell, this one is 
    much more computationally expensive.

    If force_danger is True, the algorithm will treat danger cells as valid search targets.
    """
    H, W = pinfo.enemy_wall_map.shape
    agent_posq = (agent_pos, pinfo.env_orientation - 1)
    q = deque([agent_posq])  # FIFO queue to track positions to search
    pos_trace = dict()  # Contains ((y, x), i): ((y, x), i) pairs mapping a position back to the agent with i being the relative rotation of the enemies
    path = deque()  # Stack of (y, x) tuples leading the agent towards the nearest empty cell
    checked_cells = set()

    found_empty_cell = False
    while len(q) > 0 and not found_empty_cell:
        pos, env_ori = q.popleft()
        next_env_ori = (env_ori + 1) % 4

        for dir in [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]:  # left, up, down, right, stay
            proj_pos = (dir[0] + pos[0], dir[1] + pos[1])

            # Skip if direction is out of bounds, obstructed, or already checked before
            if not (0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W) or proj_pos in checked_cells:
                continue
            # Skip obstacles as well
            elif pinfo.enemy_wall_danger_map[next_env_ori][proj_pos] in (2, 3):
                checked_cells.add((pos, env_ori))
                continue
            elif pinfo.enemy_wall_danger_map[next_env_ori][proj_pos] not in (0, 5):
                # Target danger cell if we're trying to end the run asap
                if pinfo.push_danger and pinfo.enemy_wall_danger_map[next_env_ori][proj_pos] == 1:
                    found_empty_cell = True
                    path.append(proj_pos)
                    while (pos, env_ori) != agent_posq:
                        path.append(pos)
                        pos, env_ori = pos_trace[(pos, env_ori)]
                    break
                else:
                    continue
            # Found empty cell - construct path leading from the agent to the cell
            elif explored_map[proj_pos] == 0:
                # Skip if empty cell is a trap cell and we're not looking to die yet
                if not pinfo.push_traps and (pinfo.enemy_wall_danger_map[next_env_ori][proj_pos] == 5 or is_trap(proj_pos, next_env_ori)):
                    # print("Found projected danger %s %s %s" % (pos, proj_pos, abs(next_env_ori - pinfo.env_orientation + 1) % 4))
                    continue
                found_empty_cell = True
                path.append(proj_pos)
                # print("%s: %s" % ((proj_pos, next_env_ori), (pos, env_ori)))
                while (pos, env_ori) != agent_posq:
                    path.append(pos)
                    pos, env_ori = pos_trace[(pos, env_ori)]
                break
            # Cell is white
            else:
                posq = (proj_pos, next_env_ori)
                if not (posq in checked_cells or posq in q):
                    q.append(posq)
                if not posq in checked_cells:
                    pos_trace[posq] = (pos, env_ori)
                    # print("%s: %s" % (posq, (pos, env_ori)))

        checked_cells.add((pos, env_ori))
    
    if not found_empty_cell:
        pinfo.following_bfs = False
    else:
        pinfo.bfs_cell_coords = path
        pinfo.following_bfs = True       
        pinfo.just_started_following_bfs = True


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    H, W, _ = grid.shape

    # Reset persistent info if all cells are uncleared (new episode)
    cleared_cells = np.zeros((H, W), dtype=np.uint8)
    mask = np.logical_or(np.all(grid == WHITE, axis=-1), np.all(grid == LIGHT_RED, axis=-1))
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

        # Get current danger cells and update them in memory
        danger_mask = np.logical_or(np.all(grid == RED, axis=-1), np.all(grid == LIGHT_RED, axis=-1))
        pinfo.enemy_wall_map = 2 * enemy_mask + 3 * wall_mask 
        danger_grid = danger_mask + pinfo.enemy_wall_map  # Create a grid where 0 is safe, 1 is danger, 2 is enemy, and 3 is wall
        pinfo.enemy_wall_danger_map[pinfo.env_orientation] = danger_grid

        for i in range(1, 4):
            pinfo.enemy_wall_danger_map[i] = pinfo.enemy_wall_map.copy()

        update_predicted_enemy_orientations()
    # Try to deduce unknown enemy orientations if there's any left
    elif len(pinfo.unknown_enemies) > 0:
        danger_mask = np.logical_or(np.all(grid == RED, axis=-1), np.all(grid == LIGHT_RED, axis=-1))
        danger_grid = danger_mask + pinfo.enemy_wall_map
        pinfo.enemy_wall_danger_map[pinfo.env_orientation] = danger_grid
        update_predicted_enemy_orientations()
    
    # Start looking one time step into the future
    pinfo.env_orientation = (pinfo.env_orientation + 1) % 4  

    if len(pinfo.unknown_enemies) == 0:
        # When the enemy orientations are known, finish computing the rest of the danger cell layouts
        if not pinfo.enemy_orientations_computed:
            for i in range(pinfo.env_orientation, 4):
                # Compute predicted enemy orientations and projected danger cells in the given time step
                for enemy in pinfo.enemy_positions:
                    projected_ori = DIRECTION_MAP[(pinfo.enemy_orientations[enemy] + i) % 4]
                    for cell in project_enemy_view(enemy, projected_ori):
                        pinfo.enemy_wall_danger_map[i][cell] = 1
            pinfo.enemy_orientations_computed = True
            
            # print("Deduced Enemies:\n%s" % pinfo.enemy_wall_danger_map)
            pinfo.possible_dangerous_cells = set()

    # Otherwise, project caution cells where danger cells might be in the next time step
    else:            
        pinfo.enemy_wall_danger_map[pinfo.env_orientation] = pinfo.enemy_wall_map.copy()
        # Predict where the danger cells will be in the next time step
        possible_danger_cells = set()  # Contains (y, x) tuples for cells that might be dangerous in the next time step
        pinfo.possible_dangerous_cells = set()
        for enemy in pinfo.enemy_positions:
            # Cells that may be seen by enemies with yet to be determined orientations
            if enemy in pinfo.unknown_enemies:
                for dir in pinfo.unknown_enemy_possible_orientations[enemy]:
                    for cell in project_enemy_view(enemy, DIRECTION_MAP[(dir + pinfo.env_orientation) % 4]):
                        possible_danger_cells.add(cell)
                        pinfo.possible_dangerous_cells.add(cell[0] * W + cell[1])
            # Cells that will certainly be seen by enemies
            else:
                for i in range(pinfo.env_orientation, 4):
                    for cell in project_enemy_view(enemy, DIRECTION_MAP[(pinfo.enemy_orientations[enemy] + i) % 4]):
                        pinfo.enemy_wall_danger_map[i][cell] = 1

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
    # Flood the agent's senses with pheromones as they're dying during a last stand so that they don't learn to associate anything else with reward
    if pinfo.is_dying:
        ohe[:, COLOR_MAP[PATH_OF_LEAST_RESISTANCE]] = 1
        return ohe
    
    agent_pos = (agent_pos[0][0], agent_pos[1][0])
    cleared_cells[agent_pos] = 1

    pinfo.trap_cells_prev = pinfo.trap_cells
    pinfo.trap_cells = set()
    # Update observation for adjacent cells and current cell with semi-OHE encoding
    observation_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # up, right, down, left, stay - don't change the order of these pls
    black_adjacent = False  # Indicates there's a safe unexplored tile in the immediate vicinity
    adjacent_explored_cells = []  # Indices of observation_offsets indicating directions of adjacent explored cells
    curr_cell_danger = 0  # Indicates if the agent's current cell will be dangerous in the next time step
    for i in range(4, -1, -1):
        action_offset = observation_offsets[i]

        # Check if adjacent cell is within bounds and get its color otherwise mark it as out of bounds
        proj_pos = (agent_pos[0] + action_offset[0], agent_pos[1] + action_offset[1])
        proj_pos_to_int = (proj_pos[0] * W + proj_pos[1])
        if 0 <= proj_pos[0] < H and 0 <= proj_pos[1] < W:
            # Position is dangerous
            if pinfo.enemy_wall_danger_map[pinfo.env_orientation][proj_pos] == 1:
                color = COLOR_MAP[DANGER]
                ohe[i, color] = 1
                if i == 4: curr_cell_danger = 1
            # Position is a previously computed trap (guaranteed to get the agent caught in the future)
            elif pinfo.enemy_wall_danger_map[pinfo.env_orientation][proj_pos] == 5 and not pinfo.push_traps:
                color = COLOR_MAP[DANGER]
                if i == 4: curr_cell_danger = 1
                ohe[i, color] = 1
            # Position is a trap (freshly computed)
            elif is_trap(proj_pos, pinfo.env_orientation) and not pinfo.push_traps:
                color = COLOR_MAP[DANGER]
                ohe[i, color] = 1
                pinfo.trap_cells.add(proj_pos_to_int)
                # print(pinfo.env_orientation, pinfo.enemy_wall_danger_map)
                if i == 4: curr_cell_danger = 1
            # Position might be dangerous
            elif not pinfo.enemy_orientations_computed and proj_pos in possible_danger_cells:
                color = COLOR_MAP[POSSIBLE_DANGER]
                ohe[i, color] = 1
                if i == 4: curr_cell_danger = 2
            else:
                color = COLOR_MAP[tuple(grid[proj_pos])]
                # Position is white (and safe to go to)
                if color == COLOR_MAP[WHITE]:
                    ohe[i, color] = 1
                    adjacent_explored_cells.append(i)
                # Position is unexplored (and safe to go to)
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
                # Any other position
                else:
                    ohe[i, color] = 1
        # Position is out of bounds - treat with the same logic as walls
        else:
            if curr_cell_danger == 2:
                color = COLOR_MAP[POSSIBLE_DANGER]
            elif curr_cell_danger == 1:
                color = COLOR_MAP[DANGER]
            else:
                color = COLOR_MAP[OUT_BOUNDS]
            ohe[i, color] = 1
        
    # Navigate the agent towards the nearest empty cell if there's no immediate empty cells in its vicinity
    if not black_adjacent or pinfo.following_bfs:
        if not pinfo.following_bfs or len(pinfo.bfs_cell_coords) == 0:
            advanced_bfs_empty_cell(cleared_cells, agent_pos)
            # print("Recomputed BFS: %s" % pinfo.bfs_cell_coords)
            if not pinfo.following_bfs:
                # If the only reachable empty cell is a trap, kamikaze it
                pinfo.push_traps = True
                advanced_bfs_empty_cell(cleared_cells, agent_pos)
                # print("Recomputed BFS (Trap): %s" % pinfo.bfs_cell_coords)

                # In case all remaining empty cells are impossible to clear, push into a danger cell asap to end the run
                if not pinfo.following_bfs:    
                    pinfo.push_danger = True
                    advanced_bfs_empty_cell(cleared_cells, agent_pos)
                    # print("Recomputed BFS (Danger): %s" % pinfo.bfs_cell_coords)
        # Painting a "pheromone trail" for the agent if we're following the path generated by BFS
        if pinfo.following_bfs and len(pinfo.bfs_cell_coords) > 0:
            bfs_viable = False
            target_cell = pinfo.bfs_cell_coords.pop()
            if pinfo.just_started_following_bfs:
                pinfo.path_of_least_resistance = agent_pos[0] * W + agent_pos[1]
                pinfo.just_started_following_bfs = False
            # Find the direction of the path cell and update the OHE
            for i in range(5):
                offset = observation_offsets[i]
                if target_cell == (offset[0] + agent_pos[0], offset[1] + agent_pos[1]):
                    bfs_viable = True
                    ohe[i, :] = 0
                    ohe[i, COLOR_MAP[PATH_OF_LEAST_RESISTANCE]] = 1       
                    pinfo.prev_path_of_least_resistance = pinfo.path_of_least_resistance
                    pinfo.path_of_least_resistance = target_cell[0] * W + target_cell[1]   
                    break
            if bfs_viable:
                # Make all other cells look dangerous if we're following the BFS policy because it KEEPS IGNORING MY TRAIL NO MATTER HOW I TRAIN IT ISTG
                if not pinfo.push_danger:
                    for i in range(5):
                        if ohe[i, COLOR_MAP[PATH_OF_LEAST_RESISTANCE]] != 1:
                            ohe[i, :] = 0
                            ohe[i, COLOR_MAP[DANGER]] = 1
                # Make danger cells look enticing if we're trying to die
                else:
                    for i in range(5):
                        if ohe[i, COLOR_MAP[DANGER]] == 1:
                            ohe[i, COLOR_MAP[PATH_OF_LEAST_RESISTANCE]] = 1
                            ohe[i, COLOR_MAP[DANGER]] = 0

    else:
        # The agent has just reached a trap cell during its last stand
        if pinfo.push_traps and len(pinfo.bfs_cell_coords) == 0:
            pinfo.is_dying = True

    return ohe


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

    reward = 0
    
    if new_cell_covered:
        reward += 1

    # Reward for remaining moves upon level completion
    if cells_remaining == 0:
        return steps_remaining
    
    # Punish for not following the pheromone trail produced by BFS
    if pinfo.following_bfs: 
        if not agent_pos == pinfo.prev_path_of_least_resistance:
            if not new_cell_covered:
                reward -= 10
            pinfo.following_bfs = False
    
    # Punish for taking unneccessary risks
    if agent_pos in pinfo.possible_dangerous_cells:
        return -100

    # Reward agent for making a last stand in impossible maps and punish for going into danger cells otherwise
    if agent_pos in pinfo.trap_cells_prev or game_over:
        if pinfo.push_danger or (pinfo.push_traps and len(pinfo.bfs_cell_coords) <= 1):
            return steps_remaining
        else:
            return -200

    pinfo.prev_agent_pos = agent_pos      
    return reward
