from collections import deque

import numpy as np
import gymnasium as gym
import persistent_env_info as pinfo

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# Penalize when the agent does not change cell for more than this many *consecutive* steps.
_IDLE_STREAK_THRESHOLD = 3
_IDLE_STREAK_PENALTY = -1

# Per revisit to the same cell this episode: penalty = _REVISIT_FLAT * num_revisits
# (1st time on a cell in the episode: 0; 2nd time: 1 revisit; 3rd time: 2 revisits; …).
_REVISIT_FLAT = -0.05

# Penalty for dying (spotted by an enemy).
_GAME_OVER_PENALTY = -20.0

# Penalty for standing on a cell that will enter enemy FOV on the very next step.
_NEXT_FOV_PENALTY = -1.0
_ENEMY_FOV_DISTANCE = 4  # must match env enemy_fov_distance default

# Row/col delta for each orientation (LEFT=0, DOWN=1, RIGHT=2, UP=3).
_ORIENT_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

# RGB values for each cell type (must match env.py COLOR_IDS for mapping)
_NUM_CELL_TYPES = 7
_CELL_COLORS = [
    (0, 0, 0),           # 0: unexplored
    (255, 255, 255),     # 1: explored
    (101, 67, 33),       # 2: wall
    (160, 161, 161),     # 3: agent
    (31, 198, 0),        # 4: enemy
    (255, 0, 0),         # 5: unexplored in enemy FOV
    (255, 127, 127),     # 6: explored in enemy FOV
]

_COLOR_ARRAY = np.array(_CELL_COLORS, dtype=np.uint8)
_AGENT_TYPE_ID = 3
_WALL_RGB = _COLOR_ARRAY[2]
_ENEMY_RGB = _COLOR_ARRAY[4]
_UNEXPLORED_FOV = _COLOR_ARRAY[5]
_EXPLORED_FOV = _COLOR_ARRAY[6]

# Local egocentric window for CNN: Man./Chebyshev radius 3 → (2*3+1)² = 7×7; agent at center.
_LOCAL_PATCH_RADIUS = 3
# Number of future timesteps to project enemy FOV for danger channels (t+1, t+2, t+3).
_NUM_DANGER_STEPS = 1


def _rgb_to_cell_type(cell_rgb: np.ndarray) -> int:
    for i in range(_NUM_CELL_TYPES):
        if np.array_equal(cell_rgb, _COLOR_ARRAY[i]):
            return i
    return 0


def _cell_type_at(grid: np.ndarray, row: int, col: int) -> int:
    h, w = grid.shape[0], grid.shape[1]
    if row < 0 or row >= h or col < 0 or col >= w:
        return 2  # out of bounds: same semantics as an impassable cell
    return _rgb_to_cell_type(grid[row, col])

def _agent_row_col(grid: np.ndarray) -> tuple[int, int]:
    agent = _COLOR_ARRAY[_AGENT_TYPE_ID]
    match = np.all(grid == agent, axis=2)
    ys, xs = np.where(match)
    if ys.size == 0:
        return 0, 0
    return int(ys[0]), int(xs[0])


def _local_patch_onehot(grid: np.ndarray, r: int, c: int, radius: int) -> np.ndarray:
    """
    One-hot cell types in a (2*radius+1)² window centered on the agent.
    Shape ``(_NUM_CELL_TYPES, side, side)`` channel-first for Stable-Baselines3 ``CnnPolicy``.
    Out-of-map positions are typed as wall (see ``_cell_type_at``).
    """
    side = 2 * radius + 1
    out = np.zeros((_NUM_CELL_TYPES, side, side), dtype=np.float32)
    for pi in range(side):
        for pj in range(side):
            t = _cell_type_at(grid, r + pi - radius, c + pj - radius)
            out[t, pi, pj] = 1.0
    return out


def _unexplored_mask(grid: np.ndarray) -> np.ndarray:
    """1.0 where the cell is still uncoverable unexplored (black or enemy-FOV red)."""
    black = np.all(grid == _COLOR_ARRAY[0], axis=-1)
    red = np.all(grid == _COLOR_ARRAY[5], axis=-1)
    return np.logical_or(black, red).astype(np.float32)


def _next_step_fov_cells(enemies, grid: np.ndarray) -> set:
    """
    Returns the set of (row, col) cells that will be inside enemy FOV after
    enemies rotate once (90° counterclockwise). FOV rays stop at walls and enemies,
    matching env.__spawn_fov behaviour.
    """
    h, w = grid.shape[0], grid.shape[1]
    dangerous = set()
    for enemy in enemies:
        next_orient = (enemy.orientation + 1) % 4
        dy, dx = _ORIENT_DELTA[next_orient]
        for i in range(1, _ENEMY_FOV_DISTANCE + 1):
            fy, fx = enemy.y + dy * i, enemy.x + dx * i
            if fy < 0 or fy >= h or fx < 0 or fx >= w:
                break
            cell = grid[fy, fx]
            if np.array_equal(cell, _WALL_RGB) or np.array_equal(cell, _ENEMY_RGB):
                break
            dangerous.add((fy, fx))
    return dangerous


def _future_fov_channels(grid: np.ndarray, enemies_in_sight: list, agent_r: int, agent_c: int, radius: int) -> np.ndarray:
    """
    Returns (_NUM_DANGER_STEPS, side, side) binary channels marking cells that will be
    under enemy FOV 1, 2, and 3 steps from now (channels 0, 1, 2 respectively).

    enemies_in_sight stores (row, col, eff_orientation) where eff_orientation =
    (enemy.orientation + 1) % 4 captured at the end of the previous step. This
    pre-rotation aligns it with the enemy's actual orientation at observation time
    (enemies rotate once more before each observation call), so delta 1/2/3 here
    maps exactly to 1/2/3 steps ahead from the agent's current perspective.
    """
    side = 2 * radius + 1
    channels = np.zeros((_NUM_DANGER_STEPS, side, side), dtype=np.float32)
    h, w = grid.shape[0], grid.shape[1]
    for step_idx in range(_NUM_DANGER_STEPS):
        rot_delta = step_idx + 1
        for er, ec, eff_orient in enemies_in_sight:
            fut_orient = (eff_orient + rot_delta) % 4
            dy, dx = _ORIENT_DELTA[fut_orient]
            for i in range(1, _ENEMY_FOV_DISTANCE + 1):
                fy, fx = er + dy * i, ec + dx * i
                if fy < 0 or fy >= h or fx < 0 or fx >= w:
                    break
                cell = grid[fy, fx]
                if np.array_equal(cell, _WALL_RGB) or np.array_equal(cell, _ENEMY_RGB):
                    break
                pi = fy - agent_r + radius
                pj = fx - agent_c + radius
                if 0 <= pi < side and 0 <= pj < side:
                    channels[step_idx, pi, pj] = 1.0
    return channels



def _nearest_unexplored_unit_direction(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    """
    Rolling-horizon time-expanded BFS over states (row, col, phase), phase = t % 4.

    Finds the first step of the shortest safe path to any unexplored cell, where
    "safe at step t" means the destination is not inside enemy FOV at that moment.
    Enemy FOV advances by one 90° CCW rotation per timestep (period = 4), so the
    full state space is 10×10×4 = 400 nodes — guaranteed to terminate.

    FOV sources:
    - Phases 0-3: computed from pinfo.enemies_in_sight (enemies in the 7×7 patch).
    - Phase 0 additionally augmented with all current RED/LIGHT_RED grid cells so
      distant enemies' current FOV is respected even if outside the patch.

    Actions: 4-directional moves + STAY (needed to wait for a sweeping FOV to pass).
    Returns (0, 0) when the best first action is STAY, no reachable goal exists, or
    coverage is already complete.
    Coordinates: positive row = down, positive col = right.
    """
    unexp = _unexplored_mask(grid) > 0.5
    if not np.any(unexp):
        return np.zeros(2, dtype=np.float32)

    h, w = grid.shape[0], grid.shape[1]

    # Build one FOV set per rotation phase from known nearby enemies.
    fov_sets: list[set] = []
    for phase in range(4):
        dangerous: set = set()
        for er, ec, eff_orient in pinfo.enemies_in_sight:
            fut_orient = (eff_orient + phase) % 4
            dy, dx = _ORIENT_DELTA[fut_orient]
            for i in range(1, _ENEMY_FOV_DISTANCE + 1):
                fy, fx = er + dy * i, ec + dx * i
                if fy < 0 or fy >= h or fx < 0 or fx >= w:
                    break
                cell = grid[fy, fx]
                if np.array_equal(cell, _WALL_RGB) or np.array_equal(cell, _ENEMY_RGB):
                    break
                dangerous.add((fy, fx))
        fov_sets.append(dangerous)

    # Augment phase-0 with every current RED/LIGHT_RED cell so all enemies'
    # current FOV is blocked at t=0, even those outside pinfo.enemies_in_sight.
    cur_fov_mask = (
        np.all(grid == _UNEXPLORED_FOV, axis=-1) |
        np.all(grid == _EXPLORED_FOV, axis=-1)
    )
    fy_arr, fx_arr = np.where(cur_fov_mask)
    fov_sets[0].update(zip(fy_arr.tolist(), fx_arr.tolist()))

    # Time-expanded BFS. State: (row, col, phase).
    # Queue entries: (row, col, t, first_dr, first_dc).
    # first_dr/dc stays None until the first expansion so we capture the seed's action.
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1), (0, 0))  # 4 dirs + STAY
    visited: set = {(r, c, 0)}
    q: deque = deque([(r, c, 0, None, None)])

    while q:
        y, x, t, fdr, fdc = q.popleft()

        if unexp[y, x] and (y, x) not in fov_sets[t % 4]:
            if fdr is None:
                return np.zeros(2, dtype=np.float32)
            return np.array([float(fdr), float(fdc)], dtype=np.float32)

        nphase = (t + 1) % 4
        for dr, dc in moves:
            ny, nx = y + dr, x + dc
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if np.array_equal(grid[ny, nx], _WALL_RGB) or np.array_equal(grid[ny, nx], _ENEMY_RGB):
                continue
            if (ny, nx) in fov_sets[nphase]:
                continue
            key = (ny, nx, nphase)
            if key in visited:
                continue
            visited.add(key)
            q.append((ny, nx, t + 1,
                       dr if fdr is None else fdr,
                       dc if fdc is None else fdc))
            
    # all search exhausted (maybe intentionally run into enemy or try running directly towards unexplored cell)
    print('weird')
    return np.zeros(2, dtype=np.float32)

def observation_space(env):
    """
    Dict observation for SB3 ``MultiInputPolicy``:
    - ``patch``: one-hot cell types + future FOV danger, shape ``(_NUM_CELL_TYPES + _NUM_DANGER_STEPS, 2*r+1, 2*r+1)``.
      Channels 0-6: cell type one-hot. Channels 7-9: binary danger at t+1, t+2, t+3.
    - ``vector``: BFS first-step unit direction toward nearest unexplored, shape ``(2,)``, in [-1, 1].
    """
    side = 2 * _LOCAL_PATCH_RADIUS + 1
    return gym.spaces.Dict(
        {
            "patch": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(_NUM_CELL_TYPES + _NUM_DANGER_STEPS, side, side),
                dtype=np.float32,
            ),
            "vector": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
        }
    )


def observation(grid):
    H, W, _ = grid.shape
    cleared_cells = np.zeros((H, W), dtype=np.uint8)
    mask = np.logical_or(np.all(grid == (255, 255, 255), axis=-1), np.all(grid == (255, 127, 127), axis=-1))
    cleared_cells[mask] = 1
    if np.sum(cleared_cells) == 0:
        pinfo.reset_episode()

    pinfo.last_grid = grid
    r, c = _agent_row_col(grid)
    patch = _local_patch_onehot(grid, r, c, _LOCAL_PATCH_RADIUS)
    danger = _future_fov_channels(grid, pinfo.enemies_in_sight, r, c, _LOCAL_PATCH_RADIUS)
    toward = _nearest_unexplored_unit_direction(grid, r, c).astype(np.float32)
    return {"patch": np.concatenate([patch, danger], axis=0), "vector": toward}



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

    Idle penalty: consecutive steps with unchanged ``agent_pos`` (STAY, blocked move,
    etc.). When streak exceeds ``_IDLE_STREAK_THRESHOLD`` (default 3), i.e. from the
    4th consecutive step without moving, add ``_IDLE_STREAK_PENALTY``.

    Revisit penalty: each time the agent ends a step on a cell, add
    ``_REVISIT_FLAT * (n - 1)`` where ``n`` is how many times that cell has been
    occupied at step ends this episode (so first occupancy is free; 2nd adds
    ``_REVISIT_FLAT * 1``, 3rd adds ``_REVISIT_FLAT * 2``, …).
    """
    r = 0
    if info["game_over"]:
        r += _GAME_OVER_PENALTY

    if info["cells_remaining"] == 0:
        r += 10.0 + info["steps_remaining"] * 0.05

    if info["new_cell_covered"]:
        r += 1

    curr = int(info["agent_pos"])
    pinfo.cell_visit_counts[curr] = pinfo.cell_visit_counts.get(curr, 0) + 1
    n_visits = pinfo.cell_visit_counts[curr]
    # only count an visit if it moved
    if pinfo.pos_before_step is None or curr != pinfo.pos_before_step:
        num_revisits = n_visits - 1
        if num_revisits > 0:
            r += _REVISIT_FLAT * num_revisits
    
    # should never have to wait more than 3 steps for clear
    if pinfo.prev_agent_pos_for_streak is None:
        pinfo.same_cell_streak = 0
    elif curr == pinfo.prev_agent_pos_for_streak:
        pinfo.same_cell_streak += 1
    else:
        pinfo.same_cell_streak = 0

    if pinfo.same_cell_streak > _IDLE_STREAK_THRESHOLD:
        r += _IDLE_STREAK_PENALTY

    
    '''
    if (
        pinfo.last_action is not None
        and pinfo.pos_before_step is not None
        and pinfo.last_action != 4
        and curr == pinfo.pos_before_step
    ):
        r += -1.0
    '''

    if pinfo.last_grid is not None and info["enemies"]:
        agent_row, agent_col = curr // 10, curr % 10
        if (agent_row, agent_col) in _next_step_fov_cells(info["enemies"], pinfo.last_grid):
            if info["new_cell_covered"]:
                # Moved to a new unexplored cell — exploring is worth the risk.
                r += _NEXT_FOV_PENALTY * 0.1
            else:
                r += _NEXT_FOV_PENALTY

    #pinfo.pos_history.append(curr)
    pinfo.prev_agent_pos_for_streak = curr
    pinfo.pos_before_step = curr
    pinfo.last_step_coverable_cell_count = info["total_covered_cells"]

    # Collect nearby enemies for the next observation's danger channels.
    # Store (enemy.orientation + 1) % 4 so it aligns with the enemy's actual
    # orientation at the next observation() call (enemies rotate once more before then).
    agent_row, agent_col = curr // 10, curr % 10
    pinfo.enemies_in_sight = [
        (enemy.y, enemy.x, (enemy.orientation + 1) % 4)
        for enemy in info["enemies"]
        if max(abs(enemy.y - agent_row), abs(enemy.x - agent_col)) <= _LOCAL_PATCH_RADIUS
    ]

    return r
