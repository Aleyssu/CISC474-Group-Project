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

# Local egocentric window for CNN: Man./Chebyshev radius 3 → (2*3+1)² = 7×7; agent at center.
_LOCAL_PATCH_RADIUS = 3


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


def _is_walkable(grid: np.ndarray, row: int, col: int) -> bool:
    """Matches env step rules: walls and enemy cells are impassable."""
    h, w = grid.shape[0], grid.shape[1]
    if row < 0 or row >= h or col < 0 or col >= w:
        return False
    cell = grid[row, col]
    if np.array_equal(cell, _WALL_RGB) or np.array_equal(cell, _ENEMY_RGB):
        return False
    return True

def _nearest_unexplored_unit_direction(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    """
    Unit vector along the first edge of a shortest path (4-neighbor BFS) from the agent
    to any unexplored cell, avoiding walls and enemies as in ``env`` movement.
    If no unexplored remains or none is reachable, returns (0, 0).
    Coordinates: positive row = down, positive col = right.
    """
    unexp = _unexplored_mask(grid) > 0.5
    if not np.any(unexp):
        return np.zeros(2, dtype=np.float32)

    h, w = grid.shape[0], grid.shape[1]
    q: deque[tuple[int, int]] = deque([(r, c)])
    visited = np.zeros((h, w), dtype=bool)
    visited[r, c] = True
    parent_r = np.full((h, w), -1, dtype=np.int16)
    parent_c = np.full((h, w), -1, dtype=np.int16)

    goal: tuple[int, int] | None = None
    # Consistent expand order (does not change shortest length; picks a tie-break).
    deltas = ((-1, 0), (0, 1), (1, 0), (0, -1))

    while q:
        y, x = q.popleft()
        if unexp[y, x]:
            goal = (y, x)
            break
        for dy, dx in deltas:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or visited[ny, nx]:
                continue
            if not _is_walkable(grid, ny, nx):
                continue
            visited[ny, nx] = True
            parent_r[ny, nx] = y
            parent_c[ny, nx] = x
            q.append((ny, nx))

    if goal is None:
        return np.zeros(2, dtype=np.float32)

    gy, gx = goal
    cy, cx = gy, gx
    while True:
        py = int(parent_r[cy, cx])
        px = int(parent_c[cy, cx])
        if py == r and px == c:
            dr = float(cy - r)
            dc = float(cx - c)
            dist = float(np.hypot(dr, dc))
            if dist < 1e-6:
                return np.zeros(2, dtype=np.float32)
            return np.array([dr / dist, dc / dist], dtype=np.float32)
        if py < 0:
            return np.zeros(2, dtype=np.float32)
        cy, cx = py, px

def observation_space(env):
    """
    Dict observation for SB3 ``MultiInputPolicy``:
    - ``patch``: one-hot cell types, shape ``(_NUM_CELL_TYPES, 2*r+1, 2*r+1)``, values in [0, 1].
    - ``vector``: BFS first-step unit direction toward nearest unexplored, shape ``(2,)``, in [-1, 1].
    """
    side = 2 * _LOCAL_PATCH_RADIUS + 1
    return gym.spaces.Dict(
        {
            "patch": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(_NUM_CELL_TYPES, side, side),
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

    r, c = _agent_row_col(grid)
    patch = _local_patch_onehot(grid, r, c, _LOCAL_PATCH_RADIUS)
    toward = _nearest_unexplored_unit_direction(grid, r, c).astype(np.float32)
    return {"patch": patch, "vector": toward}



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
        r += -10.0

    if info["cells_remaining"] == 0:
        r += 10.0 + info["steps_remaining"] * 0.05

    if info["new_cell_covered"]:
        r += 1

    curr = int(info["agent_pos"])
    pinfo.cell_visit_counts[curr] = pinfo.cell_visit_counts.get(curr, 0) + 1
    n_visits = pinfo.cell_visit_counts[curr]
    # only count an visit if it moved
    #if pinfo.pos_before_step is None or curr != pinfo.pos_before_step:
    num_revisits = n_visits - 1
    if num_revisits > 0:
        r += _REVISIT_FLAT * num_revisits
    
    
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
    #pinfo.pos_history.append(curr)
    pinfo.prev_agent_pos_for_streak = curr
    pinfo.pos_before_step = curr

    return r
