import numpy as np
import gymnasium as gym
import persistent_env_info as pinfo
from collections import deque

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
_LOCAL_ENEMY_RADIUS = _ENEMY_FOV_DISTANCE + 2
# Number of future timesteps to project enemy FOV for danger channels.
# 3 covers the full 4-phase enemy rotation cycle (t+0 is visible as red on the grid itself).
_NUM_DANGER_STEPS = 3

# Semantic binary channels per patch cell (see _local_patch_semantic).
_NUM_SEMANTIC_CHANNELS = 6  # wall | explored | unexplored | agent | enemy | current_fov


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


def _local_patch_semantic(grid: np.ndarray, r: int, c: int, radius: int) -> np.ndarray:
    """
    6 independent binary channels for the (2*radius+1)² window centered on the agent:
      ch 0 — wall (including out-of-bounds)
      ch 1 — explored (visited, with or without FOV overlay)
      ch 2 — unexplored (not yet visited, with or without FOV overlay)
      ch 3 — agent (center pixel)
      ch 4 — enemy body
      ch 5 — currently in enemy FOV (red = unvisited, pink = visited)
    A cell can set multiple channels simultaneously (e.g. unexplored=1 AND current_fov=1).
    Shape: (_NUM_SEMANTIC_CHANNELS, side, side), float32, values in {0, 1}.
    """
    side = 2 * radius + 1
    out = np.zeros((_NUM_SEMANTIC_CHANNELS, side, side), dtype=np.float32)
    for pi in range(side):
        for pj in range(side):
            t = _cell_type_at(grid, r + pi - radius, c + pj - radius)
            if t == 2:          # wall / OOB
                out[0, pi, pj] = 1.0
            elif t in (1, 6):   # explored (white or pink)
                out[1, pi, pj] = 1.0
            elif t in (0, 5):   # unexplored (black or red)
                out[2, pi, pj] = 1.0
            elif t == 3:        # agent
                out[3, pi, pj] = 1.0
            elif t == 4:        # enemy body
                out[4, pi, pj] = 1.0
            if t in (5, 6):     # currently in FOV (red or pink) — can overlap with ch1/ch2
                out[5, pi, pj] = 1.0
    return out


def _unexplored_mask(grid: np.ndarray) -> np.ndarray:
    """1.0 where the cell is unexplored (black = never visited, red = in enemy FOV but unvisited)."""
    black = np.all(grid == _COLOR_ARRAY[0], axis=-1)
    red = np.all(grid == _UNEXPLORED_FOV, axis=-1)  # unvisited cell currently under enemy FOV
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



def _global_nav_signal(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    """
    5-element global navigation signal giving a density-based view of where
    unexplored cells are relative to the agent — no pathfinding, no FOV logic.

    Returns float32 array of shape (5,):
      [0] NW fraction  — share of unexplored cells in rows < r, cols < c
      [1] NE fraction  — share of unexplored cells in rows < r, cols >= c
      [2] SW fraction  — share of unexplored cells in rows >= r, cols < c
      [3] SE fraction  — share of unexplored cells in rows >= r, cols >= c
      [4] total fraction — unexplored / coverable cells (how much is left)

    Quadrant fractions sum to 1 (or are all 0 when fully explored).
    Walls and enemy positions are ignored — the agent sees those in the patch.
    """
    unexp = _unexplored_mask(grid)  # (H, W) float32
    total = float(unexp.sum())
    if total == 0.0:
        return np.zeros(5, dtype=np.float32)

    coverable = float(np.sum(
        ~np.all(grid == _WALL_RGB, axis=-1) & ~np.all(grid == _ENEMY_RGB, axis=-1)
    ))

    nw = float(unexp[:r, :c].sum())
    ne = float(unexp[:r, c:].sum())
    sw = float(unexp[r:, :c].sum())
    se = float(unexp[r:, c:].sum())

    return np.array([nw / total, ne / total, sw / total, se / total,
                     total / max(coverable, 1.0)], dtype=np.float32)


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
    
    # On the very first observation enemies_in_sight hasn't been populated yet.
    # Stay put so we don't walk into an unseen FOV on the very first step.
    if (not pinfo.enemies_in_sight
            and pinfo.pos_before_step is None
            and np.any(np.all(grid == _ENEMY_RGB, axis=-1))):
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

    # Order moves toward denser quadrants first so BFS prefers clustered areas.
    nw = float(unexp[:r, :c].sum())
    ne = float(unexp[:r, c:].sum())
    sw = float(unexp[r:, :c].sum())
    se = float(unexp[r:, c:].sum())
    move_scores = {
        (-1, 0): nw + ne,   # up → toward north quadrants
        (1, 0):  sw + se,   # down → toward south quadrants
        (0, -1): nw + sw,   # left → toward west quadrants
        (0, 1):  ne + se,   # right → toward east quadrants
        (0, 0):  -1,        # STAY — always last
    }
    moves = sorted(move_scores, key=move_scores.get, reverse=True)
    visited: set = {(r, c, 0)}
    q: deque = deque([(r, c, 0, None, None)])

    while q:
        y, x, t, fdr, fdc = q.popleft()

        if unexp[y, x] and (y, x) not in fov_sets[t % 4]:
            # Trap check: can the agent leave this cell safely next step?
            escape_phase = (t + 1) % 4
            has_exit = False
            for edr, edc in moves:
                ey, ex = y + edr, x + edc
                if ey < 0 or ey >= h or ex < 0 or ex >= w:
                    continue
                if np.array_equal(grid[ey, ex], _WALL_RGB) or np.array_equal(grid[ey, ex], _ENEMY_RGB):
                    continue
                if (ey, ex) not in fov_sets[escape_phase]:
                    has_exit = True
                    break
            if not has_exit:
                continue  # trapped — skip this cell, keep searching
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
    - ``patch``: semantic binary channels + future FOV danger, shape ``(_NUM_SEMANTIC_CHANNELS + _NUM_DANGER_STEPS, 2*r+1, 2*r+1)`` = (9, 7, 7).
      Ch 0: wall  Ch 1: explored  Ch 2: unexplored  Ch 3: agent  Ch 4: enemy  Ch 5: current FOV
      Ch 6: danger t+1  Ch 7: danger t+2  Ch 8: danger t+3
    - ``vector``: BFS direction (2,) — unit first-step toward nearest unexplored cell.
    """
    side = 2 * _LOCAL_PATCH_RADIUS + 1
    return gym.spaces.Dict(
        {
            "patch": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(_NUM_SEMANTIC_CHANNELS + _NUM_DANGER_STEPS, side, side),
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
    if np.sum(cleared_cells) == 0 and not pinfo.episode_reset_done:
        pinfo.reset_episode()
        pinfo.episode_reset_done = True  # prevent re-clearing on STAY steps

    pinfo.last_grid = grid
    r, c = _agent_row_col(grid)
    patch = _local_patch_semantic(grid, r, c, _LOCAL_PATCH_RADIUS)
    danger = _future_fov_channels(grid, pinfo.enemies_in_sight, r, c, _LOCAL_PATCH_RADIUS)
    direction = _nearest_unexplored_unit_direction(grid, r, c)
    return {"patch": np.concatenate([patch, danger], axis=0), "vector": direction}



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

    n_enemies = len(info["enemies"])
    # Scale difficulty linearly with enemy count.
    # 0 enemies → easy (fast finish rewarded); 4 enemies → hard (completion rewarded).
    # Clamp at 4 so extra enemies on bonus maps don't break the scale.
    t = min(n_enemies, 4) / 4.0  # 0.0 (no enemies) → 1.0 (4 enemies)
    finish_bonus          = 10  + 40  * t   # 10  → 50
    step_remain_multiplier = 0.4 - 0.36 * t  # 0.4 → 0.04
    coverage_bonus        = 0.5 + 0.5  * t   # 2.0 → 2.5
    revisit_penalty = -0.1 + 0.08 * t

    if info["cells_remaining"] == 0:
        r += finish_bonus + info["steps_remaining"] * step_remain_multiplier

    if info["new_cell_covered"]:
        r += coverage_bonus

    
    if info["cells_remaining"] > 0 and info["steps_remaining"] == 0:
        r += -100
    
    
    curr = int(info["agent_pos"])
    pinfo.cell_visit_counts[curr] = pinfo.cell_visit_counts.get(curr, 0) + 1
    n_visits = pinfo.cell_visit_counts[curr]
    # Only penalise revisits when the agent actually moved to the cell.
    # Without this guard, STAY increments the count every step and the
    # penalty grows quadratically (−0.1 * 1, −0.1 * 2, …, −0.1 * 499).
    if pinfo.pos_before_step is None or curr != pinfo.pos_before_step:
        num_revisits = n_visits - 1
        if num_revisits > 0:
            r += revisit_penalty * num_revisits
    
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
    '''
    if pinfo.last_grid is not None and info["enemies"]:
        agent_row, agent_col = curr // 10, curr % 10
        if (agent_row, agent_col) in _next_step_fov_cells(info["enemies"], pinfo.last_grid):
            if info["new_cell_covered"]:
                # Moved to a new unexplored cell — exploring is worth the risk.
                r += _NEXT_FOV_PENALTY * 0.1
            else:
                r += _NEXT_FOV_PENALTY
    '''
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
        if max(abs(enemy.y - agent_row), abs(enemy.x - agent_col)) <= _LOCAL_ENEMY_RADIUS
    ]

    # Re-arm the episode-reset guard so the next episode's first observation()
    # call will properly clear all state.
    if info["game_over"] or info["steps_remaining"] == 0 or info["cells_remaining"] == 0:
        pinfo.episode_reset_done = False

    return r