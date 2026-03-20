from collections import deque

import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# RGB values for each cell type (must match env.py COLOR_IDS for mapping)
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

# Neighbor offsets for frontier computation
_NEIGHBOR_OFFSETS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def _rgb_to_cell_ids(grid: np.ndarray) -> np.ndarray:
    H, W, _ = grid.shape
    flat = grid.reshape(-1, 3)
    matches = np.all(flat[:, None, :] == _COLOR_ARRAY[None, :, :], axis=-1)
    return np.argmax(matches, axis=-1).reshape(H, W)


_DIR_DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # LEFT, DOWN, RIGHT, UP


def _compute_future_dangers(cell_ids, H, W, fov_distance=4):
    """Compute danger maps for t+1, t+2, t+3 based on deterministic rotation.

    Enemies rotate 90 degrees clockwise each step, cycling through all 4
    directions in 4 steps.  Together with the ground-truth t+0 danger
    channel, these 3 maps give the agent full visibility of the complete
    enemy rotation cycle so it can plan safe traversals.
    """
    dangers = [np.zeros((H, W), dtype=np.float32) for _ in range(3)]
    enemy_positions = np.argwhere(cell_ids == 4)

    for ey, ex in enemy_positions:
        current_orient = None
        for orient, (dy, dx) in enumerate(_DIR_DELTAS):
            ny, nx = ey + dy, ex + dx
            if 0 <= ny < H and 0 <= nx < W and cell_ids[ny, nx] in (5, 6):
                current_orient = orient
                break

        if current_orient is None:
            continue

        for step in range(3):
            future_orient = (current_orient + step + 1) % 4
            dy, dx = _DIR_DELTAS[future_orient]
            for i in range(1, fov_distance + 1):
                ny, nx = ey + dy * i, ex + dx * i
                if not (0 <= ny < H and 0 <= nx < W):
                    break
                if cell_ids[ny, nx] in (2, 4):
                    break
                dangers[step][ny, nx] = 1.0

    return dangers


def _compute_frontier_distance(walls, enemy_pos, visited, H, W):
    """Multi-source BFS from frontier cells; returns normalized proximity field.

    Cells near the frontier get values close to 1.0, far cells get values
    closer to 0.0.  Walls and enemy positions block the BFS (matching the
    movement rules in env.py).
    """
    frontier = compute_frontier(walls, visited)
    sources = np.argwhere(frontier > 0)
    if len(sources) == 0:
        return np.zeros((H, W), dtype=np.float32)

    blocked = (walls > 0) | (enemy_pos > 0)
    dist = np.full((H, W), -1, dtype=np.int32)
    q = deque()
    for r, c in sources:
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not blocked[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    reachable = dist >= 0
    max_d = int(dist[reachable].max()) if np.any(reachable) else 1
    return np.where(reachable, 1.0 - dist / max(max_d, 1), 0.0).astype(np.float32)


def compute_frontier(walls, visited):
    H, W = walls.shape
    has_visited_neighbor = np.zeros((H, W), dtype=bool)
    unvisited_open = (visited == 0) & (walls == 0)

    for dr, dc in _NEIGHBOR_OFFSETS:
        shifted = np.roll(np.roll(visited, -dr, axis=0), -dc, axis=1)
        if dr == -1:
            shifted[-1, :] = 0
        elif dr == 1:
            shifted[0, :] = 0
        if dc == -1:
            shifted[:, -1] = 0
        elif dc == 1:
            shifted[:, 0] = 0
        has_visited_neighbor |= (shifted > 0)

    return (has_visited_neighbor & unvisited_open).astype(np.float32)


def observation_space(env: gym.Env) -> gym.spaces.Space:
    H = env.grid_size
    W = env.grid_size
    return gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(6, H, W),
        dtype=np.float32
    )


def observation(grid: np.ndarray):
    H, W, _ = grid.shape
    cell_ids = _rgb_to_cell_ids(grid)

    walls       = (cell_ids == 2).astype(np.float32)
    visited     = np.isin(cell_ids, [1, 3, 6]).astype(np.float32)
    agent       = (cell_ids == 3).astype(np.float32)
    unexplored  = ((cell_ids == 0) | (cell_ids == 5)).astype(np.float32)
    frontier    = compute_frontier(walls, visited)
    enemy_pos   = (cell_ids == 4).astype(np.float32)
    frontier_dist = _compute_frontier_distance(walls, enemy_pos, visited, H, W)
    '''
    danger_t0   = ((cell_ids == 5) | (cell_ids == 6)).astype(np.float32)
    danger_t1, danger_t2, danger_t3 = _compute_future_dangers(cell_ids, H, W)

    obs = np.stack([
        walls,
        visited,
        agent,
        unexplored,
        frontier,
        enemy_pos,
        danger_t0,
        danger_t1,
        danger_t2,
        danger_t3,
        frontier_dist,
    ], axis=0)
    '''
    obs = np.stack([
        walls,
        visited,
        agent,
        unexplored,
        frontier,
        frontier_dist
    ], axis=0)
    return obs.astype(np.float32)


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
    if info["game_over"]:
        return -10.0

    if info["cells_remaining"] == 0:
        return 10.0 + info["steps_remaining"] * 0.02

    if info["new_cell_covered"]:
        #coverage = info["total_covered_cells"] / info["coverable_cells"]
        #return 1.0 + 0.5 * coverage
        return 1

    return -0.03