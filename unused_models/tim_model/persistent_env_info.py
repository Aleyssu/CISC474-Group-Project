"""Process-global state (e.g. SubprocVecEnv: one copy per worker)."""

from collections import deque
from typing import Dict, Optional

pos_before_step: Optional[int] = None

# Same-cell streak (reward): compare consecutive agent_pos across steps (no wrapper needed).
prev_agent_pos_for_streak: Optional[int] = None
same_cell_streak: int = 0

pos_history: deque = deque(maxlen=3)

# coord -> times agent ended a step on that cell this episode (for revisit penalty).
cell_visit_counts: Dict[int, int] = {}

# Current grid snapshot stored by observation() so reward() can compute next-step FOV.
last_grid = None

enemies_in_sight = []

last_step_covered_cells = 0

# FOV cells (row, col) at the end of the previous step, used to classify deaths.
prev_fov_cell_set: set = set()

# Set to True after the first reset_episode() call within an episode so that
# observation() doesn't keep re-clearing state on every STAY step where cleared_cells == 0.
episode_reset_done: bool = False


def reset_step_penalty_context() -> None:
    """Clear episode-local pinfo (call from train wrapper reset or unwrapped eval)."""
    global pos_before_step, prev_agent_pos_for_streak, same_cell_streak, last_grid, last_step_covered_cells, episode_reset_done
    pos_before_step = None
    prev_agent_pos_for_streak = None
    same_cell_streak = 0
    last_grid = None
    pos_history.clear()
    cell_visit_counts.clear()
    enemies_in_sight.clear()
    last_step_covered_cells = 0
    prev_fov_cell_set.clear()
    episode_reset_done = False


def reset_episode() -> None:
    """Call at the start of each `env.reset()` (e.g. from a small gym Wrapper)."""
    reset_step_penalty_context()
