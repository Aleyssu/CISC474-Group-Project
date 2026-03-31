import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import coverage_gridworld
import persistent_env_info as pinfo
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

free = [

]

easy = [
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 2, 2],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 2, 0, 0, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    ],
    [
        [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    ],
]

medium = [
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 0, 2, 2],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 2, 0, 2, 0, 0],
        [4, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 4],
        [0, 0, 0, 2, 2, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 2, 4, 0, 2, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],

]

hard = [
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4]
    ],
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 4, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    ],
]

insane = [
    [
        [3, 0, 4, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 4, 0, 4, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 4, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 4, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 4],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 2, 0, 0, 0, 0, 0],
        [0, 4, 0, 4, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 2, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 2],
        [0, 2, 0, 0, 0, 0, 2, 0, 0, 0]
    ],
    [
        [3, 0, 4, 0, 4, 0, 4, 0, 4, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 4, 0, 4, 0, 4, 0],
        [0, 0, 0, 4, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
        [4, 0, 4, 0, 4, 0, 4, 0, 0, 0]
    ]
]

class GridCNN(BaseFeaturesExtractor):
    """
    CNN for the 9-channel 7×7 egocentric patch.

    Layer 1: 3×3 conv — local features, keeps 7×7.
    Layer 2: 3×3 conv stride 2 — reduces to 3×3, preserving directional layout.
    The 3×3 spatial grid maps to 8 compass directions + centre, so the
    downstream linear can learn "unexplored cells are to the east."
    """

    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0] + 2  # +2 for CoordConv row/col channels
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, kernel_size=3, stride=1, padding=1),   # 7×7 → 7×7
            nn.ReLU(),
            nn.Flatten(),                                                # → 32*7*7 = 1568
        )
        self.linear = nn.Sequential(
            nn.Linear(1568, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        B, C, H, W = observations.shape
        # CoordConv: append row/col coordinate channels (centered at agent, range -1..+1)
        rows = th.linspace(-1, 1, H, device=observations.device).view(1, 1, H, 1).expand(B, 1, H, W)
        cols = th.linspace(-1, 1, W, device=observations.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = th.cat([observations, rows, cols], dim=1)  # (B, C+2, H, W)
        return self.linear(self.cnn(x))


class PatchAndDirExtractor(BaseFeaturesExtractor):
    """For ``MultiInputPolicy``: GridCNN on ``patch`` + small MLP on ``vector``, then fuse."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        patch_dim = 64
        vec_dim = 64
        self.patch_net = GridCNN(observation_space["patch"], features_dim=patch_dim)
        self.vec_mlp = nn.Sequential(
            nn.Linear(2, vec_dim),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(patch_dim + vec_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        x_p = self.patch_net(observations["patch"].float())
        x_v = self.vec_mlp(observations["vector"].float())
        return self.fuse(th.cat([x_p, x_v], dim=1))


class SafeEvalCallback(EvalCallback):
    """EvalCallback that isolates pinfo state so eval doesn't corrupt training."""

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        snap = {
            'pos_before_step': pinfo.pos_before_step,
            'prev_agent_pos_for_streak': pinfo.prev_agent_pos_for_streak,
            'same_cell_streak': pinfo.same_cell_streak,
            'pos_history': list(pinfo.pos_history),
            'cell_visit_counts': dict(pinfo.cell_visit_counts),
            'last_grid': pinfo.last_grid.copy() if pinfo.last_grid is not None else None,
            'enemies_in_sight': list(pinfo.enemies_in_sight),
            'last_step_covered_cells': pinfo.last_step_covered_cells,
            'prev_fov_cell_set': set(pinfo.prev_fov_cell_set),
            'episode_reset_done': pinfo.episode_reset_done,
        }
        pinfo.reset_step_penalty_context()
        result = super()._on_step()
        pinfo.pos_before_step = snap['pos_before_step']
        pinfo.prev_agent_pos_for_streak = snap['prev_agent_pos_for_streak']
        pinfo.same_cell_streak = snap['same_cell_streak']
        pinfo.pos_history.clear()
        pinfo.pos_history.extend(snap['pos_history'])
        pinfo.cell_visit_counts.clear()
        pinfo.cell_visit_counts.update(snap['cell_visit_counts'])
        pinfo.last_grid = snap['last_grid']
        pinfo.enemies_in_sight.clear()
        pinfo.enemies_in_sight.extend(snap['enemies_in_sight'])
        pinfo.last_step_covered_cells = snap['last_step_covered_cells']
        pinfo.prev_fov_cell_set.clear()
        pinfo.prev_fov_cell_set.update(snap['prev_fov_cell_set'])
        pinfo.episode_reset_done = snap['episode_reset_done']
        return result


class CurriculumCallback(BaseCallback):
    """
    Advances curriculum stages based on eval performance.

    Stages progress easy → medium → hard.
    Each stage has a mean-reward threshold; once the agent clears it the
    training and eval envs are updated to the next map list.
    The best model seen across all stages is saved to `save_path`.

    Thresholds to tune:
      - Too low  → advance before the agent has mastered the stage (forgetting).
      - Too high → agent stalls on a stage it already handles well.
    Start conservative and lower if training gets stuck.
    """

    def __init__(self, train_env, eval_env, stages, thresholds,
                 eval_freq=20_000, n_eval_episodes=12,
                 save_path="./best_model", verbose=1):
        super().__init__(verbose)
        # Keep references to the unwrapped envs for map-list updates.
        self._train_raw = train_env.unwrapped
        self._eval_env  = eval_env          # wrapped — passed to evaluate_policy
        self._eval_raw  = eval_env.unwrapped
        self.stages     = stages            # list of (name, map_list)
        self.thresholds = thresholds        # reward threshold per stage; None = final
        self.eval_freq  = eval_freq
        self.n_eval     = n_eval_episodes
        self.save_path  = save_path
        self.stage_idx  = 0
        self.best_mean  = -np.inf
        os.makedirs(save_path, exist_ok=True)
        self._apply_stage(0)

    def _apply_stage(self, idx):
        name, maps = self.stages[idx]
        # Eval env: current stage only — keeps the threshold signal clean.
        self._eval_raw.predefined_map_list    = maps
        self._eval_raw.predefined_map         = maps[0]
        self._eval_raw.current_predefined_map = 0
        # Training env: accumulate all stages seen so far to prevent forgetting.
        train_maps = []
        for i in range(idx + 1):
            train_maps.extend(self.stages[i][1])
        self._train_raw.predefined_map_list    = train_maps
        self._train_raw.predefined_map         = train_maps[0]
        self._train_raw.current_predefined_map = 0
        if self.verbose:
            thresh = self.thresholds[idx]
            t_str = f"  threshold ≥ {thresh}" if thresh is not None else "  (final stage)"
            counts = " + ".join(f"{len(self.stages[i][1])} {self.stages[i][0]}" for i in range(idx + 1))
            print(f"\n[Curriculum] Stage {idx}: '{name}'{t_str}")
            print(f"[Curriculum] Training pool: {counts} = {len(train_maps)} maps")

    @staticmethod
    def _snapshot_pinfo():
        return {
            'pos_before_step': pinfo.pos_before_step,
            'prev_agent_pos_for_streak': pinfo.prev_agent_pos_for_streak,
            'same_cell_streak': pinfo.same_cell_streak,
            'pos_history': list(pinfo.pos_history),
            'cell_visit_counts': dict(pinfo.cell_visit_counts),
            'last_grid': pinfo.last_grid.copy() if pinfo.last_grid is not None else None,
            'enemies_in_sight': list(pinfo.enemies_in_sight),
            'last_step_covered_cells': pinfo.last_step_covered_cells,
            'prev_fov_cell_set': set(pinfo.prev_fov_cell_set),
            'episode_reset_done': pinfo.episode_reset_done,
        }

    @staticmethod
    def _restore_pinfo(snap):
        pinfo.pos_before_step = snap['pos_before_step']
        pinfo.prev_agent_pos_for_streak = snap['prev_agent_pos_for_streak']
        pinfo.same_cell_streak = snap['same_cell_streak']
        pinfo.pos_history.clear()
        pinfo.pos_history.extend(snap['pos_history'])
        pinfo.cell_visit_counts.clear()
        pinfo.cell_visit_counts.update(snap['cell_visit_counts'])
        pinfo.last_grid = snap['last_grid']
        pinfo.enemies_in_sight.clear()
        pinfo.enemies_in_sight.extend(snap['enemies_in_sight'])
        pinfo.last_step_covered_cells = snap['last_step_covered_cells']
        pinfo.prev_fov_cell_set.clear()
        pinfo.prev_fov_cell_set.update(snap['prev_fov_cell_set'])
        pinfo.episode_reset_done = snap['episode_reset_done']

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Isolate eval from training: save training pinfo, reset for eval, restore after.
        snap = self._snapshot_pinfo()
        pinfo.reset_step_penalty_context()

        mean_r, std_r = evaluate_policy(
            self.model, self._eval_env,
            n_eval_episodes=self.n_eval,
            deterministic=True,
        )

        self._restore_pinfo(snap)

        name   = self.stages[self.stage_idx][0]
        thresh = self.thresholds[self.stage_idx]
        if self.verbose:
            t_str = f" | need ≥ {thresh:.0f}" if thresh is not None else ""
            print(f"[Curriculum] '{name}' | mean={mean_r:.1f} ± {std_r:.1f}{t_str}")

        if mean_r > self.best_mean:
            self.best_mean = mean_r
            self.model.save(f"{self.save_path}/{name}_best_model")
            if self.verbose:
                print(f"[Curriculum] Best model saved ({mean_r:.1f})")

        if (thresh is not None
                and mean_r >= thresh
                and self.stage_idx < len(self.stages) - 1):
            self.stage_idx += 1
            self._apply_stage(self.stage_idx)

        return True


if __name__ == "__main__":
    prev_model_path = None
    #prev_model_path = "best_model/best_model"

    # Curriculum: easy → medium → hard.
    # Thresholds are mean episode reward on the current stage's eval env.
    # Lower a threshold if the agent stalls; raise it to demand more mastery.
    '''
    stages = [
        ("easy",   easy),
        ("medium", medium),
        ("hard",   hard),
    ]
    thresholds = [100, 75, None]
    
    train_env = gym.make("standard", predefined_map_list=easy)
    eval_env  = gym.make("standard", predefined_map_list=easy)

    policy_kwargs = dict(
        features_extractor_class=PatchAndDirExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )
    if prev_model_path:
        model = PPO.load(prev_model_path, env=train_env, verbose=1, device="cpu")
    else:
        model = PPO(
            "MultiInputPolicy",
            train_env,
            #policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu",
        )

    curriculum_cb = CurriculumCallback(
        train_env=train_env,
        eval_env=eval_env,
        stages=stages,
        thresholds=thresholds,
        eval_freq=10_000,
        n_eval_episodes=12,
        save_path="./best_model",
        verbose=1,
    )

    model.learn(total_timesteps=500_000, progress_bar=True, callback=curriculum_cb)
    model.save("test_model")
    '''
    '''
    env = gym.make("standard", predefined_map_list=free+easy+medium+hard)
    
    policy_kwargs = dict(
        features_extractor_class=PatchAndDirExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )
    
    if prev_model_path:
        model = PPO.load(prev_model_path, env=env, verbose=1, device="cpu")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            #policy_kwargs=policy_kwargs,
            verbose=1,
            device="cpu",
        )

    eval_env = gym.make("standard", predefined_map_list=free+easy+medium+hard)
    eval_callback = SafeEvalCallback(eval_env, best_model_save_path="./best_model", eval_freq=10000, verbose=1)
    model.learn(total_timesteps=500_000, progress_bar=True, callback=eval_callback)
    '''
    model = PPO.load("test_model.zip")
    # test direction
    #dir_to_action = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3, (0, 0): 4}
    render_env = gym.make("standard", render_mode="human", predefined_map_list=medium+hard, activate_game_status=True)
    for ep in range(len(medium+hard)):
        obs, _ = render_env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            #dr, dc = obs["vector"][0], obs["vector"][1]
            #action = dir_to_action.get((int(dr), int(dc)), 4)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = render_env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.1)
        print(f"Episode {ep+1}: steps={steps} reward={total_reward:.2f} covered={info['total_covered_cells']}/{info['coverable_cells']}")
        time.sleep(2)
    render_env.close()

    