import random

import numpy as np
import gymnasium
import gymnasium as gym
import coverage_gridworld
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

N_ENVS = 8


# ---------------------------------------------------------------------------
# Wrapper for procedural map generation
# ---------------------------------------------------------------------------
class RandomizedMapWrapper(gym.Wrapper):
    """Randomizes wall/enemy counts every reset for infinite map diversity.

    The underlying CoverageGridworld already generates random layouts when
    no predefined map is provided.  This wrapper varies the *number* of
    walls and enemies each episode so the agent cannot memorize any
    particular configuration.
    """

    def __init__(self, env, min_walls, max_walls, min_enemies, max_enemies,
                 seed=None):
        super().__init__(env)
        self.min_walls = min_walls
        self.max_walls = max_walls
        self.min_enemies = min_enemies
        self.max_enemies = max_enemies
        self._rng = np.random.RandomState(seed)

    def reset(self, **kwargs):
        uw = self.env.unwrapped
        uw.num_walls = int(self._rng.randint(self.min_walls, self.max_walls + 1))
        uw.num_enemies = int(self._rng.randint(self.min_enemies, self.max_enemies + 1))
        return self.env.reset(**kwargs)


# ---------------------------------------------------------------------------
# Fixed evaluation maps (handcrafted, consistent benchmarks)
# ---------------------------------------------------------------------------
EVAL_MAPS_BASIC = [
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
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
]

EVAL_MAPS_MAZE = [
    # "safe" from __init__.py
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
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    ],
]

EVAL_MAPS_ENEMIES = [
    # "maze" with enemies
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
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    ],
]

EVAL_MAPS_HARD = [
    # "chokepoint"
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
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    ],
    # "sneaky_enemies"
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
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
    ],
]

HELD_OUT_MAPS = [
    # unseen maze, no enemies
    [
        [3, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 2, 0, 0],
        [0, 2, 0, 2, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 2, 2, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 0, 0, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    ],
    # unseen maze + 2 enemies
    [
        [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 2, 0, 2, 2, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 0, 2, 0, 2, 0, 4, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # unseen hard maze + 4 enemies
    [
        [3, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 4, 0, 2, 0, 2, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 2, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [2, 2, 0, 0, 0, 4, 0, 0, 2, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 2, 0, 4, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    ],
]


# ---------------------------------------------------------------------------
# Curriculum stages — procedural training + benchmark maps
# ---------------------------------------------------------------------------
def _resolve_map_lists(stage):
    """`benchmark_maps` feeds both eval & fixed training (same pool).

    Override if you want stricter generalization:
      eval_maps=HELD_OUT_ONLY, fixed_train_maps=BENCHMARKS+EXTRA
    """
    if "benchmark_maps" in stage:
        m = stage["benchmark_maps"]
        return stage.get("eval_maps", m), stage.get("fixed_train_maps", m)
    return stage["eval_maps"], stage.get("fixed_train_maps")


ALL_STAGES = [
    {
        "name": "stage1_basic",
        "walls_range": (0, 0),
        "enemies_range": (0, 0),
        "benchmark_maps": EVAL_MAPS_BASIC,
        "timesteps": 500_000,
        "lr": 3e-4,
        "ent_coef": 0.1,
        "reward_threshold": 110.0,
    },
    {
        "name": "stage2_mazes",
        "walls_range": (5, 20),
        "enemies_range": (0, 0),
        # Must include "safe" maze; random walls alone never teach that topology.
        "benchmark_maps": EVAL_MAPS_BASIC + EVAL_MAPS_MAZE,
        "timesteps": 2_000_000,
        "lr": 3e-4,
        "ent_coef": 0.1,
        #"reward_threshold": 85.0,
    },

]
'''
{
    "name": "stage3_enemies",
    "walls_range": (8, 20),
    "enemies_range": (1, 2),
    "benchmark_maps": EVAL_MAPS_MAZE + EVAL_MAPS_ENEMIES,
    "timesteps": 3_000_000,
    "lr": 2e-4,
    "ent_coef": 0.05,
    "reward_threshold": 55.0,
},
{
    "name": "stage4_hard",
    "walls_range": (8, 20),
    "enemies_range": (3, 5),
    "benchmark_maps": EVAL_MAPS_ENEMIES + EVAL_MAPS_HARD,
    "timesteps": 5_000_000,
    "lr": 1e-4,
    "ent_coef": 0.04,
    "reward_threshold": None,
},
'''


# ---------------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------------
class GridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_in = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def make_env(rank, min_walls, max_walls, min_enemies, max_enemies, seed=0):
    """Create a single env with randomized map parameters per episode."""
    def _init():
        env = gymnasium.make("standard")
        env = RandomizedMapWrapper(
            env, min_walls, max_walls, min_enemies, max_enemies,
            seed=seed + rank,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def make_env_fixed_maps(rank, map_list, seed=0):
    """Cycle handcrafted / fixed maps (matches eval topology, teaches backtracking)."""
    def _init():
        env = gymnasium.make("standard", predefined_map_list=map_list)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_envs(stage):
    """Train envs: mix procedural + fixed maps when a benchmark pool is set.

    Random wall placement alone does not teach the *specific* corridor topology
    used in eval (e.g. \"safe\" maze). Without fixed maps in the training
    mixture, eval reward stays near random-walk levels no matter how many
    timesteps you spend.
    """
    min_w, max_w = stage["walls_range"]
    min_e, max_e = stage["enemies_range"]
    eval_maps, fixed_pool = _resolve_map_lists(stage)

    factories = []
    for i in range(N_ENVS):
        if fixed_pool is not None and (i % 2 == 0):
            factories.append(make_env_fixed_maps(i, fixed_pool))
        else:
            factories.append(make_env(i, min_w, max_w, min_e, max_e))

    train_env = SubprocVecEnv(factories)

    eval_raw = Monitor(
        gymnasium.make("standard", predefined_map_list=eval_maps)
    )
    eval_env = DummyVecEnv([lambda: eval_raw])

    return train_env, eval_env


# ---------------------------------------------------------------------------
# Generic stage trainer (PPO + early stopping + checkpointing)
# ---------------------------------------------------------------------------
def train_stage(stage, prev_model_path=None):
    name = stage["name"]
    timesteps = stage["timesteps"]
    lr = stage["lr"]
    ent_coef = stage["ent_coef"]
    reward_threshold = stage.get("reward_threshold")

    wr = stage["walls_range"]
    er = stage["enemies_range"]
    print("=" * 60)
    print(f"  {name.upper()} — walls={wr}, enemies={er}, {timesteps:,} steps")
    if reward_threshold is not None:
        print(f"  Early-stop threshold: {reward_threshold}")
    print("=" * 60)

    train_env, eval_env = create_envs(stage)

    save_dir = f"./models/{name}"

    cb_list = []

    if reward_threshold is not None:
        stop_cb = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold, verbose=1,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=max(10_000 // N_ENVS, 1),
            n_eval_episodes=20,
            deterministic=True,
            callback_on_new_best=stop_cb,
        )
    else:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            eval_freq=max(10_000 // N_ENVS, 1),
            n_eval_episodes=20,
            deterministic=True,
        )

    cb_list.append(eval_cb)

    ckpt_cb = CheckpointCallback(
        save_freq=max(500_000 // N_ENVS, 1),
        save_path=save_dir,
        name_prefix=name,
    )
    cb_list.append(ckpt_cb)

    callbacks = CallbackList(cb_list)

    policy_kwargs = dict(
        features_extractor_class=GridCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )

    tb_log = "./tensorboard_logs"

    if prev_model_path is not None:
        model = PPO.load(
            prev_model_path, env=train_env, tensorboard_log=tb_log,
        )
        model.learning_rate = lr
        model.ent_coef = ent_coef
        model.gamma = 0.99
    else:
        model = PPO(
            "CnnPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=lr,
            n_steps=512,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=tb_log,
        )

    model.learn(
        total_timesteps=timesteps, callback=callbacks, tb_log_name=name,
    )

    final_path = f"./models/{name}/final"
    model.save(final_path)
    train_env.close()
    eval_env.close()

    best_path = f"{save_dir}/best_model"
    print(f"  {name} complete → best: {best_path}, final: {final_path}")
    return best_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, map_list, num_episodes=1, render=True):
    render_mode = "human" if render else None
    raw_env = gymnasium.make(
        "standard", predefined_map_list=map_list, render_mode=render_mode,
    )
    vec_env = DummyVecEnv([lambda: raw_env])

    for ep in range(num_episodes):
        obs = vec_env.reset()
        total_reward = 0.0
        done = False
        last_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            last_info = infos[0]
            done = dones[0]
        covered = last_info.get("total_covered_cells", 0)
        coverable = last_info.get("coverable_cells", 0)
        game_over = last_info.get("game_over", False)
        status = "DIED" if game_over else (
            "FULL" if covered == coverable else "TIMEOUT"
        )
        print(f"  Episode {ep + 1}: reward={total_reward:.1f}  "
              f"coverage={covered}/{coverable}  [{status}]")

    vec_env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    '''
    prev_path = None
    for stage in ALL_STAGES:
        prev_path = train_stage(stage, prev_model_path=prev_path)
    '''
    #trained = PPO.load(prev_path)
    trained = PPO.load("models/stage2_mazes/stage2_mazes_1500000_steps.zip")

    print("\n=== FINAL EVALUATION (hard maps) ===")
    evaluate(trained, EVAL_MAPS_MAZE)
    
    print("\n=== GENERALIZATION TEST (held-out unseen maps) ===")
    evaluate(trained, HELD_OUT_MAPS)
    