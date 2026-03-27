import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
import coverage_gridworld
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

maps = [
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
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
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
]

extras = [
]
'''
class GridCNN(BaseFeaturesExtractor):
    """CNN trunk for the egocentric ``patch`` only (channel-first ``(C, H, W)``)."""

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_in = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
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


class PatchAndDirExtractor(BaseFeaturesExtractor):
    """For ``MultiInputPolicy``: ``GridCNN`` on ``patch`` + small MLP on ``vector``, then fuse."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        patch_dim = features_dim - 16
        self.patch_net = GridCNN(observation_space["patch"], features_dim=patch_dim)
        self.vec_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(patch_dim + 16, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        x_p = self.patch_net(observations["patch"].float())
        x_v = self.vec_mlp(observations["vector"].float())
        return self.fuse(th.cat([x_p, x_v], dim=1))
'''

if __name__ == "__main__":
    # Training - default exploration rate is 0.05
    prev_model_path = None
    #prev_model_path = "best_model/best_model"
    """
    env = gym.make("standard", predefined_map_list=maps[0:6])
    '''
    policy_kwargs = dict(
        features_extractor_class=PatchAndDirExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )
    '''
    if prev_model_path:
        model = PPO.load(prev_model_path, env=env, verbose=1, device="cpu")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            #policy_kwargs=policy_kwargs,
            #gamma=0.999,
            verbose=1,
            device="cpu",
        )

    eval_callback = EvalCallback(env, best_model_save_path="./best_model", eval_freq=5000, verbose=1)
    model.learn(total_timesteps=400_000, progress_bar=True, callback=eval_callback)

    # model.exploration_rate = 0
    model.save("test_model")
    """
    model = PPO.load("best_model/best_model")

    # Testing
    env = gym.make("standard", render_mode="human", predefined_map_list=maps[0:6], activate_game_status=True)
    num_episodes = 6
    for i in range(num_episodes):
        done = False
        obs, _ = env.reset()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action taken: {action}, Reward received: {reward}")

            time.sleep(0.1)
        if done:
            time.sleep(2)
    env.close()