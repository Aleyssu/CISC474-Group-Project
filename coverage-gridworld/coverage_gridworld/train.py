import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback
import coverage_gridworld
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

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
    ]
]

test_map = [
        [3, 0, 0, 0, 0, 4, 2, 0, 0, 0],
        [0, 0, 0, 0, 4, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 4, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 2, 0, 0, 0, 0],
        [4, 4, 4, 0, 4, 2, 0, 4, 2, 0],
        [2, 2, 2, 0, 2, 2, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    ],

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

# Training - default exploration rate is 0.05
# env = gym.make("standard", predefined_map_list=maps[0:5])
# model = DQN("MlpPolicy", env, verbose=1, device="cuda", learning_rate=0.001)
# eval_callback = EvalCallback(env, 
#                              best_model_save_path='./best_model',
#                              log_path='./logs/', 
#                              eval_freq=500, 
#                              verbose=1)
# model.learn(total_timesteps=20_000, progress_bar=True, callback=eval_callback)

# # Make model always pick greediest action after training
# model.exploration_rate = 0
# model.save("test_model", include=["exploration_rate"])


model = DQN.load("../../models/final_model")
model.exploration_rate = 0

# Testing
env = gym.make("standard", render_mode="human", predefined_map_list=maps[4:5], activate_game_status=True)
num_episodes = 10
for i in range(num_episodes):
    done = False
    obs, _ = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action taken: {action}, Reward received: {reward}")

        # Sleep may be used to allow each step to be visualized. Value can be changed
        print(obs)
        time.sleep(0.04)
    if done:
        time.sleep(2)
env.close()

