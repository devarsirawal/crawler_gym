import gym
import crawler_gym
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from plotter import Plotter
import argparse
from collections import defaultdict

MAX_PLOT_TIME = 1000 

parser = argparse.ArgumentParser()
parser.add_argument("--headless", help="Run in headless mode", action="store_true", default=False)
parser.add_argument("--add_noise", help="Add noise to observations", action="store_true", default=False)
parser.add_argument("--add_bias", help="Add bias to observations", action="store_true", default=False)
parser.add_argument("--random_orient", help="Start crawler with random heading", action="store_true", default=True)
args = parser.parse_args()

# Create the environment
env = gym.make('Crawler-v0', headless=args.headless, add_noise=args.add_noise, add_bias=args.add_bias, resample_cmd=False, random_orient=args.random_orient)
env.reset()
episodes = 1
model = PPO.load("crawler_ppo", env=env)

plot_params = {}
plotter = Plotter(1, plot_params)

# lin_vels = set()
# ang_vels = set()
# for l in range(-8,9):
#     for r in range(-8,9):
#         lin_vels.add(0.025/2 * (l + r))
#         ang_vels.add(0.025/0.14 * (r - l))

mean_lin_vels = defaultdict(list)
mean_ang_vels = defaultdict(list)
for lin_vel in np.arange(-0.2, 0.24, 0.08):
    for ang_vel in np.arange(-1.0, 1.0, 0.4):
        print(f"V: {lin_vel}, W: {ang_vel}")
        measured_lin_vels = []
        measured_ang_vels = []
        
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                env.set_commands(lin_vel, ang_vel)
                obs, rewards, done, info = env.step(action)
                # print(f"{env.step_counter}: {rewards}")
                measured_lin_vels.append(env.crawler.get_state()[7])
                measured_ang_vels.append(env.crawler.get_state()[12])
                
        
        mean_lin_vels[lin_vel].append(np.mean(measured_lin_vels))
        mean_ang_vels[ang_vel].append(np.mean(measured_ang_vels))

logger_vars = {
        "measured_lin_vel_mean": np.mean(list(mean_lin_vels.values()), axis=1),
        "measured_lin_vel_std": np.std(list(mean_lin_vels.values()), axis=1),
        "measured_ang_vel_mean": np.mean(list(mean_ang_vels.values()), axis=1),
        "measured_ang_vel_std": np.std(list(mean_ang_vels.values()), axis=1),
        "target_lin_vel": list(mean_lin_vels.keys()),
        "target_ang_vel": list(mean_ang_vels.keys())
        }
plotter.dump_states(logger_vars)
        
plotter.plot_eval()
env.close()
