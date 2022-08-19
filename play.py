import gym
import crawler_gym
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from plotter import Plotter
import argparse

MAX_PLOT_TIME = 1000 

parser = argparse.ArgumentParser()
parser.add_argument("--headless", help="Run in headless mode", action="store_true", default=False)
args = parser.parse_args()

# Create the environment
env = gym.make('Crawler-v0', headless=args.headless)
env.reset()
episodes = 1
model = PPO.load("crawler_ppo", env=env)

plot_params = {}
plotter = Plotter(1, plot_params)

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(f"{env.step_counter}: {rewards}")
        if env.step_counter < MAX_PLOT_TIME:
            logger_vars = {
                'x_pos': env.crawler.get_state()[2],
                'y_pos': env.crawler.get_state()[1],
                'track_lin_vel': env.crawler.get_state()[7],
                'track_ang_vel': env.crawler.get_state()[12],
                'cmd_lin_vel': env.commands[0],
                'cmd_ang_vel': env.commands[1],
                'l_wheel': env.actions[0],
                'r_wheel': env.actions[1]
            }
            plotter.log_states(logger_vars)
        elif env.step_counter == MAX_PLOT_TIME:
            plotter.plot_states()
env.close()
