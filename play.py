import gym
import crawler_gym
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from plotter import Plotter


MAX_PLOT_TIME = 1000
# Create the environment
env = gym.make('Crawler-v0')
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
                'track_lin_vel': env.crawler.get_state()[9],
                'track_ang_vel': env.crawler.get_state()[10],
                'cmd_lin_vel': action[0],
                'cmd_ang_vel': action[1]
            }
env.close()
