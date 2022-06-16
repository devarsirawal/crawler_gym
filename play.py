import gym
import crawler_gym
import numpy as np
from stable_baselines3 import PPO

# Create the environment
env = gym.make('Crawler-v0')
env.reset()
episodes = 1
model = PPO.load("crawler_ppo", env=env)

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(f"{env.step_counter}: {rewards}")

env.close()
