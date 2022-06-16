import gym
import crawler_gym
import numpy as np
from stable_baselines3 import PPO
import argparse

# parser = argparse.ArgumentParser()
# parser.add
# Create the environment
env = gym.make('Crawler-v0')

env.reset()

episodes = 1

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50_000)
model.save("crawler_ppo")
model = PPO.load("crawler_ppo")

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(f"{env.step_counter}: {rewards}")

env.close()
