import gym
import crawler_gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make('Crawler-v0', headless=False)

env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("crawler_ppo")