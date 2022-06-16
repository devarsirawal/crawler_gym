import gym
import crawler_gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make('Crawler-v0')

env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50_000)
model.save("crawler_ppo")