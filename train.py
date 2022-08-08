import gym
import crawler_gym
from stable_baselines3 import PPO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", help="Run in headless mode", action="store_true", default=False)
args = parser.parse_args()
# Create the environment
env = gym.make('Crawler-v0', headless=args.headless)

env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_crawler_tensorboard")
model.learn(total_timesteps=100_000)
model.save("crawler_ppo")