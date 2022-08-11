import gym
import crawler_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", help="Run in headless mode", action="store_true", default=False)
args = parser.parse_args()
# Create the environment
env = gym.make('Crawler-v0', headless=args.headless)

env.reset()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_crawler_tensorboard")
model.learn(total_timesteps=200_000, callback=TensorboardCallback())
model.save("crawler_ppo")
