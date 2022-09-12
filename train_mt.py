import gym
import crawler_gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", help="Run in headless mode", action="store_true", default=False)
parser.add_argument("--add_noise", help="Add noise to observations", action="store_true", default=False)
parser.add_argument("--add_bias", help="Add bias to observations", action="store_true", default=False)
parser.add_argument("--random_orient", help="Start crawler with random heading", action="store_true", default=False)
args = parser.parse_args()

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, headless=True, add_noise=args.add_noise, add_bias=args.add_bias, random_orient=args.random_orient)
        env.seed(seed+rank)
        env.reset()
        return env
    set_random_seed(seed)
    return _init

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env
        self.actions_buffer = []
        self.lin_vel_buffer = []
        self.ang_vel_buffer = []
        self.command_buffer = []

    def _on_step(self) -> bool:
        self.actions_buffer.append(list(self.env.actions))
        self.lin_vel_buffer.append(self.env.crawler.get_observations()[7])
        self.ang_vel_buffer.append(self.env.crawler.get_observations()[12])
        self.command_buffer.append(self.env.commands)
        self.logger.record('vels/cmd_vel', self.env.commands[0])
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('actions/l_wheel_avg', np.mean(np.asarray(self.actions_buffer)[:,0]))
        self.logger.record('actions/r_wheel_avg', np.mean(np.asarray(self.actions_buffer)[:,1]))
        lin_vel_avg = np.mean(self.lin_vel_buffer)
        ang_vel_avg = np.mean(self.ang_vel_buffer)
        self.logger.record('vels/lin_vel_avg', lin_vel_avg)
        self.logger.record('vels/ang_vel_avg', ang_vel_avg)
        self.logger.record('vels/rmse_lin_vel', np.sqrt(np.mean((np.array(self.lin_vel_buffer)-np.array(self.command_buffer)[:,0])**2)))
        self.logger.record('vels/rmse_ang_vel', np.sqrt(np.mean((np.array(self.ang_vel_buffer)-np.array(self.command_buffer)[:,1])**2)))
        self.actions_buffer = []
        self.lin_vel_buffer = []
        self.ang_vel_buffer = []
        self.command_buffer = []

if __name__ == "__main__":
    env_id = "Crawler-v0" 
    num_cpu = 16 
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])], log_std_init=0.1)
    policy_kwargs = dict() 
    model = PPO('MlpPolicy', env, use_sde=False, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_crawler_tensorboard")
    model.learn(total_timesteps=1_000_000) # , callback=[TensorboardCallback(env),
                                           #         CheckpointCallback(save_freq=100_000, save_path="./logs/", name_prefix="crawler")
                                           #        ])
    model.save("crawler_ppo")

    eval_env = gym.make(env_id, headless=True, add_noise=args.add_noise, add_bias=args.add_bias, random_orient=args.random_orient)
    reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=20) 

    print(f"Evalution Reward: {reward}")
