import gym

from crawler_gym.agents.crawler import Crawler
from crawler_gym.agents.wall import Wall

import pybullet as p
import math
import numpy as np
import random

MAX_EPISODE_LEN = 1e3
TRACKING_SIGMA = 0.25
class CrawlerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, headless=False):
        self.step_counter = 0

        # actions for left & right front wheel velocities, 
        # need to change to linear and angular velocities
        self.action_space = gym.spaces.Box(low=np.array([-1,-1]),high=np.array([1,1]))
        self.observation_space = gym.spaces.Box(low=np.array([-100]*10),high=np.array([100]*10))

        self.np_random, _ = gym.utils.seeding.np_random()
        self.dt = 1./20.
        self.client = p.connect(p.DIRECT if headless else p.GUI)

        self.crawler = None
        self.commands = np.zeros((2,), dtype=float)
        
        self.init_done = True
        self.done = False



    def step(self, action):
        self.actions = action
        self.crawler.apply_action(self.actions)
        p.stepSimulation()

        if self.step_counter % 250 == 0:
            self._resample_commands()

        self.physics_step()

        reward = self.compute_reward()

        obs = self.compute_observations()

        self.step_counter += 1 

        if self.step_counter > MAX_EPISODE_LEN:
            self.done = True

        info = {}
        return obs, reward, self.done, info 

    def physics_step(self):
        p.applyExternalForce(objectUniqueId=self.crawler.crawler, linkIndex=-1,
                         forceObj=[0,0,-200], posObj=self.crawler.lw_pos, flags=p.LINK_FRAME, physicsClientId=self.client)
        p.applyExternalForce(objectUniqueId=self.crawler.crawler, linkIndex=-1,
                         forceObj=[0,0,-200], posObj=self.crawler.rw_pos, flags=p.LINK_FRAME, physicsClientId=self.client)
        p.applyExternalForce(objectUniqueId=self.crawler.crawler, linkIndex=-1,
                         forceObj=[0,0,-200], posObj=self.crawler.cw_pos, flags=p.LINK_FRAME, physicsClientId=self.client)

    def compute_observations(self):
        obs = self.crawler.get_observations()[7:]
        obs += self.actions.tolist()
        obs += self.commands.tolist()
        obs = np.array(obs)
        return obs 

    def compute_reward(self):
        reward = 0
        reward += self._reward_tracking_lin_vel()
        reward += self._reward_tracking_ang_vel()
        return reward

    def _resample_commands(self):
        # self.commands[0] = random.uniform(0,1)
        self.commands[0] = 0 
        self.commands[1] = 1

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
    

    def reset(self):
        self.step_counter = 0
        p.resetSimulation(self.client)
        p.setGravity(0,0,-9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSubSteps=10)
        self.actions = np.array([0,0])
        self.commands = np.zeros((2,), dtype=float)


        Wall(self.client)
        self.crawler = Crawler(self.client)

        obs = self.compute_observations() 

        self.done = False

        return obs

    
    def render(self, mode="human"):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    

    def close(self):
        p.disconnect(self.client)
    

    def _reward_tracking_lin_vel(self):
        lin_vel_error = np.square(self.commands[0] - self.crawler.get_state()[7])
        return np.exp(-lin_vel_error/TRACKING_SIGMA)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = np.square(self.commands[1] - self.crawler.get_state()[12])
        return np.exp(-ang_vel_error/TRACKING_SIGMA)
    
