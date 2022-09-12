import gym

from crawler_gym.agents.crawler import Crawler
from crawler_gym.agents.wall import Wall

import pybullet as p
import math
import numpy as np
import random
from csv import writer

MAX_EPISODE_LEN = 1e3
TRACKING_SIGMA = 0.25
class CrawlerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, headless=False, push_robot=True, add_noise=False, add_bias=False, resample_cmd=True, random_orient=True):
        self.step_counter = 0

        # actions for left & right front wheel velocities, 
        # need to change to linear and angular velocities
        self.action_space = gym.spaces.Box(low=np.array([-1,-1]),high=np.array([1,1]))
        self.observation_space = gym.spaces.Box(low=np.array([-100]*4),high=np.array([100]*4))

        self.np_random, _ = gym.utils.seeding.np_random()
        self.dt = 1./20.
        self.client = p.connect(p.DIRECT if headless else p.GUI)

        self.crawler = None
        self.commands = np.zeros((2,), dtype=float)
        self.action_buffer = [] 
        self.init_done = True
        self.done = False
        self.push_robot = push_robot
        self.add_noise = add_noise
        self.add_bias = add_bias
        self.lin_vel_noise = 0.005
        self.ang_vel_noise = 0.1
        self.bias_period = 1./10.
        self.resample_cmd = resample_cmd
        self.random_orient = random_orient

    def step(self, action):
        self.actions = action
        self.action_buffer.append(list(action) + [self.crawler.get_observations()[7], self.crawler.get_observations()[12]])
        self.crawler.apply_action(self.actions)
        p.stepSimulation()
        
        if self.resample_cmd:
            if self.step_counter % 250 == 0:
                self._resample_commands()

        self.physics_step()

        reward = self.compute_reward()

        obs = self.compute_observations()

        self.step_counter += 1 

        self.prev_actions = self.actions

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
        if self.push_robot:
            force_vec = np.array(self.crawler.get_state()[0:3]) - np.array([0,0,5])
            force_vec = force_vec / np.linalg.norm(force_vec) * 5
            p.applyExternalForce(objectUniqueId=self.crawler.crawler, linkIndex=-1,
                                 forceObj=force_vec, posObj=(0,0,0), flags=p.WORLD_FRAME, physicsClientId=self.client)

    def compute_observations(self):
        obs = [] 
        lin_vel = self.crawler.get_observations()[7] + \
                  (np.random.normal(size=1, scale=self.lin_vel_noise)[0] if self.add_noise else 0) + \
                  (np.sin(self.step_counter * self.bias_period) * self.lin_vel_noise if self.add_bias else 0) 
        ang_vel = self.crawler.get_observations()[12] + \
                  (np.random.normal(size=1, scale=self.ang_vel_noise)[0] if self.add_noise else 0) + \
                  (np.sin(self.step_counter * self.bias_period) * self.ang_vel_noise if self.add_bias else 0) 
        obs += [lin_vel, ang_vel]
        obs += self.commands.tolist()
        obs = np.array(obs)
        return obs 

    def compute_reward(self):
        reward = 0
        reward += self._reward_tracking_lin_vel()
        reward += self._reward_tracking_ang_vel()
        reward += -0.5 * self._reward_action_rate()
        return reward

    def set_commands(self, lin_vel, ang_vel):
        self.commands[0] = lin_vel
        self.commands[1] = ang_vel

    def _resample_commands(self):
        # TODO: Get rid of magic numbers
        self.commands[0] = random.uniform(-0.2, 0.2) 
        self.commands[1] = random.uniform(-1.0, 1.0) 

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
    

    def reset(self):
        self.step_counter = 0
        p.resetSimulation(self.client)
        p.setGravity(0,0,-9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSubSteps=50)
        self.actions = np.array([0,0])
        self.prev_actions = self.actions
        self.commands = np.zeros((2,), dtype=float)


        Wall(self.client)
        self.crawler = Crawler(self.client, self.random_orient)

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
        # lin_vel_error = np.square(self.commands[0] - self.crawler.get_state()[7])
        # return np.exp(-lin_vel_error/TRACKING_SIGMA)
        return 1 - math.fabs((self.commands[0] - self.crawler.get_state()[7])/(self.commands[0] if self.commands[0] else 1))

    def _reward_tracking_ang_vel(self):
        # ang_vel_error = np.square(self.commands[1] - self.crawler.get_state()[12])
        # return np.exp(-ang_vel_error/TRACKING_SIGMA)
        return 1 - math.fabs((self.commands[1] - self.crawler.get_state()[12])/(self.commands[1] if self.commands[1] else 1))

    def _reward_action_rate(self):
        return np.sum(np.square(self.prev_actions - self.actions))
