import pybullet as p
import os
from math import pi
import math
import random
from crawler_gym.utils import inverse_rotation, quaternion_multiply

class Crawler:
    def __init__(self, client, random_orient=True):
        start_pos = [0.01,0,0]
        heading = random.uniform(-pi/2, pi) if random_orient else 0  
        start_orient = quaternion_multiply(p.getQuaternionFromEuler([0,-pi/2.,pi]), p.getQuaternionFromEuler([0, 0, heading]))
        
        f_name = os.path.join(os.path.dirname(__file__), "resources/crawler/crawler_caster.urdf")
        self.crawler = p.loadURDF(f_name, 
                                  start_pos,
                                  start_orient,
                                  physicsClientId=client)
        self.action_scale = 12 
        self.lw_pos = p.getJointInfo(self.crawler, 1)[14]
        self.rw_pos = p.getJointInfo(self.crawler, 2)[14]
        self.cw_pos = p.getJointInfo(self.crawler, 4)[14]
        p.setJointMotorControl2(self.crawler, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=client)
        p.setJointMotorControl2(self.crawler, 2, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=client)
        p.setJointMotorControl2(self.crawler, 3, p.POSITION_CONTROL, targetPosition=0, force=0, physicsClientId=client)
        p.setJointMotorControl2(self.crawler, 4, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=client)
        self.client = client

    def apply_action(self, action):
        p.setJointMotorControl2(self.crawler, 1, p.VELOCITY_CONTROL, targetVelocity=action[0]*self.action_scale, force=1, physicsClientId=self.client)
        p.setJointMotorControl2(self.crawler, 2, p.VELOCITY_CONTROL, targetVelocity=action[1]*self.action_scale, force=1, physicsClientId=self.client)


    def get_observations(self):
        obs = self.get_state() 
        return obs
    
    def get_state(self):
        pos_orient = p.getBasePositionAndOrientation(self.crawler, physicsClientId=self.client)
        vels = p.getBaseVelocity(self.crawler, physicsClientId=self.client)
        roll, pitch, yaw = p.getEulerFromQuaternion(pos_orient[1])
        local_lin_vel, local_ang_vel = inverse_rotation(vels[0], roll, pitch, yaw), inverse_rotation(vels[1], roll, pitch, yaw)
        return list(pos_orient[0]) + list(pos_orient[1])+ list(local_lin_vel) + list(local_ang_vel) 
