import pybullet as p
import os
from math import pi
import math

class Crawler:
    def __init__(self, client):
        # start_pos = [0.01,0,0]
        # start_orient = p.getQuaternionFromEuler([0,-pi/2.,pi])
        start_pos = [0,0,0.01]
        start_orient = p.getQuaternionFromEuler([0,0,0])
        
        f_name = os.path.join(os.path.dirname(__file__), "resources/crawler/crawler_caster.urdf")
        self.crawler = p.loadURDF(f_name, 
                                  start_pos,
                                  start_orient,
                                  physicsClientId=client)
        self.action_scale = 10
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
        # print("X VEL: ", obs[7]) 
        return obs
    
    def get_state(self):
        pos_orient = p.getBasePositionAndOrientation(self.crawler, physicsClientId=self.client)
        vels = p.getBaseVelocity(self.crawler, physicsClientId=self.client)
        return list(pos_orient[0]) + list(pos_orient[1])+ list(vels[0]) + list(vels[1]) 