import pybullet as p
import os

class Wall:
    def __init__(self, client) -> None:
        f_name = os.path.join(os.path.dirname(__file__), "resources/wall.urdf")
        wall = p.loadURDF(fileName=f_name,
                          physicsClientId=client)