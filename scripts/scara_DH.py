from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class ScaraRobot(ScaraRobotTemplate):
    def __init__(self):
        super().__init__()


    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()

        th1, th2, d3 = curr_joint_values[0], curr_joint_values[1], curr_joint_values[2]
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5, 

        DH_table = np.array([[th1, l1, l2, 0],
                             [th2, l3-l5, l4, 0],
                             [0, -d3, 0, pi]])
        
        H = np.zeros([12,4])

        for i in range(3):
            H[4*i:4*i+4, :] = np.array([[cos(DH_table[i][0]), -sin(DH_table[i][0])*cos(DH_table[i][3]), sin(DH_table[i][0])*sin(DH_table[i][3]), DH_table[i][2]*cos(DH_table[i][0])], 
                                        [sin(DH_table[i][0]), cos(DH_table[i][0])*cos(DH_table[i][3]), -cos(DH_table[i][0])*sin(DH_table[i][3]), DH_table[i][2]*sin(DH_table[i][0])], 
                                        [0, sin(DH_table[i][3]), cos(DH_table[i][3]), DH_table[i][1]], 
                                        [0, 0, 0, 1]])
        
        H0_1 = H[0:4, :]
        H1_2 = H[4:8, :]
        H2_3 = H[8:12, :]

        Hlist = [H0_1, H1_2, H2_3]

        # Calculate EE position and rotation
        H_ee = H0_1@H1_2@H2_3  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


if __name__ == "__main__":
    model = ScaraRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()