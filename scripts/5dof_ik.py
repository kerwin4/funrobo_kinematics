from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class FiveDOF(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()

    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            joint_values (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        curr_joint_values = joint_values.copy()

        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

        # Ensure that the joint angles respect the joint limits
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
        
        # DH parameters for each joint
        DH = np.zeros((self.num_dof, 4))
        DH[0] = [curr_joint_values[0], self.l1, 0, -pi/2]
        DH[1] = [curr_joint_values[1]-(pi/2), 0, self.l2, pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, pi]
        DH[3] = [curr_joint_values[3]+(pi/2), 0, 0, pi/2]
        DH[4] = [curr_joint_values[4], self.l4+self.l5, 0, 0]

        # Compute the transformation matrices
        Hlist = [ut.dh_to_matrix(dh) for dh in DH]

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Calculate EE position and rotation
        H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist
    
    def calc_inverse_kinematics(self, ee, joint_values, soln = 0):
        '''
        calculate some stuff
        '''
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        pee = np.array([ee.x, ee.y, ee.z])
        rpy = np.array([ee.rotx, ee.roty, ee.rotz])
        rot_m = ut.euler_to_rotm(rpy)

        p_wrist = pee - (l4+l5)*(rot_m @ np.array([0,0,1]))
        x_wrist = p_wrist[0]
        y_wrist = p_wrist[1]
        z_wrist = p_wrist[2]

        th1 = atan2(y_wrist,x_wrist)

        L = sqrt(x_wrist**2+y_wrist**2+(z_wrist-l1)**2)
        beta = np.arccos((l2**2+l3**2-L**2)/(2*l2*l3))
        
        if soln == 0:
            th3 = pi-beta
        if soln == 1:
            th3 = -(pi-beta)

        alpha = atan2(l3*sin(th3),l2+l3*cos(th3))
        gamma = atan2(z_wrist-l1,sqrt(x_wrist**2+y_wrist**2))
        th2_placeholder = gamma-alpha
        th2 = pi/2 - th2_placeholder

        



if __name__ == "__main__":
    model = FiveDOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()