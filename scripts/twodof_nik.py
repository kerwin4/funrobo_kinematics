from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)

def wrap_to_pi(angle):
    """Wrap an angle in radians to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class TwoDOFRobot(TwoDOFRobotTemplate):
    def __init__(self):
        super().__init__()


    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        
        th1, th2 = curr_joint_values[0], curr_joint_values[1]
        l1, l2 = self.l1, self.l2

        H0_1 = np.array([[cos(th1), -sin(th1), 0, l1*cos(th1)],
                         [sin(th1), cos(th1), 0, l1*sin(th1)],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]
                        )

        H1_2 = np.array([[cos(th2), -sin(th2), 0, l2*cos(th2)],
                         [sin(th2), cos(th2), 0, l2*sin(th2)],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]
                        )
        
        Hlist = [H0_1, H1_2]

        # Calculate EE position and rotation
        H_ee = H0_1@H1_2  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist

    def calc_numerical_ik(self, ee, joint_values, tol=0.002, ilimit=300):
        """
        Numerical IK with angles wrapped to [-pi, pi].

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (list[float]): Initial guess for joint angles (radians).
            tol (float, optional): Convergence tolerance. Defaults to 0.01.
            ilimit (int, optional): Maximum number of iterations. Defaults to 200.

        Returns:
            list[float]: Estimated joint angles in radians, wrapped to [-pi, pi].
        """
        x_target, y_target = ee.x, ee.y
        new_joint_values = np.array(joint_values, dtype=float)

        for _ in range(10):  # outer retry loop
            for _ in range(ilimit):
                current_ee, _ = self.calc_forward_kinematics(new_joint_values)
                error = np.array([x_target, y_target]) - np.array([current_ee.x, current_ee.y])

                if np.linalg.norm(error) <= tol:
                    # Wrap angles before returning
                    return [wrap_to_pi(angle) for angle in new_joint_values]

                # IK update
                new_joint_values += self.inverse_jacobian(new_joint_values) @ error
                # Wrap after update
                new_joint_values = np.array([wrap_to_pi(angle) for angle in new_joint_values])

            # If still not converged, restart from a new random valid joint config
            new_joint_values = np.array(ut.sample_valid_joints(self), dtype=float)

        # Return the last attempt wrapped to [-pi, pi]
        return [float(wrap_to_pi(angle)) for angle in new_joint_values]
    
    def jacobian(self, joint_values: list):
        """
        Returns the Jacobian matrix for the robot. 

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
        """
        
        return np.array([
            [-self.l1 * sin(joint_values[0]) - self.l2 * sin(joint_values[0] + joint_values[1]), 
             -self.l2 * sin(joint_values[0] + joint_values[1])],
            [self.l1 * cos(joint_values[0]) + self.l2 * cos(joint_values[0] + joint_values[1]), 
             self.l2 * cos(joint_values[0] + joint_values[1])]
        ])
    

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        return np.linalg.pinv(self.jacobian(joint_values))

if __name__ == "__main__":
    model = TwoDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()