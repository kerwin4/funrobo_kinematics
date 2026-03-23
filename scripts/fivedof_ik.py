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

    def _compute_transforms(self, joint_values):
        """
        Helper to calculate cumulative transformation matrices (H_cumulative)
        and individual joint transforms (Hlist) based on the FiveDOF robot model.
        """
        theta = joint_values
        DH = np.array([ # DH parameters for each joint
            [theta[0],          self.l1,         0,       -pi/2],
            [theta[1]-(pi/2),   0,               self.l2,  pi],
            [theta[2],          0,               self.l3,  pi],
            [theta[3]+(pi/2),   0,               0,        pi/2],
            [theta[4],          self.l4+self.l5, 0,        0]
        ])

        Hlist = [ut.dh_to_matrix(dh) for dh in DH] # Compute transformation matrices for each joint

        # Compute cumulative transformations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)
            
        return H_cumulative, Hlist
    
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            joint_values (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).

        Returns:

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
    
    def calc_inverse_kinematics(self, ee, joint_values=None, soln=0):
            """
            Calculates the analytical inverse position kinematics for the 5-DOF manipulator.
            Utilizes kinematic decoupling to resolve the proximal spatial positioning 
            independently from the distal spatial orientation.

            Args:
                ee (EndEffector): Desired end-effector pose.
                joint_values (list[float]): Initial guess for joint angles (radians), not used for analytical.
                soln (int): Which of 2 valid solutions to choose, defaults to 0.

            Returns:
                A list of joint angles that satisfies the desired end effector position within joint limits.
            """
            for soln in range(5):
                p_ee = np.array([ee.x, ee.y, ee.z])
                R_05 = ut.euler_to_rotm([ee.rotx, ee.roty, ee.rotz])
                

                d_6 = self.l4 + self.l5
                p_wrist = p_ee - d_6 * (R_05 @ np.array([0, 0, 1]))
                
                wx, wy, wz = p_wrist[0], p_wrist[1], p_wrist[2]
                
                theta_1 = atan2(wy, wx)
                
                r = sqrt(wx**2 + wy**2)
                s = wz - self.l1
                
                # To achieve nearest location, we clip the value of D to be within the range of -1 to 1, which corresponds to the valid range of the cosine function.
                D = (r**2 + s**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
                D = np.clip(D, -1.0, 1.0) 
                
                # Elbow up vs. elbow down
                if soln == 0:
                    theta_3_planar = np.arccos(D)
                else:
                    theta_3_planar = -np.arccos(D)
                    
                alpha = atan2(s, r)
                beta = atan2(self.l3 * sin(theta_3_planar), self.l2 + self.l3 * cos(theta_3_planar))
                theta_2_planar = alpha - beta
                
                theta_2 = (pi / 2) - theta_2_planar 
                theta_3 = theta_3_planar
                
                # Use calculated theta_1, theta_2, and theta_3 to compute the rotation from joint 3 to the end effector (R_35), which is needed to solve for theta_4 and theta_5.
                q_list = [theta_1, theta_2, theta_3, 0, 0]
                H_cumulative, _ = self._compute_transforms(q_list)
                R_03 = H_cumulative[3][:3, :3] 
                
                R_35 = R_03.T @ R_05
                
                # Calculate theta_4 and theta_5 from R_35
                theta_4 = atan2(R_35[1, 2], R_35[0, 2])
                theta_5 = atan2(R_35[2, 0], R_35[2, 1])
                
                new_joints = [theta_1, theta_2, theta_3, theta_4, theta_5]
                        # Ensure joint angles stay within limits
                new_joint_clipped = np.clip(new_joints, 
                                    [limit[0] for limit in self.joint_limits], 
                                    [limit[1] for limit in self.joint_limits]
                                    )
            
            return [self.normalize_angle(q) for q in new_joint_clipped]
    
    def normalize_angle(self, angle):
        """
        Normalize an angle to the range (-pi, pi].
        
        Args:
            angle (float): The angle to be wrapped

        Returns:
            The input angle wrapped between -pi and pi
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def calc_numerical_ik(self, ee, joint_values, tol=0.002, ilimit=100):
        """
        Numerical IK within joint limits.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (list[float]): Initial guess for joint angles (radians).
            tol (float, optional): Convergence tolerance. Defaults to 0.002.
            ilimit (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            list[float]: Estimated joint angles in radians and within joint limits.
        """
        # unpack end effector position and rotation
        x_target, y_target, z_target = ee.x, ee.y, ee.z
        new_joint_values = np.array(joint_values, dtype=float)

        for _ in range(100):  # allow 100 attempted starting configurations
            for _ in range(ilimit): # 100 iterations for each attempted configuration
                # get the end effector position based on the current joint guess and find the error from the desired position
                current_ee, _ = self.calc_forward_kinematics(new_joint_values)
                error = np.array([x_target, y_target, z_target]) - np.array([current_ee.x, current_ee.y, current_ee.z])

                # if the error is within tolerance, return the joint angle solution
                if np.linalg.norm(error) <= tol:
                    return new_joint_values

                # get next iteration by updating with inverse jacobian
                new_joint_values += self.inverse_jacobian(new_joint_values) @ error

                # enforce joint limits
                for i, (low, high) in enumerate(self.joint_limits):
                    new_joint_values[i] = np.clip(new_joint_values[i], low, high)

            # if not converged, return a random configuration and try again
            new_joint_values = np.array(ut.sample_valid_joints(self), dtype=float)
        
        # return null if not converged
        return np.zeros(len(joint_values))

    def jacobian(self, joint_values: list):
        """
        Returns the Jacobian matrix for the robot. 

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
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

        p_ee = H_cumulative[-1][:3, 3] 
        J = np.zeros((6, self.num_dof)) 
        
        for i in range(self.num_dof):
            transform_i = H_cumulative[i] 
            z_axis = transform_i[:3, 2] 
            p_joint = transform_i[:3, 3]
            J[:3, i] = np.cross(z_axis, (p_ee - p_joint))
            J[3:, i] = z_axis
        return J[:3, :]
    

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        return np.linalg.pinv(self.jacobian(joint_values))
    
if __name__ == "__main__":
    model = FiveDOF()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()