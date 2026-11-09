"""``rt/robot/state`` as HTD's controller computes it, from the walker's
robot state.

HTD's ``G1_29_WholeBodyController.get_current_estimated_state``
(robot_control/g1_wholebody.py) runs the feet-flat kinematic estimator on the
29 measured joints and converts its transforms to xyz + rpy. Here the same
three methods run on the joints and IMU the teleop's ``G1_29_JetsonBody``
decodes from rt/lowstate (motor order 0..28, as HTD's ``G1_29_JointIndex``).
The message's ``velocity_commands`` is the walk command the walker applied on
its last tick, with vy zero: what HTD's controller reports there is the
command its own lower body is executing.

Import after ``paths.add_teleop()`` and ``paths.enter_htd_robot_control()``.
"""
import time

import numpy as np

from g1_wholebody import G1_29_WholeBodyController
from robot_state_estimator_g1 import G1_29_KinematicStateEstimator
from teleop_dds_msgs import EEFState, G1RobotState

G1_URDFS = {0: "g1_body29_hand14_wrist4010.urdf", 1: "g1_body29_hand14_wrist5010.urdf"}


class HtdStateEstimate:
    """The three ``G1_29_WholeBodyController`` methods that turn measured
    joints into the poses of ``rt/robot/state``, bound to another joint
    source (the class itself would open rt/lowcmd)."""

    get_current_estimated_state = G1_29_WholeBodyController.get_current_estimated_state
    _convert_T_pose_to_xyz_rpy = G1_29_WholeBodyController._convert_T_pose_to_xyz_rpy
    rot_mat_to_rpy = G1_29_WholeBodyController.rot_mat_to_rpy

    def __init__(self, g1_type, motor_q_fn):
        urdf_dir = "./assets/g1/"
        self.state_estimator = G1_29_KinematicStateEstimator(
            urdf_path=urdf_dir + G1_URDFS[g1_type], urdf_dir=urdf_dir)
        self.get_current_motor_q = motor_q_fn


def robot_state_message(estimate, body, walk_command):
    """One ``G1RobotState`` from the body's latest rt/lowstate; raises the
    body's ``LowStateStale`` when that state is too old to report."""
    imu = body.get_imu_data()
    s = estimate.get_current_estimated_state()
    vx, wz = walk_command
    return G1RobotState(
        timestamp=time.time(),
        torso_pose=s["torso_pose"].tolist(),
        left_end_effector_pose=s["left_end_effector_pose"].tolist(),
        right_end_effector_pose=s["right_end_effector_pose"].tolist(),
        pelvis_pose=s["pelvis_pose"].tolist(),
        left_foot_pose=s["left_foot_pose"].tolist(),
        right_foot_pose=s["right_foot_pose"].tolist(),
        left_end_effector_to_torso_pose=s["left_end_effector_to_torso_pose"].tolist(),
        right_end_effector_to_torso_pose=s["right_end_effector_to_torso_pose"].tolist(),
        joint_positions=body.get_current_motor_q().tolist(),
        joint_velocities=body.get_current_motor_dq().tolist(),
        velocity_commands=[float(vx), 0.0, float(wz)],
        imu_quaternion=list(imu.quaternion),
        imu_accelerometer=list(imu.accelerometer),
        imu_gyroscope=list(imu.gyroscope),
        imu_rpy=list(imu.rpy),
    )


def eef_state_message(joints, forces, tips):
    return EEFState(timestamp=time.time(), joint_positions=np.asarray(joints).tolist(),
                    finger_forces=np.asarray(forces).tolist(),
                    hand_link_poses=np.asarray(tips).tolist())
