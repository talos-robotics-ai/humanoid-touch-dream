"""The Inspire RH56DFTP pair: HTD's retargeting and hand state on the
teleop's Modbus driver.

Commands: the 75 keypoints of each hand go through HTD's DexPilot chain
exactly as ``Inspire_Controller.control_process`` runs it
(robot_control/robot_hand_inspire_dftp.py) -- retarget, hardware order,
``joint_angles_to_hardware`` (1 = open) -- and become registers 0..1000 the way
its ``_publish_hand_command`` scales them. Those registers go to the hands
through the teleop's ``Inspire_Modbus_Controller``
(teleop/robot_control/robot_hand_inspire_modbus.py), the driver that commanded
the hands during the recording, with its own slew, force and speed settings.

State: the driver's angleAct and FORCE_ACT registers are put through HTD's
own ``hardware_to_joint_angles`` / ``get_current_hand_proprio`` /
``get_current_hand_tip_poses``, so ``rt/robot/eef_state`` carries what HTD's
controller would publish from its DDS hand driver; the 2 x 1062 taxels the
driver sweeps are laid out as the ``inspire_hand_touch`` message
policy_rollout subscribes to.

Import after ``paths.add_teleop()`` and ``paths.enter_htd_robot_control()``.
"""
import numpy as np

from inspire_sdkpy import inspire_hand_defaut
from robot_hand_inspire_dftp import DataBuffer, Inspire_Controller, Inspire_Num_Motors
from hand_retargeting import HandRetargeting, HandType
from teleop.robot_control.robot_hand_inspire_modbus import Inspire_Modbus_Controller
from teleop.utils.inspire_touch import (DFTP_TACTILE_PER_HAND, DFTP_TACTILE_REGIONS,
                                        FORCE_PER_HAND)

REGISTER_MAX = 1000


class HtdInspireModel(Inspire_Controller):
    """``Inspire_Controller``'s retargeting, unit conversions and fingertip
    kinematics without its control process and DDS endpoints: the hand
    I/O is the Modbus driver's. Its state buffers are fed by
    :meth:`set_measured` instead of the DDS subscriber."""

    def __init__(self):
        # Not super().__init__(): that starts the DDS hand endpoints and the
        # retargeting process this class exists to replace.
        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_DFTP)
        self._init_hand_forward_kinematics()
        self.left_hand_proprio_buffer = DataBuffer()
        self.right_hand_proprio_buffer = DataBuffer()
        zero = np.zeros(2 * Inspire_Num_Motors)
        self.left_hand_proprio_buffer.SetData(zero.copy())
        self.right_hand_proprio_buffer.SetData(zero.copy())

    def retarget(self, left75, right75):
        """``(left, right)`` 6 values each in [0, 1] (1 = open), hardware
        order, or None while either hand's keypoints are all zero -- one
        iteration of ``control_process``."""
        r = self.hand_retargeting
        left = np.asarray(left75, float).reshape(25, 3)
        right = np.asarray(right75, float).reshape(25, 3)
        if np.all(left == 0.0) or np.all(right == 0.0):
            return None
        ref_l = left[r.left_indices[1, :]] - left[r.left_indices[0, :]]
        ref_r = right[r.right_indices[1, :]] - right[r.right_indices[0, :]]
        q_l = r.left_retargeting.retarget(ref_l)[r.left_dex_retargeting_to_hardware]
        q_r = r.right_retargeting.retarget(ref_r)[r.right_dex_retargeting_to_hardware]
        return self.joint_angles_to_hardware(q_l, q_r)

    def set_measured(self, left_regs, right_regs, left_force, right_force):
        """The DDS subscriber's work (``_subscribe_hand_state``) on Modbus
        readings: registers / 1000, angles through ``hardware_to_joint_angles``."""
        left = np.zeros(2 * Inspire_Num_Motors)
        right = np.zeros(2 * Inspire_Num_Motors)
        left[:Inspire_Num_Motors] = np.asarray(left_regs, float) / 1000.0
        right[:Inspire_Num_Motors] = np.asarray(right_regs, float) / 1000.0
        left[Inspire_Num_Motors:] = np.asarray(left_force, float) / 1000.0
        right[Inspire_Num_Motors:] = np.asarray(right_force, float) / 1000.0
        left[:Inspire_Num_Motors], right[:Inspire_Num_Motors] = self.hardware_to_joint_angles(
            left[:Inspire_Num_Motors], right[:Inspire_Num_Motors])
        self.left_hand_proprio_buffer.SetData(left)
        self.right_hand_proprio_buffer.SetData(right)

    def fingertips(self, side, q_target_order):
        """The five fingertip positions (5, 3) in the hand base frame at
        joint angles given in the retargeting order, through the same
        kinematics ``get_current_hand_tip_poses`` uses."""
        if side == "left":
            wrapper, names, adaptor, links = (self.left_hand_wrapper, self.left_target_joint_names,
                                              self.left_adaptor, self.left_hand_link_indices)
        else:
            wrapper, names, adaptor, links = (self.right_hand_wrapper, self.right_target_joint_names,
                                              self.right_adaptor, self.right_hand_link_indices)
        wrapper.compute_forward_kinematics(
            self._build_full_qpos_from_target(q_target_order, wrapper, names, adaptor))
        return np.array([wrapper.get_link_pose(i)[:3, 3] for i in links])


def to_registers(q01):
    """[0, 1] -> Inspire registers, as ``_publish_hand_command`` scales them."""
    return [int(np.clip(v * REGISTER_MAX, 0, REGISTER_MAX)) for v in q01]


def touch_message(taxels):
    """One hand's 1062 taxels (the teleop's region order, which is the
    vendor SDK's) as an ``inspire_hand_touch`` message."""
    msg = inspire_hand_defaut.get_inspire_hand_touch()
    at = 0
    for name, _, rows, cols in DFTP_TACTILE_REGIONS:
        n = rows * cols
        setattr(msg, name + "_touch", [int(v) for v in taxels[at:at + n]])
        at += n
    return msg


class InspireHands:
    """The pair, commanded and read through the teleop's Modbus driver.
    Addresses come from the driver's own PSI_INSPIRE_LEFT_IP /
    PSI_INSPIRE_RIGHT_IP / PSI_INSPIRE_PORT (and PSI_INSPIRE_HANDS)."""

    def __init__(self):
        self.model = HtdInspireModel()
        self.driver = Inspire_Modbus_Controller(touch=True)

    def command(self, left75, right75):
        """Retarget one frame and send it; False while the keypoints are
        not initialised (the hands keep their last command)."""
        q = self.model.retarget(left75, right75)
        if q is None:
            return False
        self.driver.ctrl_dual_hand(to_registers(q[0]), to_registers(q[1]))
        return True

    def open(self):
        self.driver.open_hands()

    def read(self):
        """``(joint_positions, finger_forces, hand_link_poses, touch_l,
        touch_r)`` as HTD's controller publishes them: right hand first,
        joints in the retargeting order, fingertips from the measured
        angles; and the two raw taxel vectors."""
        regs = self.driver.get_current_dual_hand_registers()
        press = self.driver.get_current_dual_hand_pressure()
        n = DFTP_TACTILE_PER_HAND
        force_l = press[2 * n:2 * n + FORCE_PER_HAND]
        force_r = press[2 * n + FORCE_PER_HAND:2 * n + 2 * FORCE_PER_HAND]
        self.model.set_measured(regs[:6], regs[6:12], force_l, force_r)
        proprio = self.model.get_current_hand_proprio()
        left, right = proprio["left_hand_proprio"], proprio["right_hand_proprio"]
        joints = np.concatenate([right[:Inspire_Num_Motors], left[:Inspire_Num_Motors]])
        forces = np.concatenate([right[Inspire_Num_Motors:], left[Inspire_Num_Motors:]])
        tips = self.model.get_current_hand_tip_poses()
        return joints, forces, tips, press[:n], press[n:2 * n]
