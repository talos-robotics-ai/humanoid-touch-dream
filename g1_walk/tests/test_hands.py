"""HTD's DexPilot chain to Inspire registers, and the touch message layout,
offline (no Modbus, no DDS).

    python -m unittest discover -s g1_walk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

paths.add_teleop()
paths.enter_htd_robot_control()
from hands import HtdInspireModel, to_registers, touch_message  # noqa: E402
from teleop.utils.inspire_touch import DFTP_TACTILE_PER_HAND, DFTP_TACTILE_REGIONS  # noqa: E402

TIPS = (4, 9, 14, 19, 24)
SCALING = 1.20      # inspire_hand_dftp.yml scaling_factor


class HandsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = HtdInspireModel()

    def keypoints(self, side, q):
        kp = np.zeros((25, 3))
        kp[list(TIPS)] = self.model.fingertips(side, q) / SCALING
        return kp.ravel()

    def registers(self, q):
        out = self.model.retarget(self.keypoints("left", q), self.keypoints("right", q))
        self.assertIsNotNone(out)
        return to_registers(out[0]), to_registers(out[1])

    def test_open_hand_is_near_1000_and_a_closed_finger_lower(self):
        open_q = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        closed_q = np.array([0.5, 0.25, 1.0, 1.0, 1.0, 1.0])
        for _ in range(30):     # the retargeter low-pass filters (alpha 0.2)
            l_open, r_open = self.registers(open_q)
        for _ in range(30):
            l_closed, r_closed = self.registers(closed_q)
        # hardware order: little, ring, middle, index, thumb bend, thumb rotation
        for regs_open, regs_closed in ((l_open, l_closed), (r_open, r_closed)):
            self.assertTrue(all(v > 850 for v in regs_open[:4]), regs_open)
            self.assertTrue(all(c < o - 300 for c, o in zip(regs_closed[:4], regs_open[:4])),
                            (regs_open, regs_closed))
            self.assertTrue(all(0 <= v <= 1000 for v in regs_closed))

    def test_uninitialised_keypoints_send_nothing(self):
        self.assertIsNone(self.model.retarget(np.zeros(75), np.ones(75)))

    def test_measured_registers_round_trip_through_htd_units(self):
        regs = [1000, 750, 500, 250, 1000, 0]
        self.model.set_measured(regs, regs, [100] * 6, [200] * 6)
        p = self.model.get_current_hand_proprio()
        left = p["left_hand_proprio"]
        # retargeting order: thumb yaw (rotation), thumb pitch (bend), index .. pinky
        np.testing.assert_allclose(left[:6], [1.3, 0.0, 0.75 * 1.435, 0.5 * 1.435,
                                              0.25 * 1.435, 0.0], atol=1e-9)
        np.testing.assert_allclose(left[6:], [0.1] * 6)
        self.assertEqual(self.model.get_current_hand_tip_poses().shape, (30,))


class TouchMessageTest(unittest.TestCase):
    def test_regions_in_order(self):
        taxels = np.arange(DFTP_TACTILE_PER_HAND)
        msg = touch_message(taxels)
        at = 0
        for name, _, rows, cols in DFTP_TACTILE_REGIONS:
            field = getattr(msg, name + "_touch")
            self.assertEqual(list(field), list(range(at, at + rows * cols)), name)
            at += rows * cols
        self.assertEqual(at, 1062)


if __name__ == "__main__":
    unittest.main()
