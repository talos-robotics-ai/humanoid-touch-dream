"""TeleopDataWB -> walker commands, offline.

    python -m unittest discover -s g1_walk/tests -p 'test_*.py'
"""
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commands import WalkGate, parse_teleop_data, torso_command, walk_command  # noqa: E402

STAND = 0.78                    # teleop G1_STAND_HEIGHT
VX_BOX, WZ_BOX = (-0.4, 0.6), (-1.0, 1.0)


def message(**over):
    fields = dict(timestamp=12.5, left_arm_pose=np.eye(4).ravel().tolist(),
                  right_arm_pose=np.eye(4).ravel().tolist(),
                  left_hand_pose=[0.0] * 75, right_hand_pose=[0.0] * 75,
                  torso_rpy=[0.0, 0.0, 0.0], torso_height=0.78, velocity_commands=[])
    fields.update(over)
    return SimpleNamespace(**fields)


def frame(**over):
    f = parse_teleop_data(message(**over))
    assert f is not None
    return f


class ParseTest(unittest.TestCase):
    def test_valid_message(self):
        pose = np.arange(16.0)
        f = frame(left_arm_pose=pose.tolist(), velocity_commands=[0.2, 0.0, 0.3])
        np.testing.assert_array_equal(f.left_pose, pose.reshape(4, 4))
        np.testing.assert_array_equal(f.velocity, [0.2, 0.0, 0.3])
        self.assertEqual(f.stamp, 12.5)

    def test_no_velocity_is_none(self):
        self.assertIsNone(frame().velocity)

    def test_wrong_lengths_are_dropped(self):
        for over in (dict(left_arm_pose=[0.0] * 15), dict(right_hand_pose=[0.0] * 74),
                     dict(torso_rpy=[0.0, 0.0]), dict(velocity_commands=[0.1, 0.2])):
            self.assertIsNone(parse_teleop_data(message(**over)), over)

    def test_non_finite_values_are_dropped(self):
        for over in (dict(torso_height=float("nan")), dict(velocity_commands=[np.inf, 0, 0]),
                     dict(left_hand_pose=[np.nan] + [0.0] * 74)):
            self.assertIsNone(parse_teleop_data(message(**over)), over)


class WalkTest(unittest.TestCase):
    def test_passes_vx_and_wz_and_drops_vy(self):
        self.assertEqual(walk_command(frame(velocity_commands=[0.2, 0.3, -0.5]), VX_BOX, WZ_BOX),
                         (0.2, -0.5))

    def test_clips_to_the_walker_box(self):
        f = frame(velocity_commands=[0.9, 0.0, -2.0])
        self.assertEqual(walk_command(f, VX_BOX, WZ_BOX), (0.6, -1.0))
        f = frame(velocity_commands=[-0.9, 0.0, 2.0])
        self.assertEqual(walk_command(f, VX_BOX, WZ_BOX), (-0.4, 1.0))

    def test_no_velocity_is_rest(self):
        self.assertEqual(walk_command(frame(), VX_BOX, WZ_BOX), (0.0, 0.0))


class TorsoTest(unittest.TestCase):
    def test_height_is_travel_about_the_stand_height(self):
        # The HTD converter wrote actions/body z = 0.78 + travel - 0.75 and
        # policy_rollout publishes torso_height = z + 0.75.
        for travel in (0.0, -0.05, 0.02):
            z = STAND + travel - 0.75
            h, _, _, _ = torso_command(frame(torso_height=z + 0.75), STAND)
            self.assertAlmostEqual(h, travel, places=12)

    def test_training_constant_body_z_is_the_stand(self):
        h, _, _, _ = torso_command(frame(torso_height=0.03 + 0.75), STAND)
        self.assertAlmostEqual(h, 0.0, places=12)

    def test_yaw_mode_passes_yaw_only(self):
        f = frame(torso_rpy=[0.1, -0.2, 0.45])
        self.assertEqual(torso_command(f, STAND, "yaw")[1:], (0.0, 0.0, 0.45))

    def test_hold_and_follow(self):
        f = frame(torso_rpy=[0.1, -0.2, 0.45])
        self.assertEqual(torso_command(f, STAND, "hold")[1:], (0.0, 0.0, 0.0))
        self.assertEqual(torso_command(f, STAND, "follow")[1:], (0.1, -0.2, 0.45))

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            torso_command(frame(), STAND, "full")


class WalkGateTest(unittest.TestCase):
    def test_holds_until_rest_then_passes(self):
        g = WalkGate(rest_vx=0.05, rest_wz=0.05)
        self.assertEqual(g.step(0.3, 0.0, True), (0.0, 0.0, False))
        self.assertEqual(g.step(0.02, -0.01, True), (0.02, -0.01, True))
        self.assertEqual(g.step(0.3, 0.2, True), (0.3, 0.2, False))

    def test_not_allowed_is_zero_and_keeps_the_gate(self):
        g = WalkGate()
        self.assertEqual(g.step(0.0, 0.0, False), (0.0, 0.0, False))
        self.assertTrue(g.pending)

    def test_rearm_needs_rest_again(self):
        g = WalkGate()
        g.step(0.0, 0.0, True)
        g.rearm()
        self.assertEqual(g.step(0.2, 0.0, True), (0.0, 0.0, False))


if __name__ == "__main__":
    unittest.main()
