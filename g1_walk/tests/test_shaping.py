"""The arm and torso shaping, offline, with the pinned teleop's jump guard.

    python -m unittest discover -s g1_walk/tests -p 'test_*.py'
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from shaping import ArmShaper, TorsoEngage, glide_seconds, smoothstep  # noqa: E402

paths.add_teleop()
from teleop.robot_control.drive_safety import JointJumpGuard, arm_jump_thresholds  # noqa: E402
from teleop.robot_control.g1_backend import ARM_HOLD_MAX, JUMP_CREEP_RAD  # noqa: E402

DT = 0.02


def shaper(max_vel=4.0, engage_s=1.0, engage_vel=1.0):
    guard = JointJumpGuard(arm_jump_thresholds(), "test", "arms", max_hold_steps=ARM_HOLD_MAX)
    return ArmShaper(guard, lambda q: 0.1 * np.asarray(q), JUMP_CREEP_RAD, DT, max_vel,
                     engage_s, engage_vel, engage_max_s=4.0)


class GlideTest(unittest.TestCase):
    def test_glide_seconds(self):
        z = np.zeros(14)
        self.assertEqual(glide_seconds(z, z + 0.2, 1.0, 1.0, 4.0), 1.0)
        self.assertAlmostEqual(glide_seconds(z, z + 2.0, 1.0, 1.0, 4.0), 3.0)
        self.assertEqual(glide_seconds(z, z + 9.0, 1.0, 1.0, 4.0), 4.0)

    def test_smoothstep_ends(self):
        self.assertEqual((smoothstep(-1), smoothstep(0.5), smoothstep(2)), (0.0, 0.5, 1.0))


class ArmShaperTest(unittest.TestCase):
    def run_session(self, s, q0, target, seconds):
        s.begin(q0)
        out, t = [], 0.0
        while t <= seconds + 1e-9:
            res = s.shape(target, t)
            out.append(None if res is None else res[0])
            t += DT
        return out

    def test_engage_eases_from_the_arms_onto_the_solution(self):
        q0, target = np.zeros(14), np.full(14, 0.6)
        out = self.run_session(shaper(), q0, target, 1.5)
        np.testing.assert_allclose(out[0], q0)
        self.assertTrue(np.all(np.diff([o[0] for o in out]) >= -1e-12))
        np.testing.assert_allclose(out[-1], target)
        # peak speed of a 1 s smoothstep over 0.6 rad: 0.9 rad/s
        self.assertLess(np.max(np.abs(np.diff(out, axis=0))) / DT, 0.95)

    def test_speed_cap_binds_after_the_engage(self):
        s = shaper(max_vel=1.0)
        self.run_session(s, np.zeros(14), np.zeros(14), 1.2)
        q, _ = s.shape(np.full(14, 0.5), 1.3)
        np.testing.assert_allclose(q, np.full(14, 1.0 * DT))

    def test_gravity_rides_on_the_shaped_command(self):
        s = shaper()
        s.begin(np.zeros(14))
        q, tau = s.shape(np.full(14, 0.3), 0.0)
        np.testing.assert_allclose(tau, 0.1 * q)

    def test_non_finite_solution_is_refused(self):
        s = shaper()
        self.run_session(s, np.zeros(14), np.zeros(14), 1.2)
        self.assertIsNone(s.shape(np.full(14, np.nan), 1.3))

    def test_jump_is_refused_and_the_baseline_creeps(self):
        s = shaper(max_vel=0.0)            # no speed cap: the guard alone
        self.run_session(s, np.zeros(14), np.zeros(14), 1.2)
        jump = np.zeros(14)
        jump[0] = 1.5                      # past the pi/3 shoulder threshold
        self.assertIsNone(s.shape(jump, 1.3))
        self.assertAlmostEqual(s.last[0], JUMP_CREEP_RAD)


class TorsoEngageTest(unittest.TestCase):
    def test_eases_from_the_applied_command(self):
        t = TorsoEngage(1.0)
        t.begin((0.0, 0.0, 0.0, 0.0), 0.0)
        self.assertEqual(t.shape((0.0, 0.0, 0.0, 0.4), 0.0), (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(t.shape((0.0, 0.0, 0.0, 0.4), 0.5)[3], 0.2)
        self.assertEqual(t.shape((0.0, 0.0, 0.0, 0.4), 1.0), (0.0, 0.0, 0.0, 0.4))
        self.assertEqual(t.shape((0.0, 0.0, 0.0, -0.1), 1.1), (0.0, 0.0, 0.0, -0.1))


if __name__ == "__main__":
    unittest.main()
