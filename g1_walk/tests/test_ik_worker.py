"""ik_worker.IkWorker on a fake solver, and IkProcess on a fake IK in a real child.

    python -m unittest discover -s g1_walk/tests -p 'test_*.py'
"""
import os
import signal
import sys
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.append(os.path.dirname(os.path.dirname(HERE)))
from ik_worker import IkProcess, IkWorker  # noqa: E402

Q0 = np.zeros(14)


def frame(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return SimpleNamespace(left_pose=pose, right_pose=np.eye(4), name=x)


class Log:
    def __init__(self):
        self.lines = []

    def exception(self, line):
        self.lines.append(line)

    warning = info = exception


class FakeSolve:
    """Records the frames it solves; blocks while ``gate`` is clear; raises
    for an x listed in ``fail``; q_sol is x in every joint."""

    def __init__(self, delay=0.0):
        self.solved, self.fail, self.delay = [], set(), delay
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()

    def __call__(self, left, right, q, dq):
        x = float(left[0, 3])
        self.entered.set()
        self.gate.wait(5.0)
        if self.delay:
            time.sleep(self.delay)
        if x in self.fail:
            raise ValueError("no solution for %s" % x)
        self.solved.append(x)
        return np.full(14, x), None


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.002)
    return False


class IkWorkerTest(unittest.TestCase):
    def setUp(self):
        self.solve, self.log = FakeSolve(), Log()
        self.w = IkWorker(self.solve, lambda: (Q0, Q0), self.log)

    def tearDown(self):
        self.solve.gate.set()
        self.assertTrue(self.w.stop(timeout=1.0))

    def test_latest_is_none_before_the_first_solve(self):
        self.assertIsNone(self.w.latest())
        self.w.start()
        self.assertIsNone(self.w.latest())
        f = frame(1.0)
        self.w.submit(f)
        self.assertTrue(wait_for(lambda: self.w.latest() is not None))
        got, q, t = self.w.latest()
        self.assertIs(got, f)
        np.testing.assert_array_equal(q, np.full(14, 1.0))
        self.assertLessEqual(t, time.monotonic())
        self.assertEqual(self.w.owed_for(time.monotonic()), 0.0)

    def test_newest_frame_wins_before_the_worker_runs(self):
        self.w.submit(frame(1.0))
        self.w.submit(frame(2.0))
        self.w.start()
        self.assertTrue(wait_for(lambda: self.w.latest() is not None))
        time.sleep(0.05)
        self.assertEqual(self.solve.solved, [2.0])

    def test_newest_frame_wins_while_a_solve_runs(self):
        self.solve.gate.clear()
        self.w.start()
        self.w.submit(frame(1.0))
        self.assertTrue(self.solve.entered.wait(1.0))
        for x in (2.0, 3.0, 4.0):
            self.w.submit(frame(x))
        self.solve.gate.set()
        self.assertTrue(wait_for(lambda: self.w.latest() is not None and self.w.latest()[0].name == 4.0))
        time.sleep(0.05)
        self.assertEqual(self.solve.solved, [1.0, 4.0])

    def test_a_raising_solve_keeps_the_previous_solution_and_logs_once(self):
        self.w.start()
        self.w.submit(frame(1.0))
        self.assertTrue(wait_for(lambda: self.w.latest() is not None))
        self.solve.fail.update({2.0, 3.0})
        t_submit = time.monotonic()
        self.w.submit(frame(2.0))
        self.assertTrue(wait_for(lambda: len(self.log.lines) == 1))
        self.w.submit(frame(3.0))
        time.sleep(0.05)
        self.assertEqual(self.w.latest()[0].name, 1.0)
        self.assertEqual(len(self.log.lines), 1)
        self.assertIn("IK solve failed", self.log.lines[0])
        # Still owed since the first failed frame: the controller's stall check.
        self.assertGreaterEqual(self.w.owed_for(time.monotonic()), time.monotonic() - t_submit - 0.05)
        self.w.submit(frame(5.0))
        self.assertTrue(wait_for(lambda: self.w.latest()[0].name == 5.0))
        self.assertEqual(self.w.owed_for(time.monotonic()), 0.0)

    def test_clear_drops_the_pending_frame_and_a_solve_in_flight(self):
        self.solve.gate.clear()
        self.w.start()
        self.w.submit(frame(1.0))
        self.assertTrue(self.solve.entered.wait(1.0))
        self.w.submit(frame(2.0))
        self.w.clear()
        self.solve.gate.set()
        time.sleep(0.1)
        self.assertIsNone(self.w.latest())
        self.assertEqual(self.solve.solved, [1.0])
        self.assertEqual(self.w.owed_for(time.monotonic()), 0.0)

    def test_stop_joins_within_the_timeout(self):
        self.w.start()
        t0 = time.monotonic()
        self.assertTrue(self.w.stop(timeout=1.0))
        self.assertLess(time.monotonic() - t0, 0.5)
        # Stopped with a solve in flight: it finishes, then the thread exits.
        w = IkWorker(FakeSolve(delay=0.1), lambda: (Q0, Q0), Log())
        w.start()
        w.submit(frame(1.0))
        time.sleep(0.02)
        self.assertTrue(w.stop(timeout=1.0))

    def test_stop_before_start(self):
        self.assertTrue(IkWorker(self.solve, lambda: (Q0, Q0), self.log).stop())

    def test_solve_timing_populates(self):
        self.assertEqual(self.w.stats(time.monotonic()), (0.0, 0.0))
        self.solve.delay = 0.01
        self.w.start()
        for x in range(5):
            self.w.submit(frame(float(x)))
            self.assertTrue(wait_for(lambda: self.w.latest() is not None
                                     and self.w.latest()[0].name == float(x)))
        p50, mx = self.w.stats(time.monotonic())
        self.assertGreaterEqual(p50, 9.0)
        self.assertGreaterEqual(mx, p50)
        self.assertLess(mx, 500.0)

    def test_a_failing_measure_counts_as_a_failed_solve(self):
        def measure():
            raise RuntimeError("no rt/lowstate yet")
        w = IkWorker(self.solve, measure, self.log)
        w.start()
        w.submit(frame(1.0))
        self.assertTrue(wait_for(lambda: len(self.log.lines) == 1))
        self.assertIsNone(w.latest())
        self.assertTrue(w.stop())


# -- the child process: module-level so the spawned child imports them by name --------
class FakeIk:
    def solve_ik(self, left, right, q, dq):
        if left[0, 3] < 0:
            raise ValueError("unreachable")
        return np.asarray(q, float) + left[0, 3], np.zeros(14)


def fake_ik(offset):
    return FakeIk()


def failing_ik():
    raise ValueError("no URDF here")


class IkProcessTest(unittest.TestCase):
    def test_solves_in_the_child(self):
        p = IkProcess(fake_ik, 0.0)
        try:
            p.wait_ready(30.0)
            self.assertNotEqual(p.pid, os.getpid())
            q, info = p.solve(frame(0.5).left_pose, np.eye(4), np.arange(14.0), Q0)
            np.testing.assert_array_equal(q, np.arange(14.0) + 0.5)
            self.assertIsNone(info)
            with self.assertRaisesRegex(RuntimeError, "unreachable"):
                p.solve(frame(-1.0).left_pose, np.eye(4), Q0, Q0)
            q, _ = p.solve(frame(1.0).left_pose, np.eye(4), Q0, Q0)     # still answering
            np.testing.assert_array_equal(q, np.ones(14))
        finally:
            p.close()
        self.assertFalse(p.is_alive())

    def test_a_factory_that_raises_fails_wait_ready(self):
        p = IkProcess(failing_ik)
        try:
            with self.assertRaisesRegex(RuntimeError, "no URDF here"):
                p.wait_ready(30.0)
        finally:
            p.close()

    def test_a_dead_child_leaves_the_last_solution_and_the_frames_owed(self):
        p = IkProcess(fake_ik, 0.0)
        log = Log()
        w = IkWorker(p.solve, lambda: (Q0, Q0), log)
        try:
            p.wait_ready(30.0)
            w.start()
            w.submit(frame(1.0))
            self.assertTrue(wait_for(lambda: w.latest() is not None))
            os.kill(p.pid, signal.SIGKILL)
            self.assertTrue(wait_for(lambda: not p.is_alive()))
            t0 = time.monotonic()
            w.submit(frame(2.0))
            self.assertTrue(wait_for(lambda: len(log.lines) == 1))
            self.assertLess(time.monotonic() - t0, 0.5)
            w.submit(frame(3.0))
            time.sleep(0.05)
            self.assertEqual(w.latest()[0].name, 1.0)
            self.assertEqual(len(log.lines), 1)
            self.assertGreater(w.owed_for(time.monotonic()), 0.0)
        finally:
            self.assertTrue(w.stop(timeout=1.0))
            p.close()


if __name__ == "__main__":
    unittest.main()
