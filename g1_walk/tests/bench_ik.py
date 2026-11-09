#!/usr/bin/env python3
"""The arm IK's cost and the control tick around it, without DDS or the walker.

    PYTHONPATH= python g1_walk/tests/bench_ik.py            # solve timing and the GIL
    PYTHONPATH= python g1_walk/tests/bench_ik.py --tick     # the 50 Hz tick, inline vs worker

Solve timing: (a) sequential solve_ik calls on the main thread; (b) the same
solves on a background thread while the main thread runs a 1 kHz sleep loop:
a max gap near a solve time means the solve holds the GIL; (c) (a) and (b)
again with OpenMP on one thread (IPOPT's MUMPS).

--tick: a controller-like loop at 50 Hz, a new policy frame every 1/30 s,
the latest solution through the real ArmShaper; the solve inline in the tick
(the controller before IkWorker) or on IkWorker + IkProcess. Tick time is
the loop body's, as the controller's TickTimer measures it.

The wrist targets are the IK's forward kinematics of sim_drive.py's arm swing
around the teleop's home pose (what policy_rollout sends in the sim e2e).
"""
import argparse
import os
import sys
import threading
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.append(os.path.dirname(os.path.dirname(HERE)))
import paths  # noqa: E402

FRAME_HZ = 30.0                 # policy_rollout's publish rate
TICK_HZ = 50.0


def stats(ms):
    ms = np.asarray(ms, float)
    return "p50 %6.2f  p90 %6.2f  p99 %6.2f  max %6.2f ms (n=%d)" % (
        np.percentile(ms, 50), np.percentile(ms, 90), np.percentile(ms, 99), ms.max(), len(ms))


def urdf():
    from robot_state import G1_URDFS
    return "./assets/g1/" + G1_URDFS[0], "./assets/g1/"


def build_ik():
    """The IK as the controller builds it, the home pose, and target(i)."""
    import pinocchio as pin
    from controller import load_home_pose
    from robot_arm_ik_g1 import G1_29_ArmIK
    from sim_drive import arm_trajectory
    path, d = urdf()
    ik = G1_29_ArmIK(urdf_path=path, urdf_dir=d)
    home = load_home_pose()
    model = ik.reduced_robot.model
    data = model.createData()
    ids = (model.getFrameId("L_ee"), model.getFrameId("R_ee"))

    def target(i):
        """(q the poses came from, left 4x4, right 4x4) of frame i at FRAME_HZ."""
        q = arm_trajectory(home, i / FRAME_HZ)
        pin.framesForwardKinematics(model, data, q)
        return (q,) + tuple(data.oMf[f].homogeneous.copy() for f in ids)
    return ik, home, target


def bench_sequential(ik, target, n, offset=0):
    ms = []
    for i in range(n):
        q, lw, rw = target(offset + i)
        t0 = time.perf_counter()
        ik.solve_ik(lw, rw, q, np.zeros(14))
        ms.append((time.perf_counter() - t0) * 1000.0)
    return ms


def sleep_gaps(until):
    gaps = []
    last = time.perf_counter()
    while not until():
        time.sleep(0.001)
        now = time.perf_counter()
        gaps.append((now - last) * 1000.0)
        last = now
    return gaps


def bench_gil(ik, target, n, offset, omp_threads=None):
    """Solves on a thread; the main thread's 1 kHz loop gaps meanwhile."""
    from ik_worker import set_openmp_threads
    done = threading.Event()
    solve_ms = []

    def worker():
        if omp_threads is not None:
            set_openmp_threads(omp_threads)     # per thread: a new thread starts at the default
        solve_ms.extend(bench_sequential(ik, target, n, offset))
        done.set()

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    gaps = sleep_gaps(done.is_set)
    th.join()
    return solve_ms, gaps


def solve_report(n):
    from ik_worker import set_openmp_threads
    ik, _, target = build_ik()
    bench_sequential(ik, target, 5)                 # warm-up
    t_end = time.perf_counter() + 0.5
    print("    1 kHz loop gap, no solver:            " + stats(sleep_gaps(lambda: time.perf_counter() > t_end)))
    seq = None
    for label, threads in (("OpenMP default", None), ("OpenMP 1 thread", 1)):
        if threads is not None:
            set_openmp_threads(threads)
            bench_sequential(ik, target, 5)
        a = bench_sequential(ik, target, n, 100)
        solve_ms, gaps = bench_gil(ik, target, n, 200, threads)
        seq = seq or a
        print("[%s]" % label)
        print("(a) solve_ik, main thread:            " + stats(a))
        print("(b) solve_ik on a thread:             " + stats(solve_ms))
        print("    1 kHz loop gap on the main thread: " + stats(gaps))
        verdict = "HELD" if max(gaps) > 0.5 * np.median(a) else "RELEASED"
        print("    GIL during solve_ik: %s (gap p50 %.2f / max %.2f ms vs solve p50 %.2f ms)"
              % (verdict, np.median(gaps), max(gaps), np.median(a)))


# -- the tick ----------------------------------------------------------------------
def make_shaper(model):
    import pinocchio as pin
    from shaping import ArmShaper
    from teleop.robot_control.drive_safety import JointJumpGuard, arm_jump_thresholds
    data = model.createData()
    zeros = np.zeros(model.nv)

    def gravity(q):
        return np.asarray(pin.rnea(model, data, np.asarray(q, float), zeros, zeros)).copy()
    guard = JointJumpGuard(arm_jump_thresholds(1.0), "bench", "arms", max_hold_steps=50)
    return ArmShaper(guard, gravity, 0.05, 1.0 / TICK_HZ, 4.0, 1.0, 1.0, 4.0)


def tick_loop(seconds, target, shaper, home, solve_inline=None, worker=None):
    """Tick ms, loop period ms, ticks that sent arms."""
    shaper.begin(home)
    dt = 1.0 / TICK_HZ
    t_start = time.monotonic()
    next_t = t_start
    frame, frame_i, solved_for, sol = None, -1, None, None
    tick_ms, period_ms, sent = [], [], 0
    last_t = None
    while time.monotonic() - t_start < seconds:
        now = time.monotonic()
        if last_t is not None:
            period_ms.append((now - last_t) * 1000.0)
        last_t = now
        i = int((now - t_start) * FRAME_HZ)
        if i != frame_i:                            # policy_rollout's frame, decoded elsewhere
            frame_i = i
            _, lw, rw = target(i)
            frame = types.SimpleNamespace(left_pose=lw, right_pose=rw)
        t0 = time.perf_counter()
        if solve_inline is not None:
            if solved_for is not frame:
                sol, _ = solve_inline(frame.left_pose, frame.right_pose, home, np.zeros(14))
                solved_for = frame
        else:
            if solved_for is not frame:
                worker.submit(frame)
                solved_for = frame
            latest = worker.latest()
            sol = None if latest is None else latest[1]
        if sol is not None and shaper.shape(sol, now) is not None:
            sent += 1
        tick_ms.append((time.perf_counter() - t0) * 1000.0)
        next_t += dt
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.monotonic()
    return tick_ms, period_ms, sent


def tick_report(seconds):
    from ik_worker import IkProcess, IkWorker, build_arm_ik, set_openmp_threads
    ik, home, target = build_ik()
    model = ik.reduced_robot.model
    bench_sequential(ik, target, 5)
    log = types.SimpleNamespace(exception=lambda *a, **k: print("worker:", *a), warning=print,
                                info=print)
    rows = []

    def run(label, **kw):
        tick_ms, period_ms, sent = tick_loop(seconds, target, make_shaper(model), home, **kw)
        rows.append((label, tick_ms, period_ms, sent))

    run("inline, OpenMP default (before)", solve_inline=ik.solve_ik)
    for threads, label in ((None, "IkWorker+IkProcess, OpenMP default"),
                           (1, "IkWorker+IkProcess, OpenMP 1 (after)")):
        proc = IkProcess(build_arm_ik, *urdf(), threads)
        proc.wait_ready(120.0)
        worker = IkWorker(proc.solve, lambda: (home, np.zeros(14)), log)
        worker.start()
        try:
            run(label, worker=worker)
            p50, mx = worker.stats(time.monotonic())
            rows[-1] += ("solve p50 %.2f / max %.2f ms (last 1 s)" % (p50, mx),)
        finally:
            worker.stop()
            proc.close()
    set_openmp_threads(1)
    bench_sequential(ik, target, 5)
    run("inline, OpenMP 1 (for reference)", solve_inline=ik.solve_ik)
    for row in rows:
        label, tick_ms, period_ms, sent = row[:4]
        print("[%s]%s" % (label, "  " + row[4] if len(row) > 4 else ""))
        print("    tick:   " + stats(tick_ms))
        print("    period: " + stats(period_ms) + "  -> %.1f Hz, arms sent on %d/%d ticks"
              % (1000.0 / np.mean(period_ms), sent, len(tick_ms)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=50, help="solves per measurement")
    ap.add_argument("--tick", action="store_true", help="the 50 Hz tick benchmark")
    ap.add_argument("--seconds", type=float, default=10.0, help="--tick: seconds per variant")
    args = ap.parse_args(argv)
    paths.add_teleop()
    paths.enter_htd_robot_control()
    if args.tick:
        tick_report(args.seconds)
    else:
        solve_report(args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
