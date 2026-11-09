"""The arm IK off the control thread.

``G1_29_ArmIK.solve_ik`` (CasADi + IPOPT) holds the GIL for the whole solve
(g1_walk/tests/bench_ik.py), so a thread alone would still stall the 50 Hz
tick. The solver runs in a child process (:class:`IkProcess`, spawn,
the IK built in the child); :class:`IkWorker` is the controller's side: a
one-slot mailbox of policy frames and a thread that measures the arms, calls
``solve`` and keeps the latest solution. While that thread waits on the pipe
the GIL is free.
"""
import ctypes
import multiprocessing
import os
import signal
import threading
import time

import numpy as np

from deploy.controller_bridge import TickTimer


class IkWorker:
    """``solve(left_pose, right_pose, q_meas, dq_meas) -> (q_sol, info)``;
    ``measure() -> (q_meas, dq_meas)``, read just before each solve."""

    def __init__(self, solve, measure, log, clock=time.monotonic, window_s=1.0):
        self._solve, self._measure, self._log, self._clock = solve, measure, log, clock
        self._cond = threading.Condition()
        self._pending = None
        self._latest = None
        self._gen = 0                   # bumped by clear(); a solve of an older generation is dropped
        self._owed_since = None         # a frame is waiting for a successful solve since then
        self._stop = threading.Event()
        self._thread = None
        self._said = set()
        self.timer = TickTimer(window_s)

    def submit(self, frame):
        """A newer frame replaces one not yet picked up."""
        with self._cond:
            self._pending = frame
            if self._owed_since is None:
                self._owed_since = self._clock()
            self._cond.notify()

    def latest(self):
        """``(frame, q_sol, t_solved)`` of the last successful solve, or None."""
        with self._cond:
            return self._latest

    def owed_for(self, now):
        """Seconds a submitted frame has waited for a successful solve (0 when none is owed)."""
        with self._cond:
            return 0.0 if self._owed_since is None else max(0.0, now - self._owed_since)

    def clear(self):
        """Forget the pending frame and the latest solution (a session start)."""
        with self._cond:
            self._gen += 1
            self._pending = self._latest = self._owed_since = None

    def stats(self, now):
        """(p50, max) solve ms over the timer's window."""
        return self.timer.stats(now)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="ik_worker", daemon=True)
        self._thread.start()

    def stop(self, timeout=1.0):
        """True once the thread has exited (a solve in flight finishes first)."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self):
        while True:
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait()
                if self._stop.is_set():
                    return
                frame, gen = self._pending, self._gen
                self._pending = None
            try:
                q_meas, dq_meas = self._measure()
                t0 = time.perf_counter()
                q_sol, _ = self._solve(frame.left_pose, frame.right_pose, q_meas, dq_meas)
                took = time.perf_counter() - t0
                q_sol = np.asarray(q_sol, float).copy()
            except Exception:
                if "solve" not in self._said:
                    self._said.add("solve")
                    self._log.exception("[g1_walk] IK solve failed; the previous solution "
                                        "stands (logged once)")
                continue
            now = self._clock()
            self.timer.add(took, now)
            with self._cond:
                if gen != self._gen:
                    continue
                self._latest = (frame, q_sol, now)
                self._owed_since = None if self._pending is None else now


def set_openmp_threads(n):
    """IPOPT's MUMPS runs on libgomp: on the 20-thread laptop its default pool
    makes a solve ~13 ms against ~2 ms on one thread (bench_ik.py)."""
    try:
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(int(n))
    except (OSError, AttributeError):
        pass


def build_arm_ik(urdf_path, urdf_dir, omp_threads=1):
    """HTD's ``G1_29_ArmIK``, built as the controller builds it (runs in the
    child); ``omp_threads`` None leaves OpenMP's default."""
    if omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(int(omp_threads))     # read when libgomp loads
    import paths
    paths.enter_htd_robot_control()
    from robot_arm_ik_g1 import G1_29_ArmIK
    ik = G1_29_ArmIK(urdf_path=urdf_path, urdf_dir=urdf_dir)
    if omp_threads is not None:
        set_openmp_threads(omp_threads)                         # if it was already loaded
    return ik


def _child_main(conn, factory, args):
    # Ctrl-C and a closed terminal reach the whole process group; the parent
    # decides when this process ends (close(), or the pipe's EOF if it dies).
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        ik = factory(*args)
    except BaseException as e:
        conn.send(("failed", repr(e)))
        return
    conn.send(("ready", None))
    while True:
        try:
            msg = conn.recv()
            while conn.poll():          # newest wins
                msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        fid, left, right, q, dq = msg
        try:
            q_sol, _ = ik.solve_ik(left, right, q, dq)
            reply = (fid, np.asarray(q_sol, float), None)
        except Exception as e:
            reply = (fid, None, repr(e))
        try:
            conn.send(reply)
        except (BrokenPipeError, OSError):
            return


class IkProcess:
    """``factory(*args)`` (a module-level function: it is pickled by name)
    builds the solver in a spawned child; :meth:`solve` is
    ``solve_ik`` across a Pipe. One request is in flight at a time; a reply
    to a request given up on (timeout) is dropped by its frame id."""

    def __init__(self, factory, *args, solve_timeout=1.0):
        ctx = multiprocessing.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_child_main, args=(child, factory, args),
                                 name="ik_solver", daemon=True)
        self._proc.start()
        child.close()
        self.solve_timeout = float(solve_timeout)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._id = 0

    @property
    def pid(self):
        return self._proc.pid

    def is_alive(self):
        return self._proc.is_alive()

    def wait_ready(self, timeout):
        """Raises unless the child built the solver within ``timeout`` s."""
        self._wait(time.monotonic() + timeout, "building the IK")
        try:
            tag, why = self._conn.recv()
        except EOFError:
            self._proc.join(1.0)
            raise RuntimeError("the IK process exited (code %s) while building the IK"
                               % self._proc.exitcode) from None
        if tag != "ready":
            raise RuntimeError("the IK process could not build the IK: %s" % why)

    def solve(self, left_pose, right_pose, q_meas, dq_meas):
        with self._lock:
            self._id += 1
            fid = self._id
            self._conn.send((fid, np.asarray(left_pose, float), np.asarray(right_pose, float),
                             np.asarray(q_meas, float), np.asarray(dq_meas, float)))
            deadline = time.monotonic() + self.solve_timeout
            while True:
                self._wait(deadline, "a solve")
                rid, q_sol, err = self._conn.recv()
                if rid != fid:
                    continue
                if err is not None:
                    raise RuntimeError("solve_ik raised in the IK process: %s" % err)
                return q_sol, None

    def _wait(self, deadline, what):
        while not self._conn.poll(0.05):
            if self._closed.is_set():
                raise RuntimeError("the IK process is closed")
            if not self._proc.is_alive() and not self._conn.poll(0):
                raise RuntimeError("the IK process exited (code %s) during %s"
                                   % (self._proc.exitcode, what))
            if time.monotonic() > deadline:
                raise TimeoutError("the IK process did not answer %s in time" % what)

    def close(self, timeout=1.0):
        self._closed.set()
        # The polite stop only when no solve is mid-send; otherwise terminate.
        if self._lock.acquire(timeout=0.2):
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            finally:
                self._lock.release()
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout)
