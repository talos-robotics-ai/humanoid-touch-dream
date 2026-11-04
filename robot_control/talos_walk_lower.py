#!/usr/bin/env python3
"""talos_walk as the G1's lower-body controller.

The walking policy lives in ``talos_walk/`` next to this file: the deploy class
(``talos_walk/deploy/talos_walk_policy.py``), the run it loads
(``talos_walk/logs/g1/g1_teleop_anyadapter``), and the reference modules it
rebuilds the gait from (``talos_walk/talos``, ``data``, ``gait_libraries``).
See ``talos_walk/README.md`` for the policy itself.

``TalosWalkLowerBody`` is the host around that class. It drives the 12 leg and
3 waist joints (motor slots 0..14). The arms stay on this repository's IK: the
policy only observes the IK arm target as its arm command (``amo_arm``), and
its own arm outputs are discarded. Around the deploy class it adds:

* the arm command the policy observes, clipped to the box the run was trained
  across (``amo.arm_bounds`` in the run's ``cfgs.pkl``) and slewed at the
  trained arm rate. The deploy class's own box is wider than training on the
  inward shoulder roll, so it is not used;
* the torso attitude and height command, clipped to the walking envelope the
  run was trained on and slewed at 0.4 rad/s and 0.1 m/s;
* a first-order filter on the leg and waist PD targets,
  ``f = f + alpha * (a - f)`` with ``alpha = 0.3``. The network still reads
  its own unfiltered outputs in its last-action slot, as in training. This
  filter is what removed the stand-still chatter on the real G1;
* a handover blend from the stance to the policy's targets with the gait
  clock frozen, so authority crosses over at phase 0;
* a tilt abort and a non-finite-action abort, both raised as
  ``TalosWalkFault`` for the caller to damp the robot.

``VelocityFromPosition`` differentiates joint position into velocity. The G1's
reported ``dq`` for waist roll and waist pitch (a two-motor linkage) has been
seen to disagree with their motion, so those two velocities are computed from
position wherever ``rt/lowstate`` arrives.

Height convention. HTD's ``TeleopDataWB.torso_height`` stays an absolute
command around 0.75 m. The policy takes an offset from its nominal pelvis
height, so ``amo_h = torso_height - NOMINAL_TORSO_HEIGHT``. With the rollout's
``torso_height = body_pose[2] + 0.75``, the recorded ``body_pose[2]`` is the
policy's height offset itself.
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TALOS_WALK_ROOT = os.path.join(current_dir, "talos_walk")
DEFAULT_TALOS_WALK_RUN = "g1/g1_teleop_anyadapter"
DEFAULT_TALOS_WALK_CHECKPOINT = 9999

#: Legs (12) + waist (3): the motor slots the walker drives.
N_LOWER = 15
#: Arm slots after the lower block, left arm then right.
N_ARM = 14
#: HTD's torso height command at the stand. ``amo_h = torso_height - this``.
NOMINAL_TORSO_HEIGHT = 0.75
#: Motor slots of waist roll and waist pitch.
WAIST_ROLL_PITCH_MOTORS = (13, 14)

#: The G1 29-DoF joint order, which is also the Unitree motor order 0..28.
G1_29_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

_MODULE_NAME = "talos_walk_policy"


class TalosWalkFault(RuntimeError):
    """The walker cannot continue; the caller must damp the robot."""


def load_deploy_module(root=DEFAULT_TALOS_WALK_ROOT):
    """The deploy class's module, imported by path from ``root``."""
    path = os.path.join(root, "deploy", "talos_walk_policy.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found: --talos_walk_root must be a talos_walk workspace "
            "(deploy/, talos/, data/, gait_libraries/, logs/)")
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        module = importlib.util.module_from_spec(spec)
        # Registered before it runs: its dataclasses look their module up by name.
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module


def quat_wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]])


def tilt_from_quat_wxyz(q):
    """Angle between the body z axis and world z, in rad."""
    w, x, y, z = (float(v) for v in q)
    n2 = w * w + x * x + y * y + z * z
    if n2 <= 0.0:
        return 0.0
    c = 1.0 - 2.0 * (x * x + y * y) / n2
    return float(math.acos(float(np.clip(c, -1.0, 1.0))))


def _slew(have, want, step):
    return have + np.clip(want - have, -step, step)


class _EnvData:
    """The measured fields the deploy class reads. No ``base_pos``: the pelvis
    height is then estimated from attitude and joint angles, as on the robot."""

    def __init__(self, dof_pos, dof_vel, base_quat, base_ang_vel):
        self.dof_pos = np.asarray(dof_pos, dtype=np.float64)
        self.dof_vel = np.asarray(dof_vel, dtype=np.float64)
        self.base_quat = np.asarray(base_quat, dtype=np.float64)  # xyzw
        self.base_ang_vel = np.asarray(base_ang_vel, dtype=np.float64)

    def get(self, key, default=None):
        return getattr(self, key, default)


class VelocityFromPosition:
    """Velocity of a few joints differentiated from position and low-passed.

    ``update`` is called on every ``rt/lowstate`` message (about 500 Hz to
    1 kHz); a duplicate message or a gap of more than 50 ms restarts the
    derivative instead of producing a spike.
    """

    def __init__(self, n, cutoff_hz=40.0):
        self.cutoff_hz = float(cutoff_hz)
        self._prev = None
        self._vel = np.zeros(int(n))

    def update(self, q, now=None):
        now = time.monotonic() if now is None else float(now)
        q = np.asarray(q, dtype=np.float64)
        prev, self._prev = self._prev, (now, q)
        if prev is None:
            return
        dt = now - prev[0]
        if dt <= 1e-4 or dt > 0.05:
            return
        raw = (q - prev[1]) / dt
        a = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * dt)
        # A new array each time, so a reader on another thread never sees a
        # half-written one.
        self._vel = self._vel + a * (raw - self._vel)

    @property
    def value(self):
        return self._vel.copy()


class TalosWalkLowerBody:
    """The talos_walk run driving the G1's legs and waist at its trained 50 Hz."""

    def __init__(self, root=DEFAULT_TALOS_WALK_ROOT, run=DEFAULT_TALOS_WALK_RUN,
                 checkpoint=DEFAULT_TALOS_WALK_CHECKPOINT, *, filter_alpha=0.3,
                 torso_rate=0.4, height_rate=0.1, blend_seconds=5.0,
                 tilt_abort_rad=1.0, logger=None):
        import torch

        # A batch-size-1 network ticked at 50 Hz misses its period on a
        # multi-thread pool: the thread hand-offs cost more than the work.
        torch.set_num_threads(1)
        self.log = logger
        root = os.path.abspath(root)
        module = load_deploy_module(root)
        # Robot settings: no extra action latency (the real delay supplies it)
        # and the world-model innovation loop off (it made the legs chatter at
        # a stand and did not change walking).
        self.policy = module.TalosWalkPolicy(
            root, run, int(checkpoint), device="cpu",
            action_latency_steps=0, world_model=False)
        p = self.policy
        if tuple(p.joint_names) != G1_29_JOINT_NAMES:
            raise ValueError(f"{run} is not in the G1 29-DoF joint order")
        if (p.n_lower, p.n_upper, p.aux_idx.size) != (N_LOWER, N_ARM, 0):
            raise ValueError(
                f"{run} splits its joints {p.n_lower}/{p.n_upper}/{p.aux_idx.size}, "
                f"expected {N_LOWER} lower / {N_ARM} arm / no head")
        self.run, self.checkpoint = run, p.checkpoint
        self.dt = float(p.dt)
        self.default_pos = np.asarray(p.default_pos, dtype=np.float64)
        self.kp = np.asarray(p.facts.kp, dtype=np.float64)[:N_LOWER]
        self.kd = np.asarray(p.facts.kd, dtype=np.float64)[:N_LOWER]

        # The arm box the run's arm command was sampled across in training.
        bounds = p.env_cfg.get("amo", {}).get("arm_bounds")
        if bounds:
            lo, hi = (np.asarray(b, dtype=np.float64) for b in bounds)
            if lo.shape != (N_ARM,) or hi.shape != (N_ARM,) or np.any(lo > hi):
                raise ValueError("amo.arm_bounds is not a box over the 14 arm joints")
        else:
            lo, hi = np.asarray(p.facts.arm_lo), np.asarray(p.facts.arm_hi)
        self.arm_lo, self.arm_hi = lo, hi
        self.arm_rate = float(p.facts.arm_rate)

        box = p.facts.command_box  # the walking envelope, narrower than standing
        self.rpy_lo = np.array([box[k][0] for k in ("roll", "pitch", "yaw")])
        self.rpy_hi = np.array([box[k][1] for k in ("roll", "pitch", "yaw")])
        self.h_lo, self.h_hi = (float(v) for v in box["height"])
        self.vx_range = tuple(float(v) for v in p.vx_range)
        self.wz_range = tuple(float(v) for v in p.wz_range)

        if not 0.0 < float(filter_alpha) <= 1.0:
            raise ValueError("filter_alpha must lie in (0, 1]; 1 turns the filter off")
        self.filter_alpha = float(filter_alpha)
        self.torso_rate, self.height_rate = float(torso_rate), float(height_rate)
        self.blend_ticks = max(1, int(round(float(blend_seconds) / self.dt)))
        self.tilt_abort_rad = float(tilt_abort_rad)
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def reset(self):
        """Back to a fresh run: gait clock at zero, commands at rest. The next
        ``step`` needs a new ``begin_blend`` first."""
        self.policy.reset()
        self._filtered = None
        self._arm_seen = None
        self._rpy = np.zeros(3)
        self._h = 0.0
        self._blend_from = None
        self._blend_left = 0
        self.ticks = 0

    def stance(self):
        """Leg and waist pose the policy holds at a zero command and rest arms:
        where the robot should be before it takes over."""
        return np.asarray(self.policy.get_init_dof_pos(), dtype=np.float64)[:N_LOWER]

    def begin_blend(self, from_q):
        """Cross from ``from_q`` (the leg/waist targets on the wire) to the
        policy over ``blend_seconds``, with the gait clock frozen."""
        self._blend_from = np.asarray(from_q, dtype=np.float64)[:N_LOWER].copy()
        self._blend_left = self.blend_ticks

    @property
    def blending(self):
        return self._blend_left > 0

    @property
    def applied_command(self):
        """``(vx, wz)`` after the policy's own clamp and ramp."""
        return float(self.policy.vx), float(self.policy.wz)

    # ----------------------------------------------------------------- tick
    def step(self, dof_pos, dof_vel, imu_quat_wxyz, imu_gyro, arm_q,
             torso_rpy=(0.0, 0.0, 0.0), height_offset=0.0, vx=0.0, wz=0.0):
        """One tick: the 15 absolute PD targets for motor slots 0..14.

        ``dof_pos`` / ``dof_vel`` are all 29 joints in motor order,
        ``imu_quat_wxyz`` / ``imu_gyro`` the pelvis IMU as Unitree reports it,
        ``arm_q`` the 14 arm targets on the wire (left arm, then right),
        ``height_offset`` the policy's height command (m from nominal).
        Raises ``TalosWalkFault`` on a fall or a non-finite action.
        """
        if self._blend_from is None:
            raise RuntimeError("call begin_blend() with the targets on the wire first")
        p = self.policy
        tilt = tilt_from_quat_wxyz(imu_quat_wxyz)
        if tilt > self.tilt_abort_rad:
            raise TalosWalkFault(
                f"tilt {math.degrees(tilt):.0f} deg exceeds the "
                f"{math.degrees(self.tilt_abort_rad):.0f} deg abort")

        # Arm command as the policy sees it: trained box, trained rate.
        arm = np.clip(np.asarray(arm_q, dtype=np.float64), self.arm_lo, self.arm_hi)
        if self._arm_seen is not None and self.arm_rate > 0.0:
            arm = _slew(self._arm_seen, arm, self.arm_rate * self.dt)
        self._arm_seen = arm.copy()
        p.amo_arm = arm

        # Torso attitude and height: walking envelope, slider rates.
        rpy = np.clip(np.asarray(torso_rpy, dtype=np.float64), self.rpy_lo, self.rpy_hi)
        self._rpy = _slew(self._rpy, rpy, self.torso_rate * self.dt)
        h = float(np.clip(float(height_offset), self.h_lo, self.h_hi))
        self._h = float(_slew(self._h, h, self.height_rate * self.dt))
        p.amo_rpy, p.amo_h = self._rpy.copy(), self._h

        # Walk command as stick fractions of the deployed bounds; the deploy
        # class clamps it to its trained box and ramps it. No walking while the
        # policy is blending in.
        if self.blending:
            vx = wz = 0.0
        ctrl = {"JoystickCtrl": {"axes": {
            "LeftY": float(vx) / self.vx_range[1],
            "RightX": -float(wz) / self.wz_range[1]}}}

        env = _EnvData(dof_pos, dof_vel, quat_wxyz_to_xyzw(imu_quat_wxyz), imu_gyro)
        obs, _ = p.get_observation(env, ctrl)
        action = np.asarray(p.get_action(obs), dtype=np.float64)
        if not np.all(np.isfinite(action)):
            raise TalosWalkFault("the walking policy produced a non-finite target")

        lower = action[:N_LOWER]
        if self.filter_alpha < 1.0:
            if self._filtered is None:
                self._filtered = lower.copy()
            else:
                self._filtered = self._filtered + self.filter_alpha * (lower - self._filtered)
            lower = self._filtered
        target = self.default_pos[:N_LOWER] + lower

        if self.blending:
            done = 1.0 - (self._blend_left - 1) / self.blend_ticks
            s = done * done * (3.0 - 2.0 * done)  # smoothstep
            target = (1.0 - s) * self._blend_from + s * target
            self._blend_left -= 1
        else:
            p.post_step_callback()  # the gait clock runs only once live
        self.ticks += 1
        return target

    def warm_up(self, dof_pos, dof_vel, imu_quat_wxyz, imu_gyro, n=20):
        """Tick the network ``n`` times on a fixed state so the first live tick
        is not slow, then reset."""
        arm = np.clip(np.asarray(dof_pos, dtype=np.float64)[N_LOWER:], self.arm_lo, self.arm_hi)
        self.begin_blend(np.asarray(dof_pos, dtype=np.float64)[:N_LOWER])
        for _ in range(int(n)):
            self.step(dof_pos, dof_vel, imu_quat_wxyz, imu_gyro, arm)
        self.reset()

    def describe(self):
        return (f"talos_walk {self.run} @ {self.checkpoint}: {1.0 / self.dt:.0f} Hz, "
                f"vx {self.vx_range}, wz {self.wz_range}, height offset "
                f"[{self.h_lo}, {self.h_hi}] m, target filter {self.filter_alpha}, "
                f"blend {self.blend_ticks * self.dt:.1f} s, tilt abort "
                f"{math.degrees(self.tilt_abort_rad):.0f} deg")


def _offline_check(root):
    """Load the run and tick it on a fixed standing state, without a robot:
    the files are complete, the network runs, and a tick fits the 20 ms period."""
    lb = TalosWalkLowerBody(root)
    print(lb.describe())
    stance = lb.stance()
    q, dq = np.concatenate([stance, np.zeros(N_ARM)]), np.zeros(N_LOWER + N_ARM)
    quat, gyro = [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    lb.warm_up(q, dq, quat, gyro)
    lb.begin_blend(stance)
    times, ok = [], True
    for k in range(lb.blend_ticks + 100):
        t0 = time.perf_counter()
        target = lb.step(q, dq, quat, gyro, np.zeros(N_ARM), vx=0.3 if k > lb.blend_ticks else 0.0)
        times.append(time.perf_counter() - t0)
        ok = ok and bool(np.all(np.isfinite(target)))
    print(f"stance (slots 0..14): {np.round(stance, 4).tolist()}")
    print(f"targets finite: {ok}; tick median {np.median(times) * 1e3:.2f} ms, "
          f"max {np.max(times) * 1e3:.2f} ms; applied (vx, wz) {lb.applied_command}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline check of the talos_walk lower body (no robot).")
    parser.add_argument("--talos_walk_root", default=DEFAULT_TALOS_WALK_ROOT)
    sys.exit(_offline_check(parser.parse_args().talos_walk_root))
