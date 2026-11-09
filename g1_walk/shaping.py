"""The arm and torso commands between HTD's solution and the walker's link.

HTD's arm IK answers each policy frame with 14 joint targets; the teleop put
its own arm commands through three more stages before the wire, and the same
three are applied here so the walker sees commands of the kind it ran under
during the recording:

* the **engage glide** (teleop ``ArmPipeline._engaged_goal``): a session
  starts from the arms where they stand -- at home, or where the last session
  left them -- and eases onto the live solution by a smoothstep whose length
  is fixed on the first solution (``glide_seconds``);
* a **speed cap** on what the arms are commanded to do between two ticks;
* the teleop's **jump guard** (``JointJumpGuard``, teleop
  ``G1Backend._safely_set_arms``): a command further than the per-joint
  thresholds from the last one on the wire is refused and the baseline
  creeps toward it by ``JUMP_CREEP_RAD``.

The torso command gets the teleop's engage glide too
(``G1Backend._start_torso_engage``): from what the walker last applied to the
live target over the same ``engage_s``.

Between sessions the arms glide home (``G1Backend._start_arm_home``) and
hold there.
"""
import numpy as np


def smoothstep(s):
    s = min(1.0, max(0.0, float(s)))
    return s * s * (3.0 - 2.0 * s)


def glide_seconds(q_from, q_to, floor, peak_vel, cap):
    """teleop ``ArmPipeline.glide_seconds``: ``floor``, or longer when the
    largest joint travel would otherwise peak above ``peak_vel`` (the
    smoothstep peaks at 1.5x its mean rate), at most ``cap``."""
    travel = float(np.abs(np.asarray(q_to, float) - np.asarray(q_from, float)).max())
    if peak_vel <= 0.0 or not np.isfinite(travel):
        return float(floor)
    return float(min(cap, max(floor, 1.5 * travel / peak_vel)))


class Glide:
    """A smoothstep from ``q0`` toward a target over ``duration`` seconds
    from ``t0``; the target may move while it runs."""

    def __init__(self, q0, duration, t0):
        self.q0 = np.asarray(q0, float).copy()
        self.duration = max(float(duration), 1e-3)
        self.t0 = float(t0)

    def at(self, target, now):
        """``(command, done)`` at time ``now``."""
        s = (now - self.t0) / self.duration
        return self.q0 + smoothstep(s) * (np.asarray(target, float) - self.q0), s >= 1.0


class ArmShaper:
    """The engage glide, the speed cap and the jump guard, for one session
    at a time.

    ``guard`` is a teleop ``JointJumpGuard``; ``gravity(q)`` returns the 14
    gravity torques the command at ``q`` carries."""

    def __init__(self, guard, gravity, creep_rad, dt, max_vel, engage_s,
                 engage_vel, engage_max_s):
        self.guard = guard
        self.gravity = gravity
        self.creep_rad = float(creep_rad)
        self.max_step = float(max_vel) * float(dt)
        self.engage_s = float(engage_s)
        self.engage_vel = float(engage_vel)
        self.engage_max_s = float(engage_max_s)
        self.last = None           # last command on the wire (the guard's baseline)
        self._engage_from = None
        self._engage = None

    def begin(self, q_from):
        """A session starts with the arms commanded at ``q_from``: the next
        solutions are eased onto from there, and the baseline is cleared as
        the teleop's session start clears ``last_pd_target``."""
        self._engage_from = np.asarray(q_from, float).copy()
        self._engage = None
        self.last = None

    def shape(self, q_solution, now):
        """``(q, tau)`` to send for this solution, or None when the guard
        refused it (the baseline has crept toward it)."""
        q = np.asarray(q_solution, float)
        if self._engage_from is not None:
            if self._engage is None:
                dur = glide_seconds(self._engage_from, q, self.engage_s,
                                    self.engage_vel, self.engage_max_s)
                self._engage = Glide(self._engage_from, dur, now)
            q, done = self._engage.at(q, now)
            if done:
                self._engage_from = self._engage = None
        ref = self.last if self.last is not None else self._engage_ref()
        if ref is not None and self.max_step > 0.0 and np.all(np.isfinite(q)):
            q = ref + np.clip(q - ref, -self.max_step, self.max_step)
        tau = self.gravity(q)
        if self.guard.accept(q, tau, self.last):
            self.last = q.copy()
            return q, tau
        if self.last is not None and np.all(np.isfinite(q)):
            self.last = self.last + np.clip(q - self.last, -self.creep_rad, self.creep_rad)
        return None

    def _engage_ref(self):
        return None if self._engage is None else self._engage.q0


class TorsoEngage:
    """The torso half of a session start: from the command the walker last
    applied to the live target over ``engage_s`` (teleop
    ``G1Backend._torso_command``, its engage branch)."""

    def __init__(self, engage_s):
        self.engage_s = float(engage_s)
        self._glide = None

    def begin(self, applied, now):
        self._glide = Glide(applied, self.engage_s, now)

    def shape(self, target, now):
        if self._glide is None:
            return tuple(float(v) for v in target)
        q, done = self._glide.at(target, now)
        if done:
            self._glide = None
        return tuple(float(v) for v in q)
