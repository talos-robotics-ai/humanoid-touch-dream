#!/usr/bin/env python3
"""One view: HTD inference as the simulated robot, the operator reference as its ghost.

    python render_overlay.py --solid EP_policy_<lower>.npz --ghost EP_recorded_g1_walk.npz \
        --robot_mjcf assets/g1/g1_inspire.xml --head EP_head.mp4 --out EP_overlay.mp4 [--label TEXT]

Both inputs are ``rollout.py`` logs of the same episode with the same G1 model
(base pose, the 29 body joints and, for the Inspire G1, the 24 finger joints,
per 50 Hz tick). This replays them kinematically -- no physics -- in one
Genesis scene with two copies of the robot: ``--solid`` (the HTD policy's
rollout) drawn as the robot, ``--ghost`` (the operator's teleop commands on
g1_walk) drawn translucent. The ghost's base trajectory is moved rigidly so that
both robots stand at the same position and heading when the commands start; where
the two robots part after that is where the executed motions differ. The camera
rides at the solid robot's front-left. The real robot's head camera from the
recording (``--head``) is inset, delayed by the rollouts' settle time.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rollout import INSPIRE_MOTOR_COUPLING, follow, robot_only_mjcf  # noqa: E402

GHOST_COLOR = (0.35, 0.75, 1.0)
GHOST_OPACITY = 0.35


def finger_joint_names():
    """The joints of rollout.InspireFingers.dofs, in its order (the logs' finger_q columns)."""
    names = []
    for side in ("left", "right"):
        for driver, followers in INSPIRE_MOTOR_COUPLING:
            names.append("%s_%s_joint" % (side, driver))
            names += ["%s_%s_joint" % (side, f) for f, _ in followers]
    return names


def rpy_to_quat_wxyz(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def align_to(log, ref, k0):
    """``log``'s base trajectory moved rigidly in the plane so that at tick ``k0`` it has
    ``ref``'s position and heading: the two rollouts then start the commanded part of
    the episode from the same pose. (The walker's environment spawns at a seeded random
    heading, the WBC scene at heading 0, and the WBC drifts while it settles.)"""
    pos, rpy = log["base_pos"].copy(), log["base_rpy"].copy()
    k0 = min(k0, len(pos) - 1, len(ref["base_pos"]) - 1)
    d = float(ref["base_rpy"][k0][2] - rpy[k0][2])
    c, s = math.cos(d), math.sin(d)
    xy = pos[:, :2] - pos[k0, :2]
    pos[:, :2] = xy @ np.array([[c, s], [-s, c]]) + ref["base_pos"][k0][:2]
    rpy[:, 2] = rpy[:, 2] + d
    return pos, rpy


class Posed:
    """A robot entity posed from one rollout log (base trajectory optionally replaced)."""

    def __init__(self, robot, log, base=None):
        self.robot, self.log = robot, log
        self.base_pos, self.base_rpy = base if base is not None else (log["base_pos"], log["base_rpy"])
        names = [str(n) for n in log["joint_names"]]
        if "finger_q" in log.files:
            names += finger_joint_names()
            self.q = np.concatenate([log["q"], log["finger_q"]], axis=1)
        else:
            self.q = log["q"]
        # The entity's own dof indices: ``dof_start`` counts across the whole scene, and with
        # two robots in it the second one's would point past its own dofs.
        self.dofs = [robot.get_joint(n).dof_idx_local for n in names]
        self.n = len(self.q)

    def pose(self, k):
        k = min(k, self.n - 1)                       # a run that ended early holds its last pose
        self.robot.set_pos(self.base_pos[k])
        self.robot.set_quat(rpy_to_quat_wxyz(*self.base_rpy[k]))
        self.robot.set_dofs_position(self.q[k], self.dofs, zero_velocity=True)
        return self.base_pos[k], float(self.base_rpy[k][2])


def render(args):
    import genesis as gs
    solid_log, ghost_log = np.load(args.solid), np.load(args.ghost)
    dt = float(solid_log["dt"])
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt), show_viewer=False,
                     vis_options=gs.options.VisOptions(rendered_envs_idx=[0]))
    scene.add_entity(gs.morphs.Plane())
    mjcf = robot_only_mjcf(os.path.abspath(args.robot_mjcf))
    solid = scene.add_entity(gs.morphs.MJCF(file=mjcf))
    ghost = scene.add_entity(gs.morphs.MJCF(file=mjcf),
                             surface=gs.surfaces.Default(color=GHOST_COLOR, opacity=GHOST_OPACITY))
    cam = scene.add_camera(res=(1280, 720), pos=(2.5, -2.5, 1.8), lookat=(0.0, 0.0, 0.7), fov=40, GUI=False)
    scene.build()
    k0 = int(round(float(solid_log["settle_s"]) / dt))       # the first commanded tick
    s, g = Posed(solid, solid_log), Posed(ghost, ghost_log, align_to(ghost_log, solid_log, k0))
    every = max(1, int(round(1.0 / (dt * args.fps))))
    cam.start_recording()
    for k in range(0, max(s.n, g.n), every):
        g.pose(k)
        pos, yaw = s.pose(k)
        scene.visualizer.update(force=True)
        follow(cam, pos, yaw)
        cam.render()
    raw = os.path.join(tempfile.mkdtemp(), "overlay_raw.mp4")
    cam.stop_recording(save_to_filename=raw, fps=int(round(1.0 / (dt * every))))
    return raw, float(solid_log["settle_s"])


def text(s):
    return s.replace(":", r"\:").replace("'", r"\'")


def compose(raw, head, settle_s, out, label, fps):
    """Labels, a legend and the head-camera inset over the overlay render."""
    draw = "drawtext=text='%s':x=%s:y=%s:fontsize=%d:fontcolor=%s:box=1:boxcolor=black@0.55:boxborderw=6"
    fc = ("[0:v]fps=%d,%s,%s,%s[v0];" % (fps, draw % (text(label), "16", "14", 28, "white"),
                                         draw % (text("solid: HTD inference (simulated robot)"), "16", "56", 22, "white"),
                                         draw % (text("ghost: operator teleop reference"), "16", "90", 22, "0x66C0FF"))
          + "[1:v]fps=%d,scale=360:270,tpad=start_mode=clone:start_duration=%.3f,"
            "drawtext=text='%s':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55[c];"
          % (fps, settle_s, text("real robot head camera"))
          + "[v0][c]overlay=W-w-14:H-h-14,format=yuv420p[v]")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", raw],
                               capture_output=True, text=True, check=True).stdout)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-i", head, "-filter_complex", fc,
                    "-map", "[v]", "-t", "%.2f" % dur, "-c:v", "libx264", "-crf", "23", out], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--solid", required=True, help="rollout.py log drawn as the robot (HTD inference)")
    ap.add_argument("--ghost", required=True, help="rollout.py log drawn as the ghost (operator reference)")
    ap.add_argument("--robot_mjcf", required=True, help="the G1 model both rollouts used")
    ap.add_argument("--head", required=True, help="the recording's head-camera video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="HTD policy -> g1_walk (Genesis)")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args(argv)
    raw, settle_s = render(args)
    compose(raw, args.head, settle_s, os.path.abspath(args.out), args.label, args.fps)
    print("[overlay] %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
