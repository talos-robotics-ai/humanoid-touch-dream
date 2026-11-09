#!/usr/bin/env python3
"""Compare sim replays of one episode with each other and with the real recording.

    python plot_replay.py --policy out/episode_98_policy.npz --recorded out/episode_98_recorded.npz \
        [--real episode_98_real.npz] [--out out/episode_98.png]

--policy / --recorded are replay_commands.py logs of the same episode driven by
the HTD policy's command stream and by the operator's recorded commands. --real
holds what the robot measured during the recording (processed-episode proprio:
robot_joint_pos_proprio, left_eef_to_body_proprio, left_eef_joint_pos_proprio,
and the raw take's odometry as odom_raw). Plots the base path, base yaw, the left
wrist in the torso frame, left arm joints, left-hand closure and pelvis height,
and prints summary numbers.
"""
import argparse

import numpy as np

# Column layouts written by replay_commands.py.
RS_LEFT_EEF = slice(6, 9)          # robot_state: torso(6) | left eef->torso(6) | right(6) | q(29)
RS_Q = slice(18, 47)
EEF_LEFT_JOINTS = slice(6, 10)     # eef_state joint_positions: right 6 then left 6, joint angles in rad (0 = open); first 4 = fingers
LEFT_ARM = slice(15, 22)           # G1 motor order
ARM_NAMES = ("sh pitch", "sh roll", "sh yaw", "elbow", "wr roll", "wr pitch", "wr yaw")
HTD_FINGER_RANGE = 1.435           # rad, HTD's DFTP finger joint span (robot_hand_inspire_dftp.py)


def load_run(path):
    z = np.load(path)
    t0 = float(z["t_start"])
    run = {"name": str(z["source"])}
    for key in ("lowstate", "odom", "robot_state", "eef_state"):
        if key in z.files:
            run[key] = (z[key + "_t"] - t0, z[key])
    return run


def planar(xy, yaw):
    """Path relative to its first pose, rotated so the start heading is +x."""
    xy = xy - xy[0]
    c, s = np.cos(-yaw[0]), np.sin(-yaw[0])
    return xy @ np.array([[c, -s], [s, c]]).T


def unwrap_rel(yaw):
    yaw = np.unwrap(yaw)
    return yaw - yaw[0]


def resample(t, x, grid):
    return np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], -1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--recorded", required=True)
    ap.add_argument("--real", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [(load_run(args.policy), "C3", "sim: HTD policy + g1_walk"),
            (load_run(args.recorded), "C0", "sim: operator commands + g1_walk")]
    real = np.load(args.real) if args.real else None
    T = None
    if real is not None:
        T = len(real["robot_joint_pos_proprio"]) / args.fps
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    summary = {}

    # base path and yaw
    for run, c, label in runs:
        t, od = run["odom"]
        m = (t >= 0) & (T is None or t <= T) if T else t >= 0
        p = planar(od[m, :2], od[m, 5])
        ax[0, 0].plot(p[:, 0], p[:, 1], c=c, label=label)
        ax[0, 1].plot(t[m], np.degrees(unwrap_rel(od[m, 5])), c=c, label=label)
        summary[label] = {"path_end_m": p[-1].round(2).tolist(),
                          "path_len_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())}
    if real is not None:
        od = real["odom_raw"]
        tr = np.arange(len(od)) / args.fps
        p = planar(od[:, :2], od[:, 8])
        ax[0, 0].plot(p[:, 0], p[:, 1], c="k", ls="--", label="real robot (odometry)")
        ax[0, 1].plot(tr, np.degrees(unwrap_rel(od[:, 8])), c="k", ls="--", label="real robot")
        summary["real robot"] = {"path_end_m": p[-1].round(2).tolist(),
                                 "path_len_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())}
    ax[0, 0].set_aspect("equal", adjustable="datalim")
    ax[0, 0].set(title="base path (start heading = +x)", xlabel="x [m]", ylabel="y [m]")
    ax[0, 1].set(title="base yaw change", xlabel="time [s]", ylabel="deg")

    # left wrist in the torso frame, and pelvis height
    grid = np.arange(0, T if T else 1e9, 1 / args.fps)
    wrists = {}
    for run, c, label in runs:
        t, rs = run["robot_state"]
        g = grid[grid <= t.max()]
        w = resample(t, rs[:, RS_LEFT_EEF], g)
        wrists[label] = (g, w)
        for i, ls in enumerate(("-", "--", ":")):
            ax[1, 0].plot(g, 100 * w[:, i], c=c, ls=ls, lw=1.2, label=f"{label} {'xyz'[i]}" if i == 0 else None)
    if real is not None:
        w = real["left_eef_to_body_proprio"][:, :3]
        g = np.arange(len(w)) / args.fps
        for i, ls in enumerate(("-", "--", ":")):
            ax[1, 0].plot(g, 100 * w[:, i], c="k", ls=ls, lw=1.0, label="real robot" if i == 0 else None)
        for label, (gg, ww) in wrists.items():
            n = min(len(gg), len(w))
            summary[label]["left_wrist_vs_real_cm"] = float(100 * np.linalg.norm(ww[:n] - w[:n], axis=1).mean())
    (g1, w1), (g2, w2) = wrists[runs[0][2]], wrists[runs[1][2]]
    n = min(len(g1), len(g2))
    summary["policy vs operator (both sim)"] = {"left_wrist_mean_cm": float(100 * np.linalg.norm(w1[:n] - w2[:n], axis=1).mean())}
    ax[1, 0].set(title="left wrist in torso frame (x solid, y dashed, z dotted)", xlabel="time [s]", ylabel="cm")

    # left arm joints
    for run, c, label in runs:
        t, rs = run["robot_state"]
        m = t >= 0
        q = rs[m][:, RS_Q][:, LEFT_ARM]
        for j in (0, 3, 5):
            ax[1, 1].plot(t[m], q[:, j], c=c, lw=1.0, label=f"{label}" if j == 0 else None)
    if real is not None:
        q = real["robot_joint_pos_proprio"][:, LEFT_ARM]
        g = np.arange(len(q)) / args.fps
        for j in (0, 3, 5):
            ax[1, 1].plot(g, q[:, j], c="k", ls="--", lw=1.0, label="real robot" if j == 0 else None)
    ax[1, 1].set(title="left arm joints: shoulder pitch, elbow, wrist pitch", xlabel="time [s]", ylabel="rad")

    # left hand closure (0 open, 1 closed)
    for run, c, label in runs:
        t, es = run["eef_state"]
        m = t >= 0
        ax[2, 0].plot(t[m], es[m][:, EEF_LEFT_JOINTS].mean(1) / HTD_FINGER_RANGE, c=c, label=label)
    if real is not None:
        f = real["left_eef_joint_pos_proprio"][:, :4].mean(1) / HTD_FINGER_RANGE
        ax[2, 0].plot(np.arange(len(f)) / args.fps, f, c="k", ls="--", label="real robot")
    ax[2, 0].set(title="left hand closure (0 open, 1 closed)", xlabel="time [s]", ylabel="closure")

    # pelvis height (sim odometry z)
    for run, c, label in runs:
        t, od = run["odom"]
        m = t >= 0
        ax[2, 1].plot(t[m], od[m, 2], c=c, label=label)
        summary[label]["pelvis_min_m"] = float(od[m, 2].min())
    ax[2, 1].set(title="pelvis height (sim)", xlabel="time [s]", ylabel="m")

    for a in ax.ravel():
        a.legend(fontsize=7, loc="best")
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = args.out or args.policy.replace("_policy.npz", ".png")
    fig.savefig(out, dpi=90)
    print("wrote", out)
    for k, v in summary.items():
        print(f"{k:34s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
