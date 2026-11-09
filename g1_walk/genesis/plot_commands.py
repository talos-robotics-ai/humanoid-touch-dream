#!/usr/bin/env python3
"""Plot the operator's high-level references against what HTD commands, per held-out episode.

    python plot_commands.py --commands DIR [--inputs DIR] --out DIR

``--commands`` holds ``replay_eval.py --save_commands`` streams
(``<episode>_commands_<schedule>.npz``): ``gt_*`` is what the operator sent
during the recording, ``cmd_*`` what the HTD policy sends at each 30 Hz step on
the rollout schedule (async inference, crossfade, the walk command switched at
chunk boundaries), with the policy watching the recorded observations.
``--inputs`` (optional) holds ``walker_inputs.py`` files for both sources;
their ``hand_q01`` -- the Inspire commands after HTD's retargeting -- gives
the finger closure each source asks of the hands.

One figure per episode: left wrist position and orientation in the body frame,
right wrist position, finger closure, torso height and yaw, forward speed and
yaw rate. Black dashed: operator; red: HTD. Each panel's title carries the
mean absolute difference. Also writes summary.csv with those numbers.
"""
import argparse
import csv
import glob
import os

import numpy as np

OPERATOR, HTD = dict(c="k", ls="--", lw=1.1), dict(c="C3", ls="-", lw=1.2)
AXIS_COLORS = ("C0", "C2", "C1")


def triple(ax, t, op, htd, scale, names, unit, title):
    for i, name in enumerate(names):
        ax.plot(t, scale * op[:, i], c=AXIS_COLORS[i], ls="--", lw=1.0)
        ax.plot(t, scale * htd[:, i], c=AXIS_COLORS[i], ls="-", lw=1.3, label=name)
    err = float(np.nanmean(np.abs(scale * (htd - op))))
    ax.set(title="%s  (mean |HTD - operator| %.2f %s)" % (title, err, unit), ylabel=unit)
    ax.legend(fontsize=7, loc="upper right", title="dashed operator, solid HTD", title_fontsize=7)
    return err


def single(ax, t, op, htd, scale, unit, title, min_span=0.0):
    ax.plot(t, scale * op, label="operator", **OPERATOR)
    ax.plot(t, scale * htd, label="HTD", **HTD)
    lo, hi = np.nanmin(scale * np.r_[op, htd]), np.nanmax(scale * np.r_[op, htd])
    if hi - lo < min_span:                  # a channel that barely moves: no zoom into float noise
        mid = 0.5 * (lo + hi)
        ax.set_ylim(mid - min_span / 2, mid + min_span / 2)
    ax.ticklabel_format(axis="y", useOffset=False)
    err = float(np.nanmean(np.abs(scale * (htd - op))))
    ax.set(title="%s  (mean |HTD - operator| %.3f %s)" % (title, err, unit), ylabel=unit)
    ax.legend(fontsize=7, loc="upper right")
    return err


def closure(q01):
    """Inspire command -> closure of the four fingers (0 open, 1 closed), left and right."""
    return 1.0 - q01[:, 0:4].mean(1), 1.0 - q01[:, 6:10].mean(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commands", required=True)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(args.commands, "episode_*_commands_*.npz")),
                    key=lambda p: int(os.path.basename(p).split("_")[1])):
        ep = os.path.basename(f).split("_commands_")[0]
        z = np.load(f)
        fps = float(z["fps"])
        t = np.arange(len(z["gt_body_pose"])) / fps
        g = {k[3:]: z[k] for k in z.files if k.startswith("gt_")}
        c = {k[4:]: z[k] for k in z.files if k.startswith("cmd_")}
        fig, ax = plt.subplots(4, 2, figsize=(16, 15), sharex=True)
        e = {"episode": ep}
        e["left_wrist_pos_cm"] = triple(ax[0, 0], t, g["left_eef_to_body_pose"][:, :3],
                                        c["left_eef_to_body_pose"][:, :3], 100, "xyz", "cm",
                                        "left wrist position (body frame)")
        e["left_wrist_rot_deg"] = triple(ax[0, 1], t, g["left_eef_to_body_pose"][:, 3:6],
                                         c["left_eef_to_body_pose"][:, 3:6], 180 / np.pi,
                                         ("roll", "pitch", "yaw"), "deg", "left wrist orientation")
        e["right_wrist_pos_cm"] = triple(ax[1, 0], t, g["right_eef_to_body_pose"][:, :3],
                                         c["right_eef_to_body_pose"][:, :3], 100, "xyz", "cm",
                                         "right wrist position (body frame)")
        a = ax[1, 1]
        hand = {}
        if args.inputs:
            for src in ("recorded", "policy"):
                p = os.path.join(args.inputs, "%s_%s_inputs.npz" % (ep, src))
                if os.path.isfile(p):
                    zi = np.load(p)
                    th = np.arange(len(zi["hand_q01"])) / float(zi["control_hz"]) - float(zi["settle_s"])
                    hand[src] = (th, *closure(zi["hand_q01"]))
        if len(hand) == 2:
            for src, style, lab in (("recorded", OPERATOR, "operator"), ("policy", HTD, "HTD")):
                th, left, right = hand[src]
                a.plot(th, left, label="%s left" % lab, **style)
                a.plot(th, right, label="%s right" % lab, **{**style, "c": "C7" if src == "recorded" else "C1"})
            n = min(len(hand["recorded"][1]), len(hand["policy"][1]))
            e["left_closure"] = float(np.mean(np.abs(hand["policy"][1][:n] - hand["recorded"][1][:n])))
            a.set(title="finger closure commanded to the Inspire hands  (left: mean |diff| %.3f)"
                  % e["left_closure"], ylabel="0 open, 1 closed", xlim=(0, t[-1]))
            a.legend(fontsize=7, loc="upper right")
        else:
            a.set_visible(False)
        e["torso_height_cm"] = single(ax[2, 0], t, g["body_pose"][:, 2], c["body_pose"][:, 2], 100, "cm",
                                      "torso height command (offset)", min_span=10.0)
        e["torso_yaw_deg"] = single(ax[2, 1], t, g["body_pose"][:, 5], c["body_pose"][:, 5], 180 / np.pi, "deg",
                                    "torso yaw command", min_span=10.0)
        e["vx_mps"] = single(ax[3, 0], t, g["velocity_commands"][:, 0], c["velocity_commands"][:, 0], 1, "m/s",
                             "forward speed command vx", min_span=0.2)
        e["wz_radps"] = single(ax[3, 1], t, g["velocity_commands"][:, 2], c["velocity_commands"][:, 2], 1, "rad/s",
                               "yaw rate command wz", min_span=0.2)
        for a in ax.ravel():
            a.grid(alpha=0.3)
        for a in ax[3]:
            a.set_xlabel("time [s]")
        fig.suptitle("%s: operator teleop references vs HTD policy output (open loop, rollout schedule)" % ep,
                     fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(os.path.join(args.out, "%s_commands.png" % ep), dpi=80)
        plt.close(fig)
        rows.append(e)
        print("[plot] %s" % os.path.join(args.out, "%s_commands.png" % ep))
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("[plot] summary: %s" % os.path.join(args.out, "summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
