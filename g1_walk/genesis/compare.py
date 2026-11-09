#!/usr/bin/env python3
"""Compare Genesis rollouts of the held-out episodes with each other and with the real robot.

    python compare.py --runs DIR [--real DIR] [--out DIR]

``--runs`` holds ``rollout.py`` outputs (``<episode>_<source>_<lower>.npz``);
``--real`` the processed episodes' ``<episode>_real.npz`` (proprio and the raw
take's odometry: position 0:3, roll/pitch/yaw 6:9, 30 Hz). For each episode it
plots the base path, yaw, forward speed against the command and base pitch,
and writes summary.csv: per run whether and when it fell, path length, net
displacement, yaw change, and how far forward speed and yaw rate followed the
command (mean absolute error over the episode).
"""
import argparse
import csv
import glob
import math
import os

import numpy as np

REAL_FPS = 30.0
STYLE = {("policy", "g1_walk"): ("C3", "-", "HTD policy -> g1_walk"),
         ("policy", "htd_wbc"): ("C2", "-", "HTD policy -> HTD WBC"),
         ("recorded", "g1_walk"): ("C0", "--", "operator -> g1_walk"),
         ("recorded", "htd_wbc"): ("C1", "--", "operator -> HTD WBC"),
         ("policy", "htd_wbc_h-0.08"): ("C4", "-", "HTD policy -> HTD WBC, height at its 0.70 m neutral")}


def body_speed(t, xy, yaw, win=0.5):
    """Forward and yaw rates from positions over a centred window of ``win`` seconds."""
    dt = np.median(np.diff(t))
    h = max(1, int(round(win / dt / 2)))
    vx = np.full(len(t), np.nan)
    wz = np.full(len(t), np.nan)
    yu = np.unwrap(yaw)
    for i in range(h, len(t) - h):
        d = (xy[i + h] - xy[i - h]) / (t[i + h] - t[i - h])
        vx[i] = math.cos(yu[i]) * d[0] + math.sin(yu[i]) * d[1]
        wz[i] = (yu[i + h] - yu[i - h]) / (t[i + h] - t[i - h])
    return vx, wz


def planar(xy, yaw0):
    xy = xy - xy[0]
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    return xy @ np.array([[c, -s], [s, c]]).T


def load_run(path):
    z = np.load(path)
    t = z["t"] - float(z["settle_s"])
    m = t >= 0
    d = {"t": t[m], "xy": z["base_pos"][m, :2], "z": z["base_pos"][m, 2], "rpy": z["base_rpy"][m],
         "cmd_walk": z["cmd_walk"][m], "fall_t": (int(z["fall_tick"]) * float(z["dt"]) - float(z["settle_s"]))
         if int(z["fall_tick"]) >= 0 else None, "source": str(z["source"]), "lower": str(z["lower"])}
    off = float(z["wbc_height_offset"]) if "wbc_height_offset" in z.files else 0.0
    if d["lower"] == "htd_wbc" and off:
        d["lower"] = "htd_wbc_h%+.2f" % off
    return d


def load_real(path):
    z = np.load(path)
    od = z["odom_raw"]
    t = np.arange(len(od)) / REAL_FPS
    return {"t": t, "xy": od[:, :2], "rpy": od[:, 6:9]}


def metrics(r, cmd_vx=None, cmd_wz=None):
    p = planar(r["xy"], r["rpy"][0, 2])
    yaw = np.unwrap(r["rpy"][:, 2])
    out = {"path_len_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
           "net_disp_m": float(np.linalg.norm(p[-1])),
           "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
           "max_pitch_deg": float(np.degrees(np.abs(r["rpy"][:, 1]).max()))}
    vx, wz = body_speed(r["t"], r["xy"], r["rpy"][:, 2])
    if cmd_vx is not None:
        ok = np.isfinite(vx)
        out["vx_track_mae"] = float(np.abs(vx[ok] - cmd_vx[ok]).mean())
        out["wz_track_mae"] = float(np.abs(wz[ok] - cmd_wz[ok]).mean())
    return out, p, vx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--real", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = args.out or args.runs
    os.makedirs(out_dir, exist_ok=True)
    runs = {}
    for f in sorted(glob.glob(os.path.join(args.runs, "episode_*_*.npz"))):
        r = load_run(f)
        ep = os.path.basename(f).split("_" + r["source"] + "_")[0]
        runs.setdefault(ep, {})[(r["source"], r["lower"])] = r
    rows = []
    for ep in sorted(runs, key=lambda s: int(s.split("_")[1])):
        fig, ax = plt.subplots(2, 2, figsize=(14, 9))
        for key, r in sorted(runs[ep].items()):
            c, ls, label = STYLE[key]
            m, p, vx = metrics(r, r["cmd_walk"][:, 0], r["cmd_walk"][:, 1])
            rows.append({"episode": ep, "commands": key[0], "lower": key[1],
                         "fell_at_s": "" if r["fall_t"] is None else round(r["fall_t"], 1),
                         **{k: round(v, 3) for k, v in m.items()}})
            label = label + (" (FELL %.1f s)" % r["fall_t"] if r["fall_t"] is not None else "")
            ax[0, 0].plot(p[:, 0], p[:, 1], c=c, ls=ls, label=label)
            ax[0, 1].plot(r["t"], np.degrees(np.unwrap(r["rpy"][:, 2]) - r["rpy"][0, 2]), c=c, ls=ls, label=label)
            ax[1, 0].plot(r["t"], vx, c=c, ls=ls, lw=1.0, label=label)
            if key[1] == "g1_walk":
                ax[1, 0].plot(r["t"], r["cmd_walk"][:, 0], c=c, ls=":", lw=0.8, label="%s command" % key[0])
            ax[1, 1].plot(r["t"], np.degrees(r["rpy"][:, 1]), c=c, ls=ls, lw=1.0, label=label)
        real_path = os.path.join(args.real, ep + "_real.npz") if args.real else None
        if real_path and os.path.isfile(real_path):
            rr = load_real(real_path)
            m, p, vx = metrics(rr)
            rows.append({"episode": ep, "commands": "recorded", "lower": "real robot", "fell_at_s": "",
                         **{k: round(v, 3) for k, v in m.items()}})
            ax[0, 0].plot(p[:, 0], p[:, 1], c="k", ls="--", label="real robot")
            ax[0, 1].plot(rr["t"], np.degrees(np.unwrap(rr["rpy"][:, 2]) - rr["rpy"][0, 2]), c="k", ls="--",
                          label="real robot")
            ax[1, 0].plot(rr["t"], vx, c="k", ls="--", lw=1.0, label="real robot")
            ax[1, 1].plot(rr["t"], np.degrees(rr["rpy"][:, 1]), c="k", ls="--", lw=1.0, label="real robot")
        ax[0, 0].set_aspect("equal", adjustable="datalim")
        ax[0, 0].set(title="%s: base path (start heading = +x)" % ep, xlabel="x [m]", ylabel="y [m]")
        ax[0, 1].set(title="yaw change", xlabel="time [s]", ylabel="deg")
        ax[1, 0].set(title="forward speed (body frame) vs command", xlabel="time [s]", ylabel="m/s")
        ax[1, 1].set(title="base pitch", xlabel="time [s]", ylabel="deg")
        for a in ax.ravel():
            a.grid(alpha=0.3)
            a.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, ep + ".png"), dpi=85)
        plt.close(fig)
    keys = ["episode", "commands", "lower", "fell_at_s", "path_len_m", "net_disp_m", "yaw_change_deg",
            "max_pitch_deg", "vx_track_mae", "wz_track_mae"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    for r in rows:
        print("%-11s %-9s %-11s fell=%-5s path %5.2f m  net %5.2f m  yaw %6.1f deg  pitch<=%4.1f  vx_mae %s  wz_mae %s"
              % (r["episode"], r["commands"], r["lower"], r["fell_at_s"] or "-", r["path_len_m"], r["net_disp_m"],
                 r["yaw_change_deg"], r["max_pitch_deg"], r.get("vx_track_mae", "-"), r.get("wz_track_mae", "-")))
    print("wrote", os.path.join(out_dir, "summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
