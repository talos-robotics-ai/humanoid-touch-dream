"""Compare sim_scene_rollout.sh runs: what each policy did with its hand and the vessel, and a side-by-side video.

    python deploy/tests/compare_runs.py OUT.mp4 NAME=LOG_DIR [NAME=LOG_DIR ...]

Per run, from contacts.jsonl (hand pose and finger commands), touch.jsonl (finger contact forces), objects.jsonl and
result.json: when the arm leaves rest, when the fingers first press on something, how far the fingers are commanded
to close, a timeline every 5 s, and the outcome. OUT.mp4 is the runs' head cameras side by side from the start of
each run (OUT.jpg: five moments of it).
"""
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FINGERS = ("little", "ring", "middle", "index", "thumb")


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    out = Path(sys.argv[1])
    runs = [(a.split("=")[0], Path(a.split("=")[1])) for a in sys.argv[2:]]
    caps = []
    for name, L in runs:
        rows = [json.loads(x) for x in (L / "contacts.jsonl").read_text().splitlines()]
        obj = [json.loads(x) for x in (L / "objects.jsonl").read_text().splitlines()]
        tch = [json.loads(x) for x in (L / "touch.jsonl").read_text().splitlines()]
        res = json.loads((L / "result.json").read_text())
        vessel, bin_ = res["vessel"], res["bin"]
        b0, bin0 = np.array(obj[0][vessel][:3]), np.array(obj[0][bin_][:3])
        t = np.array([r["t"] for r in rows])
        hand = np.array([r["bodies"]["left_base_link"][:3] for r in rows])
        pelvis = np.array([r["bodies"]["pelvis"][:3] for r in rows]) if "pelvis" in rows[0]["bodies"] else None
        t_start = t[-1] - res["seconds"]
        run = t >= t_start
        moving = run & (np.linalg.norm(hand - hand[run][0], axis=1) > 0.03)
        cmd = np.array([r["left_hand"]["cmd"] for r in rows])
        tt, N = np.array([x["t"] for x in tch]), np.array([x["left"]["newton"] for x in tch])
        press = (tt >= t_start) & (N.sum(1) > 0.3)
        used = collections.Counter(FINGERS[k] for n in N[press] for k in range(5) if n[k] > 0.3)
        ot = np.array([o["t"] for o in obj])
        bz = np.array([o[vessel][2] for o in obj]) - b0[2]
        held = (ot >= t_start) & (bz > 0.02)
        print(f"== {name}: arm leaves rest at +{t[moving][0] - t_start if moving.any() else float('nan'):.1f} s; fingers "
              f"first press at +{tt[press][0] - t_start if press.any() else float('nan'):.1f} s ({press.sum() / 10:.1f} s "
              f"with finger force; {dict(used)}); vessel more than 2 cm above its start for {held.sum() / 10:.1f} s")
        print(f"   max finger closure commanded [little ring middle index thumb_bend thumb_rot] rad: {cmd[run].max(0)}")
        print(f"   result: lifted {res['lifted']} (max {res['max_lift_m']} m), travelled {res['max_travel_m']} m, ends "
              f"{np.round(np.array(obj[-1][vessel][:3]) - bin0, 2).tolist()} from the bin's centre, in_bin {res['in_bin']}")
        print("   t+   | hand - vessel start     | closure fingers/thumb | vessel - its start      | finger N | pelvis xy")
        for k in range(0, len(rows), 50):
            if t[k] < t_start:
                continue
            o = obj[int(np.argmin(np.abs(ot - t[k])))]
            n = N[int(np.argmin(np.abs(tt - t[k])))]
            print(f"   {t[k] - t_start:5.1f} {np.round(hand[k] - b0, 2)} {cmd[k, :4].mean():.2f} {cmd[k, 4]:.2f} "
                  f"{np.round(np.array(o[vessel][:3]) - b0, 2)} {n.sum():5.1f} "
                  f"{'' if pelvis is None else np.round(pelvis[k, :2], 2)}")
        if (L / "head_camera.mp4").exists():
            caps.append((name, cv2.VideoCapture(str(L / "head_camera.mp4"))))
    if not caps:
        return
    n = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for _, c in caps)
    fps = caps[0][1].get(cv2.CAP_PROP_FPS)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640 * len(caps), 480))
    still = []
    for i in range(n):
        fr = []
        for name, c in caps:
            ok, im = c.read()
            if not ok:
                im = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(im, name, (8, 24), 0, 0.7, (0, 255, 0), 2)
            fr.append(im)
        f = np.hstack(fr)
        vw.write(f)
        if i in [int(n * x) for x in (0.3, 0.45, 0.6, 0.75, 0.95)]:
            still.append(f)
    vw.release()
    cv2.imwrite(str(out.with_suffix(".jpg")), cv2.resize(np.vstack(still), None, fx=0.6, fy=0.6))


if __name__ == "__main__":
    main()
