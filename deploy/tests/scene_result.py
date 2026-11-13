"""Judge a sim_scene_rollout.sh run from its object log: did the vessel go from the tray into the bin?

    python deploy/tests/scene_result.py --log deploy/tests/logs/scene_<time> --scene g1_walk/scene_sim/out/NAME

Reads ``objects.jsonl`` (the free objects' world poses at 10 Hz, MuJoCo frame) and ``scene_frame.json`` (where the
objects started and how large they are) and writes ``result.json``:

* ``lifted``: the vessel rose at least 3 cm above where it started (it left the tray's floor);
* ``max_lift_m``, ``max_travel_m``: the most it rose and the farthest it got from its start;
* ``in_bin``: at the end its centre is inside the bin's footprint (the bin's box, at the bin's final pose, shrunk
  by 1 cm), within the bin's height, and it has come to rest (under 1 cm of travel in the last second);
* ``upright``: its axis is within 30 degrees of vertical at the end;
* ``bin_moved_m``: how far the bin itself was pushed.

``success`` = ``in_bin``. The vessel is the lightest free object and the bin the heaviest (override with --vessel /
--bin).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def R_from_quat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--t_start", type=float, default=0.0, help="sim time the policy was started at")
    ap.add_argument("--vessel", default=None)
    ap.add_argument("--bin", default=None)
    a = ap.parse_args(argv)

    log = Path(a.log)
    frame = json.loads((Path(a.scene) / "scene_frame.json").read_text())
    free = {k: v for k, v in frame["objects"].items() if not v["static"]}
    vessel = a.vessel or min(free, key=lambda k: free[k]["mass"])
    bin_ = a.bin or max(free, key=lambda k: free[k]["mass"])
    rows = [json.loads(line) for line in (log / "objects.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["t"] >= a.t_start] or rows
    t = np.array([r["t"] for r in rows])
    v = np.array([r[vessel] for r in rows])
    b = np.array([r[bin_] for r in rows])
    lift = v[:, 2] - v[0, 2]
    travel = np.linalg.norm(v[:, :3] - v[0, :3], axis=1)
    # the vessel's centre in the bin's frame at the end (the bin's object frame: box centred, x along its long side)
    Rb, pb = R_from_quat(b[-1, 3:]), b[-1, :3]
    half_h = free[vessel]["extent"][2] / 2
    centre = v[-1, :3] + R_from_quat(v[-1, 3:]) @ np.array([0, 0, half_h])  # the vessel's frame sits on its base
    local = Rb.T @ (centre - pb)
    ext = np.array(free[bin_]["extent"])
    inside = bool(np.all(np.abs(local[:2]) < ext[:2] / 2 - 0.01) and abs(local[2]) < ext[2] / 2 + half_h)
    last = t >= t[-1] - 1.0
    at_rest = bool(np.linalg.norm(v[last, :3] - v[-1, :3], axis=1).max() < 0.01)
    tilt = float(np.degrees(np.arccos(np.clip(R_from_quat(v[-1, 3:])[2, 2], -1, 1))))
    res = {"vessel": vessel, "bin": bin_, "seconds": round(float(t[-1] - t[0]), 1),
           "lifted": bool(lift.max() > 0.03), "max_lift_m": round(float(lift.max()), 3),
           "max_travel_m": round(float(travel.max()), 3), "final_travel_m": round(float(travel[-1]), 3),
           "final_height_change_m": round(float(lift[-1]), 3),
           "vessel_in_bin_frame_m": np.round(local, 3).tolist(), "in_bin": inside and at_rest, "at_rest": at_rest,
           "upright": tilt < 30.0, "tilt_deg": round(tilt, 1),
           "bin_moved_m": round(float(np.linalg.norm(b[-1, :3] - b[0, :3])), 3)}
    res["success"] = res["in_bin"]
    (log / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
