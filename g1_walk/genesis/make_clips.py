#!/usr/bin/env python3
"""Side-by-side clips: HTD inference on the left, the operator's teleop on the right.

    python make_clips.py --runs DIR --real DIR --out DIR [--left g1_walk|htd_wbc|htd_wbc_h-0.08]

For each held-out episode with both rollouts in ``--runs`` (``rollout.py``
outputs): the left panel is the HTD policy's commands on ``--left``, the right
panel the operator's recorded teleop commands on g1_walk (the walker the data
was recorded with), both in Genesis with the same following camera. The real
robot's head camera from the recording (``--real``/<episode>_head.mp4) is
inset in the right panel, delayed by the rollouts' settle time so all three
show the same moment of the episode. A panel that ends early (a fall) holds
its last frame. Needs ffmpeg.
"""
import argparse
import glob
import os
import subprocess

import numpy as np

LEFT_LABEL = {"g1_walk": "HTD policy -> g1_walk (Genesis)",
              "htd_wbc": "HTD policy -> HTD WBC, as deployed (Genesis)",
              "htd_wbc_h-0.08": "HTD policy -> HTD WBC, height at its 0.70 m neutral (Genesis)"}
RIGHT_LABEL = "operator teleop commands -> g1_walk (Genesis)"
INSET_LABEL = "real robot head camera"


def text(s):
    return s.replace(":", r"\:").replace("'", r"\'")


def make_clip(left, right, head, settle_s, out, left_label, fps):
    w, h = 960, 540
    draw = "drawtext=text='%s':x=16:y=14:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6"
    fc = ("[0:v]fps=%d,scale=%d:%d,tpad=stop_mode=clone:stop_duration=60,%s[l];" % (fps, w, h, draw % text(left_label))
          + "[1:v]fps=%d,scale=%d:%d,tpad=stop_mode=clone:stop_duration=60,%s[r];" % (fps, w, h, draw % text(RIGHT_LABEL))
          + "[2:v]fps=%d,scale=320:240,tpad=start_mode=clone:start_duration=%.3f,"
            "drawtext=text='%s':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55[c];"
          % (fps, settle_s, text(INSET_LABEL))
          + "[r][c]overlay=W-w-12:H-h-12[rc];[l][rc]hstack=inputs=2,format=yuv420p[v]")
    dur = max(_duration(left), _duration(right))
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", left, "-i", right, "-i", head,
           "-filter_complex", fc, "-map", "[v]", "-t", "%.2f" % dur, "-c:v", "libx264", "-crf", "23", out]
    subprocess.run(cmd, check=True)


def _duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--left", default="g1_walk", choices=sorted(LEFT_LABEL))
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    made = 0
    for left in sorted(glob.glob(os.path.join(args.runs, "episode_*_policy_%s.mp4" % args.left))):
        ep = os.path.basename(left).split("_policy_")[0]
        right = os.path.join(args.runs, "%s_recorded_g1_walk.mp4" % ep)
        head = os.path.join(args.real, "%s_head.mp4" % ep)
        log = left[:-4] + ".npz"
        if not (os.path.isfile(right) and os.path.isfile(head) and os.path.isfile(log)):
            print("[clips] %s: skipped (missing %s)" % (ep, [p for p in (right, head, log) if not os.path.isfile(p)]))
            continue
        out = os.path.join(args.out, "%s_%s_vs_operator.mp4" % (ep, args.left))
        make_clip(left, right, head, float(np.load(log)["settle_s"]), out, LEFT_LABEL[args.left], args.fps)
        print("[clips] %s" % out)
        made += 1
    print("[clips] %d clips in %s" % (made, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
