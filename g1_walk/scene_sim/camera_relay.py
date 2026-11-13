"""Keep the renderer's latest head-camera frame beside the policy.

    python g1_walk/scene_sim/camera_relay.py --stream tcp://RENDER_HOST:5558 [--serve tcp://127.0.0.1:5556]

The renderer (real2sim ``head_camera.py --serve --stream``) pushes every frame it renders; asking it for each
frame instead costs two crossings of the link per frame, and on a 50 ms, jittery link the policy then sees a handful
of frames per second with gaps the supervisor's 1 s watchdog trips on. This relay receives the stream and answers
policy rollout's ``zmq`` head-camera requests locally with the newest frame, in the camera server's reply format
(part 0 = the colour JPEG).

It does not hide a dead camera: a frame older than ``--max_age`` (0.5 s since it arrived) is not served; the reply is
then empty, which rollout reads as "no frame", and its own staleness reporting takes over. Every 5 s it prints the
frames received and served per second and the largest gap between arrivals.
"""

import argparse
import json
import signal
import time

import zmq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", required=True, help="the renderer's PUB address")
    ap.add_argument("--serve", default="tcp://127.0.0.1:5556", help="REP bind address rollout connects to")
    ap.add_argument("--max_age", type=float, default=0.5)
    a = ap.parse_args(argv)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 2)
    sub.connect(a.stream)
    rep = ctx.socket(zmq.REP)
    rep.bind(a.serve)
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    poller.register(rep, zmq.POLLIN)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    print(f"READY: camera relay {a.stream} -> {a.serve}", flush=True)
    jpeg, arrived, sim_t = b"", 0.0, None
    n_in = n_out = n_empty = 0
    gap, last_log = 0.0, time.time()
    while not stop["now"]:
        for sock, _ in poller.poll(200):
            if sock is sub:
                while True:  # drain to the newest
                    try:
                        parts = sub.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    now = time.time()
                    if arrived:
                        gap = max(gap, now - arrived)
                    jpeg, arrived = parts[0], now
                    if len(parts) > 1:
                        sim_t = json.loads(parts[1]).get("sim_t")
                    n_in += 1
            else:
                rep.recv()
                fresh = bool(jpeg) and time.time() - arrived <= a.max_age
                rep.send_multipart([jpeg if fresh else b""])
                n_out += 1
                n_empty += 0 if fresh else 1
        if time.time() - last_log >= 5.0:
            el = time.time() - last_log
            print(f"[relay] in {n_in / el:.1f} fps, served {n_out / el:.1f}/s ({n_empty} empty), largest gap "
                  f"{gap * 1000:.0f} ms, sim t {sim_t}", flush=True)
            n_in = n_out = n_empty = 0
            gap, last_log = 0.0, time.time()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
