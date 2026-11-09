"""Record the UI server's snapshots over its WebSocket, as a browser would see them.

    python deploy/tests/ws_probe.py --port 8765 --seconds 20 --out probe.json

Every snapshot's supervisor state, the controller's tick_p50_ms / tick_max_ms, rollout's
loop_p50_ms / loop_max_ms and inference_ms, and the last supervisor log line go into
`rows`; `summary` has p50 / max of each field over the window (each DDS sample counted
once: the server sends 20 snapshots a second of 10 Hz / 5 Hz topics), overall and per
supervisor state. status_probe.py writes the same JSON straight from DDS.
"""
import argparse
import asyncio
import json
import sys
import time

SUP_FIELDS = ("tick_p50_ms", "tick_max_ms")
POLICY_FIELDS = ("loop_p50_ms", "loop_max_ms", "inference_ms")
SETTLE_S = 1.0      # the controller's and rollout's timing windows are 1 s: older ticks are another state's


def row_from(sup, pol, received):
    """One recorded row from a SupervisorStatus and a PolicyStatus as dicts (either may be None)."""
    sup, pol = sup or {}, pol or {}
    return {
        "received": received,
        "sup_timestamp": sup.get("timestamp"),
        "state": sup.get("state"),
        "since_s": sup.get("since_s"),
        "tick_p50_ms": sup.get("tick_p50_ms"),
        "tick_max_ms": sup.get("tick_max_ms"),
        "log_line": sup.get("log_line"),
        "policy_timestamp": pol.get("timestamp"),
        "policy_ready": pol.get("ready"),
        "loop_p50_ms": pol.get("loop_p50_ms"),
        "loop_max_ms": pol.get("loop_max_ms"),
        "inference_ms": pol.get("inference_ms"),
    }


def _stats(values):
    v = sorted(x for x in values if isinstance(x, (int, float)))
    if not v:
        return {"n": 0, "p50": None, "max": None, "min": None, "mean": None}
    return {"n": len(v), "p50": v[len(v) // 2], "max": v[-1], "min": v[0], "mean": sum(v) / len(v)}


def _unique(rows, key):
    seen, out = set(), []
    for r in rows:
        k = r.get(key)
        if k is None or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _block(rows):
    sup, pol = _unique(rows, "sup_timestamp"), _unique(rows, "policy_timestamp")
    out = {f: _stats(r[f] for r in sup) for f in SUP_FIELDS}
    out.update({f"policy.{f}": _stats(r[f] for r in pol) for f in POLICY_FIELDS})
    out["n_supervisor"], out["n_policy"] = len(sup), len(pol)
    return out


def summarize(rows, settle_s=SETTLE_S):
    """`all`: every row; `by_state[S]`: rows in state S at least `settle_s` after entering it."""
    states = {}
    for r in _unique(rows, "sup_timestamp"):
        states[r["state"]] = states.get(r["state"], 0) + 1
    by_state = {}
    for s in states:
        if s is None:
            continue
        sel = [r for r in rows if r["state"] == s and (r["since_s"] or 0.0) >= settle_s]
        by_state[s] = _block(sel)
    log_lines = []
    for r in _unique(rows, "sup_timestamp"):
        if r["log_line"] and (not log_lines or log_lines[-1] != r["log_line"]):
            log_lines.append(r["log_line"])
    return {"n_rows": len(rows), "states": states, "settle_s": settle_s, "all": _block(rows),
            "by_state": by_state, "log_lines": log_lines}


def write(out, rows, meta):
    with open(out, "w") as f:
        json.dump({**meta, "summary": summarize(rows), "rows": rows}, f, indent=1)


async def record(port, seconds):
    import aiohttp
    rows, binary = [], 0
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(f"http://127.0.0.1:{port}/ws", heartbeat=10.0) as ws:
            t_end = time.monotonic() + seconds
            while (left := t_end - time.monotonic()) > 0:
                try:
                    msg = await ws.receive(timeout=left)
                except asyncio.TimeoutError:
                    break
                if msg.type == aiohttp.WSMsgType.BINARY:
                    binary += 1         # camera JPEG
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(f"websocket closed: {msg.type!r}")
                snap = json.loads(msg.data)
                if snap.get("type") == "snapshot":
                    rows.append(row_from(snap.get("supervisor"), snap.get("policy"), time.time()))
    return rows, binary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    rows, binary = asyncio.run(record(args.port, args.seconds))
    write(args.out, rows, {"source": "websocket", "port": args.port, "seconds": args.seconds,
                           "start": t0, "end": time.time(), "camera_frames": binary})
    print(f"ws_probe: {len(rows)} snapshots, {binary} camera frames -> {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
