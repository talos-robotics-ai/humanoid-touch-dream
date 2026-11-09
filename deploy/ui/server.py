"""Monitoring UI: DDS subscriber -> WebSocket, static page, and UiCommand on a button press.

It only subscribes (best-effort, keep-last-1), except for one UiCommand per command, so a
slow or absent browser can never hold up a robot-side writer.
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import sys
import threading
import time

from aiohttp import WSMsgType, web

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from deploy import dds  # noqa: E402
from deploy.g1_spec import G1_SPEC  # noqa: E402
from deploy.msgs import (TOPIC_CAMERA, TOPIC_POLICY_STATUS, TOPIC_PREDICTED_CHUNK, TOPIC_SUPERVISOR,  # noqa: E402
                         TOPIC_UI_COMMAND, UI_COMMANDS, CameraFrame, PolicyStatus, PredictedChunk,
                         SupervisorStatus, UiCommand)
from deploy.plot_frames import measured_point, predicted_series  # noqa: E402
from deploy.ui.cmd import NonceSource  # noqa: E402

log = logging.getLogger("htd.ui")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SEND_TIMEOUT_S = 2.0     # a browser that cannot take one message in this long is dropped
POLL_S = 0.005


def jsonable(x):
    """Dataclasses, sequences and numbers -> plain JSON values; non-finite floats -> None."""
    if dataclasses.is_dataclass(x):
        return {f.name: jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    return x


def measured_entry(robot_state, eef_state):
    p = measured_point(robot_state, eef_state, G1_SPEC.plot)
    torso = list(robot_state.torso_pose)
    p["torso_roll"], p["torso_pitch"] = (float(torso[i]) if len(torso) > i else math.nan for i in (3, 4))
    return p


def predicted_entry(chunk, robot_state, received):
    rpy = {s: list(getattr(robot_state, f"{s}_end_effector_to_torso_pose"))[3:6] if robot_state else None
           for s in ("left", "right")}
    series = predicted_series(chunk, rpy, G1_SPEC.plot)
    if series is None:
        return None
    return {**series, "t_obs": int(chunk.t_obs), "horizon": int(chunk.horizon), "interval": int(chunk.interval),
            "inference_ms": float(chunk.inference_ms), "received": received,
            "age": time.monotonic() - received}


class Telemetry:
    """Latest sample of every topic, read by one polling thread."""

    def __init__(self, domain_id: int = 0):
        from cyclonedds.domain import DomainParticipant
        sys.path.insert(0, os.path.join(_REPO, "robot_control"))
        from teleop_dds_msgs import EEFState, G1RobotState

        self._participant = DomainParticipant(domain_id)
        p = self._participant
        self._readers = {
            "robot_state": dds.reader(p, "rt/robot/state", G1RobotState),
            "eef_state": dds.reader(p, "rt/robot/eef_state", EEFState),
            "supervisor": dds.reader(p, TOPIC_SUPERVISOR, SupervisorStatus),
            "policy_status": dds.reader(p, TOPIC_POLICY_STATUS, PolicyStatus),
            "predicted_chunk": dds.reader(p, TOPIC_PREDICTED_CHUNK, PredictedChunk),
            "camera": dds.reader(p, TOPIC_CAMERA, CameraFrame),
            "ui_command": dds.reader(p, TOPIC_UI_COMMAND, UiCommand),
        }
        self._writer = dds.writer(p, TOPIC_UI_COMMAND, UiCommand)
        self._nonces = NonceSource()
        self._lock = threading.Lock()
        self._latest = {}           # topic -> (sample, monotonic arrival)
        self._jpeg = None
        self._camera_seq = 0
        self._errors = set()
        self._last_poll = None      # monotonic end of the last sweep over the readers
        self._thread = threading.Thread(target=self._poll, name="ui-dds", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _poll(self):
        while True:
            for name, rd in self._readers.items():
                try:   # the whole iteration: a malformed sample must not end the thread
                    samples = [s for s in rd.take(N=4) if s is not None and hasattr(s, "timestamp")]
                    if not samples:
                        continue
                    last, now = samples[-1], time.monotonic()
                    jpeg = bytes(last.jpeg) if name == "camera" else None
                    with self._lock:
                        self._latest[name] = (last, now)
                        if jpeg is not None:
                            self._jpeg, self._camera_seq = jpeg, self._camera_seq + 1
                except Exception as e:
                    self._error(name, e)
            self._last_poll = time.monotonic()
            time.sleep(POLL_S)

    def _error(self, where, e):
        if where not in self._errors:
            self._errors.add(where)
            log.warning("%s failed (logged once): %s", where, e)

    def snapshot(self) -> dict:
        with self._lock:
            latest, seq = dict(self._latest), self._camera_seq
        now = time.monotonic()
        get = lambda k: latest[k][0] if k in latest else None  # noqa: E731
        rs, chunk = get("robot_state"), get("predicted_chunk")
        return jsonable({
            "supervisor": get("supervisor"),
            "policy": get("policy_status"),
            "measured": self._convert("measured", measured_entry, rs, get("eef_state")) if rs is not None else None,
            "predicted": (self._convert("predicted", predicted_entry, chunk, rs, latest["predicted_chunk"][1])
                          if chunk is not None else None),
            "ages": {k: (now - latest[k][1] if k in latest else None) for k in self._readers},
            "dds_alive": self._dds_alive(now),
            "camera_seq": seq,
        })

    def _dds_alive(self, now) -> bool:
        """The poll thread runs and finished a sweep within the last second."""
        last = self._last_poll
        return bool(self._thread.is_alive() and last is not None and now - last <= 1.0)

    def _convert(self, what, fn, *args):
        """A malformed sample shows as None on the page instead of breaking every snapshot."""
        try:
            return fn(*args)
        except Exception as e:
            self._error(what, e)
            return None

    def camera(self):
        with self._lock:
            return self._jpeg

    def send_command(self, name: str, source: str = "ui") -> int:
        if name not in UI_COMMANDS:
            raise ValueError(f"unknown command {name!r}; expected one of {', '.join(UI_COMMANDS)}")
        nonce = self._nonces.next()
        self._writer.write(UiCommand(timestamp=time.time(), nonce=nonce, command=name, source=source))
        return nonce


def _command(telemetry, data):
    """{"cmd": name} -> (reply, HTTP status)."""
    name = data.get("cmd") if isinstance(data, dict) else None
    if name not in UI_COMMANDS:
        return {"type": "error", "cmd": name, "error": f"unknown command; expected one of {', '.join(UI_COMMANDS)}"}, 400
    try:
        return {"type": "ack", "cmd": name, "nonce": telemetry.send_command(name)}, 200
    except ValueError as e:
        return {"type": "error", "cmd": name, "error": str(e)}, 400
    except Exception as e:      # a DDS failure is reported, never raised into the server
        log.exception("sending %r failed", name)
        return {"type": "error", "cmd": name, "error": f"send failed: {e}"}, 500


def make_app(telemetry, rate_hz: float = 20.0, static_dir: str | None = None) -> web.Application:
    static_dir = os.path.realpath(static_dir or STATIC_DIR)
    period = 1.0 / float(rate_hz)

    async def index(request):
        return web.FileResponse(os.path.join(static_dir, "index.html"))

    async def static(request):
        path = os.path.realpath(os.path.join(static_dir, request.match_info["name"]))
        if os.path.dirname(path) != static_dir or not os.path.isfile(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "no-cache"})

    async def cmd(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "body must be JSON {\"cmd\": ...}"}, status=400)
        reply, status = _command(telemetry, data)
        if status != 200:
            return web.json_response({"error": reply["error"]}, status=status)
        return web.json_response({"ok": True, "nonce": reply["nonce"]})

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10.0)
        await ws.prepare(request)
        send_lock = asyncio.Lock()

        async def send(coro_fn, payload):
            async with send_lock:
                await asyncio.wait_for(coro_fn(payload), SEND_TIMEOUT_S)

        async def sender():
            last_seq = None
            try:
                while not ws.closed:
                    await asyncio.sleep(period)
                    snap = telemetry.snapshot()
                    await send(ws.send_str, json.dumps(jsonable({"type": "snapshot", **snap}), allow_nan=False))
                    seq = snap.get("camera_seq")
                    if seq != last_seq:
                        jpeg = telemetry.camera()
                        if jpeg:
                            await send(ws.send_bytes, jpeg)
                        last_seq = seq
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.info("websocket sender stopped: %r", e)
                await ws.close()

        task = asyncio.create_task(sender())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    data = None
                reply, _ = _command(telemetry, data)
                await send(ws.send_str, json.dumps(reply))
        except Exception as e:
            log.info("websocket receiver stopped: %r", e)
        finally:
            task.cancel()
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/static/{name}", static)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/cmd", cmd)
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iface", default=None, help="network interface for CycloneDDS (e.g. the robot's eth)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rate", type=float, default=20.0, help="snapshot rate per browser, Hz")
    ap.add_argument("--demo", action="store_true", help="animated fake telemetry, no DDS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        os.nice(10)     # the robot-side processes come first
    except OSError:
        pass
    if args.demo:
        from deploy.ui.demo import DemoTelemetry
        telemetry = DemoTelemetry()
    else:
        dds.configure_interface(args.iface)
        telemetry = Telemetry().start()
    print(f"HTD UI on http://{args.host}:{args.port}" + (" (demo)" if args.demo else ""), flush=True)
    web.run_app(make_app(telemetry, rate_hz=args.rate), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
