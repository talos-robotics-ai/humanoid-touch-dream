import asyncio, json
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer
from deploy.ui.server import make_app

class FakeTelemetry:
    def __init__(self):
        self.sent = []
    def snapshot(self):
        return {"supervisor": {"state": "READY", "gating": ["press Start"], "log_line": "HTDPOLICY READY TO DEPLOY"},
                "policy": {"ready": True}, "measured": {"torso_height": 0.78}, "predicted": None,
                "ages": {"supervisor": 0.05}, "camera_seq": 1}
    def camera(self):
        return b"\xff\xd8jpeg"
    def send_command(self, name, source="ui"):
        self.sent.append((name, source)); return len(self.sent)

async def _scenario():
    fake = FakeTelemetry()
    async with TestClient(TestServer(make_app(fake, rate_hz=50.0))) as client:
        r = await client.get("/")
        assert r.status == 200 and "HTD" in await r.text()
        async with client.ws_connect("/ws") as ws:
            seen_text = seen_binary = False
            for _ in range(6):
                m = await ws.receive()
                if m.type == WSMsgType.TEXT:
                    d = json.loads(m.data)
                    assert d["type"] == "snapshot" and d["supervisor"]["state"] == "READY"
                    seen_text = True
                elif m.type == WSMsgType.BINARY:
                    assert m.data == b"\xff\xd8jpeg"; seen_binary = True
                if seen_text and seen_binary:
                    break
            assert seen_text and seen_binary
            await ws.send_json({"cmd": "start"})
            while True:
                m = await ws.receive_json()
                if m.get("type") == "ack":
                    break
            assert m["cmd"] == "start" and fake.sent[-1] == ("start", "ui")
            await ws.send_json({"cmd": "dance"})
            while True:
                m = await ws.receive_json()
                if m.get("type") == "error":
                    break
            assert fake.sent[-1] == ("start", "ui")
        r = await client.post("/cmd", json={"cmd": "stop"})
        assert r.status == 200 and fake.sent[-1] == ("stop", "ui")

def test_page_snapshot_camera_and_commands():
    asyncio.run(_scenario())

class _Reader:
    def __init__(self, samples):
        self.samples = samples
    def take(self, N=4):
        return self.samples


class _Malformed:
    timestamp = 1.0
    @property
    def jpeg(self):
        raise TypeError("malformed sample")


def test_a_malformed_sample_does_not_stop_the_poll_thread():
    import threading, time, types
    from deploy.ui.server import Telemetry
    t = Telemetry.__new__(Telemetry)
    ok = types.SimpleNamespace(timestamp=2.0, state="READY")
    t._readers = {"camera": _Reader([_Malformed()]), "supervisor": _Reader([ok])}
    t._lock, t._latest, t._jpeg, t._camera_seq, t._errors = threading.Lock(), {}, None, 0, set()
    t._last_poll = None
    t._thread = threading.Thread(target=t._poll, daemon=True)
    assert t.snapshot()["dds_alive"] is False                           # thread not started
    t.start()
    time.sleep(0.1)
    snap = t.snapshot()
    assert t._thread.is_alive() and snap["dds_alive"] is True and snap["supervisor"].state == "READY"
    assert "camera" in t._errors


def test_nonces_start_at_a_random_offset_per_process(monkeypatch):
    import types
    from deploy.ui import cmd
    from deploy.ui.cmd import NonceSource
    monkeypatch.setattr(cmd, "time", types.SimpleNamespace(time=lambda: 1000.0))
    firsts = {NonceSource().next() for _ in range(8)}
    assert len(firsts) > 1                                               # same millisecond, different seeds
    monkeypatch.undo()
    n = NonceSource()
    a, b = n.next(), n.next()
    assert 0 <= a < 2 ** 31 and 0 <= b < 2 ** 31 and a != b


def test_cmd_waits_for_a_matched_reader():
    # Domain 99, not 0: a robot may be live on domain 0.
    import time
    from cyclonedds.domain import DomainParticipant
    from deploy import dds
    from deploy.msgs import TOPIC_UI_COMMAND, UiCommand
    from deploy.ui.cmd import matched_writer
    p = DomainParticipant(99)
    t0 = time.monotonic()
    _, matched = matched_writer(p, timeout_s=0.3)
    assert not matched and time.monotonic() - t0 >= 0.3
    rd = dds.reader(p, TOPIC_UI_COMMAND, UiCommand)
    _, matched = matched_writer(p, timeout_s=2.0)
    assert matched and rd is not None
