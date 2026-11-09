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