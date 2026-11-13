"""The teleop's MuJoCo sim host, unchanged, with the simulation state published for an external renderer.

    cd g1_walk/teleop-cbf-perception/teleop
    HTD_SIM_STATE_PUB=tcp://RENDER_HOST:5557 HTD_SIM_STATE_LOG=run/objects.jsonl PSI_SIM_SCENE=.../scene.xml \
        python ../../scene_sim/scene_sim_host.py --iface lo --domain 0 --lower jetson --inspire

Every argument goes to ``sim/sim_host.py``, which runs as if started directly. The one addition: after each
physics step that crosses a ``1 / HTD_SIM_STATE_HZ`` (60) boundary of simulated time, the world pose of every body
is sent on a ZMQ PUB socket (it *connects*; the renderer binds) as one JSON message

    {"t": sim seconds, "wall": epoch seconds, "bodies": {name: [x, y, z, qw, qx, qy, qz]}}

in the MuJoCo world frame (also appended to ``bodies.jsonl`` beside the state log, the input of the renderer's
``--playback``; ``HTD_SIM_BODY_LOG=0`` turns that off). ``HTD_SIM_STATE_LOG`` also appends the free bodies other than the robot (the scene's
objects) to a JSON-lines file at 10 Hz, for judging the episode afterwards; with it, ``contacts.jsonl`` beside it
gets, at the same rate, the poses of the bodies named in ``HTD_SIM_LOG_BODIES`` (comma separated; default the two
hand bases) and every contact between two different bodies' trees that carries force:
``{"t", "bodies": {name: pose}, "contacts": [[body a, body b, normal force N, depth m]], "left_hand": {"cmd", "q"}}``
(the feet on the ground are left out; ``left_hand`` is the six finger driver joints, commanded and measured, rad). Neither variable set: nothing is added.

``HTD_SIM_TOUCH=touch_model.npz`` (built by ``touch_model.py``) makes the emulated Inspire hands answer their
FORCE_ACT and tactile registers like the real ones: the recorded idle readings plus a response to the simulated
fingers' contacts, instead of zeros. The registers are written right after the host's own per-step hand update;
``touch.jsonl`` beside the object log records, at 10 Hz, each finger's contact force and FORCE_ACT.

The pinned teleop is not edited: ``mujoco.mj_step`` is wrapped before the host is run.
"""

import json
import os
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "teleop-cbf-perception" / "teleop" / "sim"


class Lines:
    """A text file written from a thread of its own: a write that waits on a busy disk must not hold up the
    physics loop (the robot state then arrives late, and a supervisor holds on that)."""

    def __init__(self, path):
        import queue
        import threading

        self.q = queue.Queue()
        self.f = open(path, "a", buffering=1)
        threading.Thread(target=self._run, name="log-" + Path(path).name, daemon=True).start()

    def _run(self):
        while True:
            self.f.write(self.q.get())

    def write(self, line):
        self.q.put(line)


class StatePublisher:
    def __init__(self, address, log_path, hz):
        self.period, self.next_t = 1.0 / hz, 0.0
        self.sock = None
        if address:
            import zmq

            self.sock = zmq.Context.instance().socket(zmq.PUB)
            self.sock.setsockopt(zmq.SNDHWM, 2)  # a slow renderer gets the latest state, not a backlog
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(address)
        self.log = Lines(log_path) if log_path else None
        self.contact_log = Lines(Path(log_path).with_name("contacts.jsonl")) if log_path else None
        # every published state (all bodies' world poses), for rendering the run afterwards (HTD_SIM_BODY_LOG=0: off)
        self.body_log = (Lines(Path(log_path).with_name("bodies.jsonl"))
                         if log_path and os.environ.get("HTD_SIM_BODY_LOG", "1") != "0" else None)
        self.watch_names = [n for n in os.environ.get(
            "HTD_SIM_LOG_BODIES", "left_base_link,right_base_link").split(",") if n]
        self.log_next = 0.0
        self.names = self.objects = None

    def _index(self, model):
        import mujoco

        self.names = [model.body(b).name or f"body_{b}" for b in range(model.nbody)]
        # the scene's objects: bodies with a free joint that are not the robot's floating base
        self.objects = []
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_qposadr[j] != 0:
                self.objects.append(int(model.jnt_bodyid[j]))
        print(f"[scene_sim] publishing {len(self.names) - 1} bodies at {1 / self.period:.0f} Hz; objects "
              f"{[self.names[b] for b in self.objects]}", flush=True)

    def __call__(self, model, data):
        t = float(data.time)
        if t + 1e-9 < self.next_t:
            return
        if self.names is None:
            self._index(model)
        self.next_t = t + self.period
        xpos, xquat = data.xpos, data.xquat
        if self.sock is not None or self.body_log is not None:
            bodies = {self.names[b]: [round(float(v), 5) for v in (*xpos[b], *xquat[b])]
                      for b in range(1, model.nbody)}
            msg = json.dumps({"t": round(t, 4), "wall": time.time(), "bodies": bodies})
            if self.sock is not None:
                try:
                    self.sock.send_string(msg, flags=1)  # zmq.NOBLOCK
                except Exception:  # noqa: BLE001 - never stall the physics for the renderer
                    pass
            if self.body_log is not None:
                self.body_log.write(msg + "\n")
        if self.log is not None and t >= self.log_next:
            self.log_next = t + 0.1
            self.log.write(json.dumps({"t": round(t, 3), **{
                self.names[b]: [round(float(v), 4) for v in (*xpos[b], *xquat[b])] for b in self.objects}}) + "\n")
            self._log_contacts(model, data, t)

    def _log_contacts(self, model, data, t):
        import mujoco
        import numpy as np

        force, rows = np.zeros(6), []
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if a == 0 or b == 0:
                if "ankle" in self.names[a or b]:  # standing
                    continue
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] > 1e-3 and model.body_rootid[a] != model.body_rootid[b]:
                rows.append([self.names[a], self.names[b], round(float(force[0]), 3), round(float(c.dist), 4)])
        watch = {n: [round(float(v), 4) for v in (*data.xpos[b], *data.xquat[b])]
                 for n in self.watch_names for b in [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] if b >= 0}
        if getattr(self, "hand_act", None) is None:  # the left hand's six driver joints: commanded and measured
            drivers = ("little_1", "ring_1", "middle_1", "index_1", "thumb_2", "thumb_1")
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{d}_joint") for d in drivers]
            jds = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_{d}_joint") for d in drivers]
            self.hand_act = ([i for i in ids if i >= 0], [int(model.jnt_qposadr[j]) for j in jds if j >= 0])
        hand = {"cmd": [round(float(data.ctrl[i]), 3) for i in self.hand_act[0]],
                "q": [round(float(data.qpos[a]), 3) for a in self.hand_act[1]]}
        self.contact_log.write(json.dumps({"t": round(t, 3), "bodies": watch, "contacts": rows,
                                           "left_hand": hand}) + "\n")


def install_touch(model_path, log_path):
    """Patch the emulated hands' per-step update to also fill the force and tactile registers."""
    import inspire_modbus_sim as ims
    from teleop.utils.inspire_touch import DFTP_TACTILE_REGIONS, FORCE_ACT_ADDRESS

    sys.path.insert(0, str(HERE))
    from touch_model import FINGERS, TouchSynth

    # the registers are refreshed at 100 Hz of simulated time (the controller sweeps them at 15 to 50 Hz): doing it
    # at every physics step costs more than the step itself and the simulation falls to half of real time
    base, state = ims.REG_WINDOW[0], {"synth": None, "next_log": 0.0, "next": 0.0}
    period = 1.0 / float(os.environ.get("HTD_SIM_TOUCH_HZ", "100"))
    log = Lines(Path(log_path).with_name("touch.jsonl")) if log_path else None
    episode = os.environ.get("HTD_SIM_TOUCH_EPISODE", "").strip() or None
    original = ims.EmulatedInspireHands.apply

    def apply(self, data):
        original(self, data)
        if data.time < state["next"]:
            return
        state["next"] = data.time + period
        model = self._pairs[0][1].model
        if state["synth"] is None:
            state["synth"] = TouchSynth(model_path, model, episode=episode)
            print(f"[scene_sim] touch: {model_path}, idle FORCE_ACT left "
                  f"{state['synth'].base_force['left'].round().tolist()}", flush=True)
        synth = state["synth"]
        reading = synth.read(model, data)
        for server, binding in self._pairs:
            force, skin = reading[binding.side]
            with server._lock:
                for i, v in enumerate(force):
                    server._regs[FORCE_ACT_ADDRESS - base + i] = int(v) & 0xFFFF
                at = 0
                for _, address, rows, cols in DFTP_TACTILE_REGIONS:
                    n = rows * cols
                    server._regs[address - base:address - base + n] = skin[at:at + n].tolist()  # 0..4095
                    at += n
        if log is not None and data.time >= state["next_log"]:
            state["next_log"] = data.time + 0.1
            log.write(json.dumps({"t": round(float(data.time), 3), **{
                side: {"newton": synth.newton[side].round(3).tolist(), "force": reading[side][0].tolist(),
                       "skin_sum": [int(reading[side][1][a0:a1].sum()) for _, a0, a1, _ in FINGERS]}
                for side in ("left", "right")}}) + "\n")

    ims.EmulatedInspireHands.apply = apply


def main():
    address = os.environ.get("HTD_SIM_STATE_PUB", "").strip()
    log_path = os.environ.get("HTD_SIM_STATE_LOG", "").strip()
    touch = os.environ.get("HTD_SIM_TOUCH", "").strip()
    if address or log_path or touch:
        # the host and its plant call mujoco.mj_step through the module, so wrapping the attribute reaches both
        for p in (str(SIM_DIR), str(SIM_DIR.parent), str(SIM_DIR.parent.parent)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from sim_host_common import choose_gl_backend  # the GL backend is chosen before mujoco is imported

        choose_gl_backend()
        import mujoco

        publish = StatePublisher(address, log_path, float(os.environ.get("HTD_SIM_STATE_HZ", "60")))
        step = mujoco.mj_step

        def mj_step(model, data, *args, **kwargs):
            step(model, data, *args, **kwargs)
            publish(model, data)

        if address or log_path:
            mujoco.mj_step = mj_step
        if touch:
            install_touch(touch, log_path)
    sys.argv = [str(SIM_DIR / "sim_host.py")] + sys.argv[1:]
    runpy.run_path(str(SIM_DIR / "sim_host.py"), run_name="__main__")


if __name__ == "__main__":
    main()
