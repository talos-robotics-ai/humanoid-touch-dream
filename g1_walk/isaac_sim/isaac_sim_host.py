"""The teleop's sim host with PhysX (Isaac Sim) integrating the dynamics instead of MuJoCo.

    cd g1_walk/teleop-cbf-perception/teleop
    PYTHONPATH=~/isaac_extra PSI_SIM_SCENE=.../scene.xml HTD_SIM_STATE_PUB=tcp://127.0.0.1:5557 \
        ~/isaacsim/python.sh ../../isaac_sim/isaac_sim_host.py --iface lo --domain 0 --lower jetson --inspire

Run with Isaac Sim's python (plus a ``mujoco`` importable from it). Everything the stack talks to is the pinned
host's own code, unchanged: rt/lowcmd and rt/lowstate, the emulated Inspire hands on Modbus, the pelvis band and
its release, the wall-clock pacing. What changes is one call: ``mujoco.mj_step``. The MuJoCo model stays as the
host's book of names, addresses and actuator laws; the scene it describes is built once as a PhysX articulation
(``physx_stage.py``), and each step

1. the actuators' joint forces are computed from the controls as MuJoCo would (``mj_fwdActuation``: torque motors
   and the fingers' position servos alike) and applied as joint efforts; the per-joint damping the bridge writes
   (the drives' kd) becomes the PhysX drive damping, integrated implicitly;
2. external forces written to ``xfrc_applied`` (the pelvis band, a test push) are applied to the same bodies;
3. PhysX integrates one step;
4. joint positions and velocities, the floating base and the free objects are copied back into ``qpos``/``qvel``
   in MuJoCo's conventions, and the body poses are refreshed (``mj_kinematics``).

MuJoCo computes no contact here: ``data.contact`` is empty. The fingers' contact forces for the synthesized touch
come from PhysX's contact reports (``HTD_SIM_TOUCH``). The options of ``scene_sim/scene_sim_host.py`` apply
(``HTD_SIM_STATE_PUB``, ``HTD_SIM_STATE_LOG``, ``HTD_SIM_TOUCH``): the head camera's renderer and the result
judge read the same messages and files.
"""

import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "teleop-cbf-perception" / "teleop" / "sim"
FINGERS = ("little", "ring", "middle", "index", "thumb")


def log(msg):
    print(f"[isaac_sim] {msg}", flush=True)


class PhysxBackend:
    """One PhysX scene standing in for ``mj_step`` on a MuJoCo model."""

    def __init__(self):
        self.ready = False
        self.step_ms = 0.0
        self.tm = np.zeros(5)
        self.finger_newton = {s: np.zeros(len(FINGERS)) for s in ("left", "right")}  # until the scene exists
        self.q_last = None
        self.bridge = None

    def setup(self, model, data):
        import carb
        import mujoco
        import omni.physics.tensors as tensors
        import omni.physx
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from pxr import PhysxSchema, UsdGeom

        import physx_stage

        self.mj = mujoco
        mujoco.mj_forward(model, data)
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        lay = physx_stage.build(stage, model, data,
                                position_iterations=int(os.environ.get("HTD_PHYSX_ITERATIONS", "12")))
        self.lay = lay
        dt = float(model.opt.timestep)
        self.dt = dt
        st = carb.settings.get_settings()
        st.set_bool("/physics/updateToUsd", False)  # nothing reads the stage: writing 90 transforms a step
        st.set_bool("/physics/updateVelocitiesToUsd", False)  # costs six times the step itself
        st.set_bool("/physics/updateParticlesToUsd", False)
        st.set_int("/persistent/physics/numThreads", 0)
        st.set_int("/physics/numThreads", 0)

        # the fingers report their contacts
        finger_bodies = [n for n in lay.robot_bodies if any(n.startswith(f"{s}_{f}_") for s in ("left", "right")
                                                            for f in FINGERS)]
        for n in finger_bodies:
            api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(lay.body_path[n]))
            api.CreateThresholdAttr(0.0)

        self.sim = SimulationContext(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0, backend="numpy")
        self.sim.reset()
        self.physx = omni.physx.get_physx_simulation_interface()
        # the tensor views straight on PhysX: the high-level prim classes cost more per call than the step
        view = getattr(self.sim, "physics_sim_view", None)  # the context's own view, when it made one
        if view is None:
            view = tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
        self.view = view
        art = view.create_articulation_view(lay.robot_root)
        self.art = art
        names = list(art.shared_metatype.dof_names)
        self.one = np.arange(1, dtype=np.int32)
        jid = [model.joint(n).id for n in names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        root = model.body(lay.robot_root.rsplit("/", 1)[1]).id
        self.root, self.root_ipos = root, model.body_ipos[root].copy()
        # bodies that may get an external force (all of the robot's), and the free objects
        self.links = view.create_rigid_body_view([lay.body_path[n] for n in lay.robot_bodies])
        self.link_ids = np.array([model.body(n).id for n in lay.robot_bodies])
        self.link_idx = np.arange(len(lay.robot_bodies), dtype=np.int32)
        self.objs = view.create_rigid_body_view([lay.body_path[n] for n in lay.free_bodies]) if lay.free_bodies else None
        self.obj_idx = np.arange(len(lay.free_bodies), dtype=np.int32)
        self.obj_adr = []
        for n in lay.free_bodies:
            b = model.body(n).id
            j = int(model.body_jntadr[b])
            self.obj_adr.append((int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), model.body_ipos[b].copy()))
        self.finger_bodies = finger_bodies
        self.contacts = None
        if finger_bodies:
            try:
                self.contacts = view.create_rigid_contact_view([lay.body_path[n] for n in finger_bodies])
                self.contacts.get_net_contact_forces(dt)
            except Exception as e:  # noqa: BLE001 - touch then stays at its idle readings
                log(f"finger contact reports unavailable ({type(e).__name__}: {e}); touch keeps its idle baseline")
                self.contacts = None

        # the state the host prepared (standing pose, object poses) becomes PhysX's
        def xyzw(q):
            return [q[1], q[2], q[3], q[0]]

        art.set_root_transforms(np.array([[*data.qpos[0:3], *xyzw(data.qpos[3:7])]], np.float32), self.one)
        art.set_root_velocities(np.zeros((1, 6), np.float32), self.one)
        art.set_dof_positions(data.qpos[self.qadr][None, :].astype(np.float32), self.one)
        art.set_dof_velocities(np.zeros((1, len(names)), np.float32), self.one)
        art.set_dof_position_targets(data.qpos[self.qadr][None, :].astype(np.float32), self.one)
        art.set_dof_stiffnesses(np.zeros((1, len(names)), np.float32), self.one)
        if self.objs is not None:
            T = np.array([[*data.qpos[qa:qa + 3], *xyzw(data.qpos[qa + 3:qa + 7])] for qa, _, _ in self.obj_adr],
                         np.float32)
            self.objs.set_transforms(T, self.obj_idx)
            self.objs.set_velocities(np.zeros((len(self.obj_adr), 6), np.float32), self.obj_idx)
        self.kd = None
        self.kp = np.zeros(len(names), np.float32)
        # the actuators on joints: (actuator, qpos address, dof address, gear), for their lengths in ``step``
        jt = np.nonzero(model.actuator_trntype == self.mj.mjtTrn.mjTRN_JOINT)[0]
        jid = model.actuator_trnid[jt, 0]
        self.act = (jt, model.jnt_qposadr[jid], model.jnt_dofadr[jid], model.actuator_gear[jt, 0].copy())
        self.motor_dof = None
        b = self.bridge
        if b is not None:
            where = {int(d): k for k, d in enumerate(self.dadr)}
            self.motor_dof = np.array([where[int(v)] for v in b.body_vadr])
            self.motor_aid = np.array([int(a) for a in b.body_aid])
            limit = np.full(len(names), 1e6, np.float32)
            hi = np.minimum(np.abs(np.asarray(b._tau_lo, float)), np.abs(np.asarray(b._tau_hi, float)))
            limit[self.motor_dof] = np.where(np.isfinite(hi), hi, 1e6)
            art.set_dof_max_forces(limit[None, :], self.one)  # a drive saturates its whole PD sum
        self.finger_newton = {s: np.zeros(len(FINGERS)) for s in ("left", "right")}
        self.finger_of = [(n.split("_")[0], FINGERS.index(n.split("_")[1])) for n in finger_bodies]
        self.contact_next = 0.0
        # a collection pause of the interpreter is a robot state that arrives late, and the supervisor holds on
        # a state older than 0.1 s: what exists now is kept out of the collector, and it runs no more
        import gc

        gc.collect()
        gc.freeze()
        gc.disable()
        self.ready = True
        log(f"PhysX scene: {len(names)} joints, {len(lay.robot_bodies)} links, objects {lay.free_bodies}, "
            f"step {dt * 1e3:g} ms; finger contact reports {'on' if self.contacts is not None else 'off'}")

    def step(self, model, data):
        if not self.ready:
            self.setup(model, data)
        mujoco, art, tm = self.mj, self.art, self.tm
        t0 = time.perf_counter()
        # 0. joint states the host wrote since the last step (the bridge holds the joints in the standing pose
        # this way until the first command arrives) are PhysX's state too
        q_host = data.qpos[self.qadr]
        if self.q_last is not None and not np.array_equal(q_host, self.q_last):
            art.set_dof_positions(q_host[None, :].astype(np.float32), self.one)
            art.set_dof_velocities(data.qvel[self.dadr][None, :].astype(np.float32), self.one)
        # 1. actuation as the model defines it, and the damping written for this step
        # (the fingers' position servos act on the actuator lengths, which MuJoCo refreshes in its position stage;
        # that stage does not run here, so they are written from the joints: left stale, a servo pulls toward
        # its target from the angle of the first step forever, i.e. not at all at a zero target)
        jt, qa, da, gear = self.act
        data.actuator_length[jt] = gear * data.qpos[qa]
        data.actuator_velocity[jt] = gear * data.qvel[da]
        mujoco.mj_fwdActuation(model, data)
        effort = data.qfrc_actuator[self.dadr].astype(np.float32)
        b = self.bridge
        if b is not None and self.motor_dof is not None and b.took_over:
            # The bridge wrote each body motor's torque as kp (q_des - q) + ... at the state it read. Applied as
            # a constant over the step that spring is explicit, and at this step length it chatters on the
            # light distal joints (the model gives the body joints no rotor inertia). The same torque as a
            # drive: stiffness kp around the target that yields that torque at the present angle, so the
            # spring follows the joint inside the step.
            kp = np.zeros(len(effort), np.float32)
            start = b._n_hold if b.hold_legs else 0
            kp[self.motor_dof[start:]] = [h[2] for h in b._held_cmd[start:b.n_body]]
            on = kp > 0
            target = q_host.astype(np.float32)
            u = np.zeros(len(effort), np.float32)
            u[self.motor_dof] = data.ctrl[self.motor_aid]
            target[on] += u[on] / kp[on]
            effort[on] = 0.0
            if not np.array_equal(kp, self.kp):
                self.kp = kp
                art.set_dof_stiffnesses(kp[None, :], self.one)
            art.set_dof_position_targets(target[None, :], self.one)
        art.set_dof_actuation_forces(effort[None, :], self.one)
        kd = model.dof_damping[self.dadr]
        if self.kd is None or not np.array_equal(kd, self.kd):
            self.kd = kd.copy()
            art.set_dof_dampings(kd[None, :].astype(np.float32), self.one)
        # 2. external forces (world frame, at the body's centre of mass)
        w = data.xfrc_applied[self.link_ids]
        hit = np.nonzero(w.any(axis=1))[0]
        if len(hit):
            F, Tq, P = (np.zeros((len(self.link_ids), 3), np.float32) for _ in range(3))
            F[hit], Tq[hit], P[hit] = w[hit, 0:3], w[hit, 3:6], data.xipos[self.link_ids[hit]]
            self.links.apply_forces_and_torques_at_position(F, Tq, P, hit.astype(np.int32), True)
        t1 = time.perf_counter()
        # 3. integrate
        self.physx.simulate(self.dt, float(data.time))
        self.physx.fetch_results()
        t2 = time.perf_counter()
        # 4. the state, in MuJoCo's conventions
        data.qpos[self.qadr] = art.get_dof_positions()[0]
        data.qvel[self.dadr] = art.get_dof_velocities()[0]
        self.q_last = data.qpos[self.qadr].copy()
        T = art.get_root_transforms()[0]
        self._write_free(data, 0, 0, T[0:3], T[[6, 3, 4, 5]], art.get_root_velocities()[0], self.root_ipos)
        if self.objs is not None:
            T, V = self.objs.get_transforms(), self.objs.get_velocities()
            for k, (qa, da, ipos) in enumerate(self.obj_adr):
                self._write_free(data, qa, da, T[k, 0:3], T[k, [6, 3, 4, 5]], V[k], ipos)
        data.time += self.dt
        t3 = time.perf_counter()
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        t4 = time.perf_counter()
        # the fingers' contact forces, at 100 Hz
        if self.contacts is not None and data.time >= self.contact_next:
            self.contact_next = data.time + 0.01
            mag = np.linalg.norm(np.asarray(self.contacts.get_net_contact_forces(self.dt)).reshape(-1, 3), axis=1)
            out = {s: np.zeros(len(FINGERS)) for s in ("left", "right")}
            for (side, k), m in zip(self.finger_of, mag):
                out[side][k] += float(m)
            self.finger_newton = out
        t5 = time.perf_counter()
        tm += 0.01 * (np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]) * 1e3 - tm)
        self.step_ms += 0.01 * ((t5 - t0) * 1e3 - self.step_ms)

    @staticmethod
    def _write_free(data, qa, da, pos, quat, vel, ipos):
        """A free body's pose and velocity: PhysX reports the velocity of the centre of mass and the angular
        velocity in the world; MuJoCo keeps the velocity of the body's origin and the angular velocity in the
        body."""
        w, x, y, z = (float(v) for v in quat)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        omega = np.asarray(vel[3:6], float)
        data.qpos[qa:qa + 3] = pos
        data.qpos[qa + 3:qa + 7] = quat
        data.qvel[da:da + 3] = np.asarray(vel[0:3], float) - np.cross(omega, R @ ipos)
        data.qvel[da + 3:da + 6] = R.T @ omega


def main():
    from isaacsim import SimulationApp

    # Kit must be up before PhysX and USD are imported. One articulation is stepped from this thread: PhysX's
    # worker pool and Kit's task threads would only spin beside it, on cores the walker and controller need
    app = SimulationApp({"headless": True, "extra_args": [  # noqa: F841
        "--/persistent/physics/numThreads=0", "--/physics/numThreads=0",
        "--/plugins/carb.tasking.plugin/threadCount=2", "--/app/asyncRendering=false"]})
    for p in (str(HERE), str(HERE.parent / "scene_sim"), str(SIM_DIR), str(SIM_DIR.parent), str(SIM_DIR.parent.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # OpenCV as bundled with Kit registers a top-level ``utils`` (its own) when imported; the teleop's modules
    # must find the teleop's ``utils`` package under that name
    import cv2  # noqa: F401 - imported first, so that it cannot replace the name later
    for name in [m for m in sys.modules if m == "utils" or m.startswith("utils.")]:
        del sys.modules[name]
    import types

    module = types.ModuleType("utils")  # the teleop's is a package without __init__: a path is all it has
    module.__path__ = [str(SIM_DIR.parent / "utils")]
    sys.modules["utils"] = module
    from sim_host_common import choose_gl_backend

    choose_gl_backend()
    import mujoco

    import scene_sim_host

    # the step is a dozen short calls where MuJoCo's was one that released the interpreter lock: the DDS and
    # Modbus threads of this process must hand the lock back quickly, or the step loop spends its time waiting
    sys.setswitchinterval(float(os.environ.get("HTD_SWITCH_INTERVAL", "0.0005")))
    backend = PhysxBackend()
    import mujoco_dds_bridge

    bridge_init = mujoco_dds_bridge.UnitreeMujocoBridge.__init__

    def init(self, *a, **k):
        bridge_init(self, *a, **k)
        backend.bridge = self  # its per-motor gains make the joint springs implicit (PhysxBackend.step)

    mujoco_dds_bridge.UnitreeMujocoBridge.__init__ = init
    address = os.environ.get("HTD_SIM_STATE_PUB", "").strip()
    log_path = os.environ.get("HTD_SIM_STATE_LOG", "").strip()
    touch = os.environ.get("HTD_SIM_TOUCH", "").strip()
    publish = None
    if address or log_path:
        publish = scene_sim_host.StatePublisher(address, log_path, float(os.environ.get("HTD_SIM_STATE_HZ", "60")))
    said = {"t": 0.0}

    gap = {"t": None}

    def mj_step(model, data, *args, **kwargs):
        t_in = time.perf_counter()
        backend.step(model, data)
        t_mid = time.perf_counter()
        if publish is not None:
            publish(model, data)
        t_out = time.perf_counter()
        # a stall is a robot state that arrives late: say where the time went (outside = the host's own loop)
        if gap["t"] is not None and backend.step_ms and t_out - gap["t"] > 0.06:
            log(f"stall at t={data.time:.2f}s: {1e3 * (t_out - gap['t']):.0f} ms between steps (host loop "
                f"{1e3 * (t_in - gap['t']):.0f}, physics {1e3 * (t_mid - t_in):.0f}, state out {1e3 * (t_out - t_mid):.0f})")
        gap["t"] = t_out
        if data.time - said["t"] > 10.0:
            said["t"] = float(data.time)
            log(f"t={data.time:.1f}s: step {backend.step_ms:.2f} ms (write {backend.tm[0]:.2f}, physx {backend.tm[1]:.2f}, "
                f"read {backend.tm[2]:.2f}, kinematics {backend.tm[3]:.2f}, contacts {backend.tm[4]:.2f}); pelvis "
                f"{np.round(data.qpos[0:3], 3).tolist()}")

    mujoco.mj_step = mj_step
    if touch:
        # the synthesized touch takes the fingers' forces from PhysX instead of MuJoCo's contact list
        import touch_model

        touch_model.TouchSynth._contact_newtons = lambda self, model, data: {
            s: v.copy() for s, v in backend.finger_newton.items()}
        scene_sim_host.install_touch(touch, log_path)
    sys.argv = [str(SIM_DIR / "sim_host.py")] + sys.argv[1:]
    code = 0
    try:
        runpy.run_path(str(SIM_DIR / "sim_host.py"), run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        # Kit keeps threads alive after the host returns and would hold the hands' ports: leave for good
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
