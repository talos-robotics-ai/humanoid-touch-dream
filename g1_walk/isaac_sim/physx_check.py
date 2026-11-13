"""Stand-alone look at the PhysX scene built from a MuJoCo scene (Isaac Sim's python):

    PYTHONPATH=~/isaac_extra ~/isaacsim/python.sh g1_walk/isaac_sim/physx_check.py SCENE.xml [seconds]

Builds the stage, holds every joint at the keyframe pose with a PD, and reports what a correct conversion shows:
the articulation's joints and mass against the model's, how far the joints and the objects drift, the step time.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    import mujoco
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import Articulation, RigidPrim

    import physx_stage

    xml, seconds = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    lay = physx_stage.build(stage, model, data)
    dt = float(model.opt.timestep)
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0, backend="numpy")
    robot = Articulation(prim_paths_expr=lay.robot_root, name="robot")
    objects = {n: RigidPrim(prim_paths_expr=lay.body_path[n], name=n) for n in lay.free_bodies}
    sim.reset()
    robot.initialize()
    # a reset puts the articulation at its default joint state, which is zero: give it the pose it was built in
    names0 = list(robot.dof_names)
    q_build = data.qpos[np.array([model.jnt_qposadr[model.joint(n).id] for n in names0])].copy()
    robot.set_joint_positions(q_build[None, :])
    robot.set_joint_velocities(np.zeros((1, len(names0))))
    b = model.body(lay.robot_root.rsplit("/", 1)[1]).id
    robot.set_world_poses(data.xpos[b][None, :], data.xquat[b][None, :])
    robot.set_velocities(np.zeros((1, 6)))
    for o in objects.values():
        o.initialize()
    names = list(robot.dof_names)
    o0 = {n: o.get_world_poses()[0][0].copy() for n, o in objects.items()}
    globals()["B0"] = robot.get_world_poses()[0][0].copy()
    print("[check] objects at start:", {n: np.round(v, 3).tolist() for n, v in o0.items()}, "model:",
          {n: np.round(data.xpos[model.body(n).id], 3).tolist() for n in objects}, flush=True)
    print(f"[check] articulation: {robot.num_dof} dofs, {robot.num_bodies} links; model: {len(lay.hinge_joints)} "
          f"hinges, {len(lay.robot_bodies)} bodies", flush=True)
    print("[check] mass: physx", float(np.sum(robot.get_body_masses())), "model",
          float(sum(model.body_mass[model.body(n).id] for n in lay.robot_bodies)), flush=True)
    missing = [n for n in lay.hinge_joints if n not in names]
    print("[check] hinges missing from the articulation:", missing, flush=True)
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    q0 = data.qpos[qadr].copy()
    q_now = robot.get_joint_positions()[0]
    print("[check] initial joint error rad (max):", float(np.abs(q_now - q0).max()), flush=True)
    bad = np.argsort(-np.abs(q_now - q0))[:8]
    print("[check] worst joints (name, model q, physx q, axis):", [(names[i], round(float(q0[i]), 3), round(float(q_now[i]), 3),
          model.jnt_axis[model.joint(names[i]).id].tolist()) for i in bad], flush=True)
    import os
    if os.environ.get("CHECK_FIRST"):
        for k in range(int(os.environ["CHECK_FIRST"])):
            sim.step(render=False)
            print(f"[check] step {k + 1}: base {np.round(robot.get_world_poses()[0][0] - base0_, 4).tolist() if (base0_ := globals().get('B0')) is not None else ''} objects moved mm "
                  f"{ {n: np.round((o.get_world_poses()[0][0] - o0[n]) * 1000, 1).tolist() for n, o in objects.items()} }", flush=True)
    finger = np.array([any(f in n for f in ("thumb", "index", "middle", "ring", "little")) for n in names])
    kp, kd = np.where(finger, 3.0, 150.0), np.where(finger, 0.1, 4.0)
    robot.set_gains(kps=np.zeros((1, len(names))), kds=kd[None, :])
    p0 = {n: o.get_world_poses()[0][0].copy() for n, o in objects.items()}
    base0 = robot.get_world_poses()[0][0].copy()
    import os
    import carb
    import omni.physx
    if os.environ.get("CHECK_FAST"):
        st = carb.settings.get_settings()
        st.set_bool("/physics/updateToUsd", False)
        st.set_bool("/physics/updateVelocitiesToUsd", False)
        st.set_bool("/physics/updateParticlesToUsd", False)
        st.set_bool("/physics/disableContactProcessing", True)
        st.set_int("/persistent/physics/numThreads", int(os.environ.get("CHECK_THREADS", "0")))
    physx = omni.physx.get_physx_simulation_interface()
    direct = bool(os.environ.get("CHECK_DIRECT"))
    for label, body in (("state read+effort write", lambda: robot.set_joint_efforts((kp * (q0 - robot.get_joint_positions()[0]))[None, :])),
                        ("physics step", (lambda: (physx.simulate(dt, 0.0), physx.fetch_results())) if direct else (lambda: sim.step(render=False)))):
        t1 = time.perf_counter()
        for _ in range(200):
            body()
        print(f"[check] {label}: {(time.perf_counter() - t1) / 200 * 1e3:.2f} ms", flush=True)
    n_steps, t0 = int(seconds / dt), time.perf_counter()
    for k in range(n_steps):
        q = robot.get_joint_positions()[0]
        robot.set_joint_efforts((kp * (q0 - q))[None, :])
        if direct:
            physx.simulate(dt, k * dt)
            physx.fetch_results()
        else:
            sim.step(render=False)
        if k % 500 == 499:
            pos, quat = robot.get_world_poses()
            q = robot.get_joint_positions()[0]
            print(f"[check] t={(k + 1) * dt:.1f}s base {np.round(pos[0], 3).tolist()} quat {np.round(quat[0], 3).tolist()} "
                  f"max joint err {np.abs(q - q0).max():.3f} rad; objects moved (mm) "
                  f"{ {n: np.round((o.get_world_poses()[0][0] - p0[n]) * 1000, 1).tolist() for n, o in objects.items()} }",
                  flush=True)
    el = time.perf_counter() - t0
    print(f"[check] {n_steps} steps in {el:.1f} s: {el / n_steps * 1e3:.2f} ms per step, real-time factor "
          f"{n_steps * dt / el:.2f}; base moved {np.round(robot.get_world_poses()[0][0] - base0, 3).tolist()}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
