#!/usr/bin/env python3
"""Roll a recorded HTD command stream out in Genesis on one of two lower-body controllers.

    python rollout.py --inputs EP_<source>_inputs.npz --lower g1_walk|htd_wbc \
        --walk_root <talosdeploy-g1/g1_walk> [--wbc_dir <robot_control/g1_wbc_policy>] \
        --out DIR [--fps 25] [--no_video]

``--inputs`` comes from ``walker_inputs.py``: per 50 Hz tick, what each
controller receives from policy_rollout's stream.

* ``g1_walk``: the walking policy the data was recorded with
  (talos_walk ``g1/g1_teleop_anyadapter``, model_9999) in the environment it
  was trained in (``AdapterEnv`` rebuilt from the run's ``cfgs.pkl``, domain
  randomisation, observation noise and pushes off, world model off as
  deployed). Each tick gets controller.py's walk, torso and arm commands after
  the walker's own command stage (``g1_lower``): clamped to the run's command
  box, walk slewed at 0.5 m/s^2 and 1.0 rad/s^2, torso at 0.4 rad/s and
  0.1 m/s, the arm copy clipped to the arm box and slewed at the run's 4 rad/s.
* ``htd_wbc``: HTD's own whole-body controller
  (``robot_control/g1_wbc_policy/student_policy_jit.pt``), its deploy loop
  (``g1_wholebody._update_lower_body_state``) reimplemented on the same
  Genesis scene: the same G1 model, ground and time step as the walker's
  environment, legs and waist on the policy's PD targets and gains, arms on
  HTD's IK solution at its arm gains (no gravity feed-forward), commands as
  HTD's controller passes them (velocity, absolute height, roll/pitch/yaw,
  clipped to its command range).

Writes DIR/<episode>_<source>_<lower>.mp4 (camera following the robot) and
.npz (base pose, commands, joints per tick, and the tick of a fall if one
happened). A fall -- base below 0.45 m or tilted past 50 deg -- ends the run.
The G1 model is the walker's by default (29 joints, rigid rubber hands); with
``--robot_mjcf`` the teleop's G1 with Inspire hands, whose fingers follow the
hand commands as the teleop's Inspire sim drives them.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

FALL_HEIGHT_M = 0.45
FALL_TILT_DEG = 50.0
RUN = ("g1", "g1_teleop_anyadapter", 9999)
# g1_lower's defaults (PSI_G1_WALK_ACCEL, PSI_G1_WALK_TORSO_RATE).
WALK_ACCEL = (0.5, 1.0)
TORSO_RATE = (0.4, 0.1)          # rad/s for roll/pitch/yaw, m/s for height
# The walker's deployed walk box (the deploy class's vx_range / wz_range).
G1_WALK_VX_RANGE = (-0.4, 0.8)
G1_WALK_WZ_RANGE = (-1.0, 1.0)


# The teleop's Inspire sim (teleop/sim/inspire_modbus_sim.py): per motor, in register
# order [little, ring, middle, index, thumb_bend, thumb_rotation], the driver joint and
# its mimic followers as ratios of the driver; register 1000 (open) is the driver's lower
# limit, 0 (closed) its upper; position servos of gain 3 within +-10 N m.
INSPIRE_MOTOR_COUPLING = [
    ("little_1", (("little_2", 1.0843),)),
    ("ring_1", (("ring_2", 1.0843),)),
    ("middle_1", (("middle_2", 1.0843),)),
    ("index_1", (("index_2", 1.0843),)),
    ("thumb_2", (("thumb_3", 0.8024), ("thumb_4", 0.8024 * 0.9487))),
    ("thumb_1", ()),
]
INSPIRE_KP, INSPIRE_KV, INSPIRE_FORCE = 3.0, 0.1, 10.0


class InspireFingers:
    """Both Inspire hands of a Genesis G1: Inspire commands (1 = open) -> finger joint targets."""

    def __init__(self, robot):
        self.robot = robot
        self.dofs, self.lo, self.hi, self.ratio, self.motor = [], [], [], [], []
        for h, side in enumerate(("left", "right")):
            for m, (driver, followers) in enumerate(INSPIRE_MOTOR_COUPLING):
                d = robot.get_joint("%s_%s_joint" % (side, driver)).dof_start
                lo, hi = (float(np.asarray(v.cpu() if hasattr(v, "cpu") else v).reshape(-1)[0])
                          for v in robot.get_dofs_limit([d]))
                for name, r in (("", 1.0),) + tuple(("%s_%s_joint" % (side, f), r) for f, r in followers):
                    self.dofs.append(d if not name else robot.get_joint(name).dof_start)
                    self.lo.append(lo)
                    self.hi.append(hi)
                    self.ratio.append(r)
                    self.motor.append(6 * h + m)
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        self.ratio, self.motor = np.array(self.ratio), np.array(self.motor)
        n = len(self.dofs)
        robot.set_dofs_kp(np.full(n, INSPIRE_KP), self.dofs)
        robot.set_dofs_kv(np.full(n, INSPIRE_KV), self.dofs)
        robot.set_dofs_force_range(np.full(n, -INSPIRE_FORCE), np.full(n, INSPIRE_FORCE), self.dofs)

    def apply(self, q01):
        closed = np.clip(1.0 - np.asarray(q01, float)[self.motor], 0.0, 1.0)
        target = (self.lo + closed * (self.hi - self.lo)) * self.ratio
        self.robot.control_dofs_position(target, self.dofs)
        return target


def slew(have, want, step):
    return have + np.clip(want - have, -step, step)


def quat_wxyz_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Recorder:
    def __init__(self):
        self.rows = {}

    def add(self, **kw):
        for k, v in kw.items():
            self.rows.setdefault(k, []).append(np.asarray(v, dtype=np.float64))

    def arrays(self):
        return {k: np.stack(v) for k, v in self.rows.items()}


def remap_run_paths(cfg, walk_root):
    """cfgs.pkl stores the exporting machine's absolute paths; point them at walk_root."""
    def fix(p):
        if not isinstance(p, str):
            return p
        for marker in ("/talos/robots/", "/data/", "/gait_libraries/"):
            if marker in p:
                return os.path.join(walk_root, p[p.index(marker) + 1:])
        return p
    for key in ("mjcf_path", "kintree_path", "gait_dir"):
        if key in cfg.env:
            cfg.env[key] = fix(cfg.env[key])
    amo = cfg.env.get("amo", {})
    if "phi_ckpt" in amo:
        amo["phi_ckpt"] = fix(amo["phi_ckpt"])
    for key in ("mjcf_path", "kintree_path", "gait_dir"):
        p = cfg.env.get(key)
        if isinstance(p, str) and not os.path.exists(p):
            raise FileNotFoundError("%s = %s not found under --walk_root" % (key, p))


def robot_only_mjcf(path):
    """``path`` without its world-level mocap bodies (the teleop scene's IK target and
    guard markers) and world-level geoms (its floor: the scene adds the ground), written
    beside it so mesh paths still resolve. Genesis makes the entity's root the world when
    anything but the robot hangs off the world body, and the robot's base pose then reads
    the world's (zero) instead of the pelvis'."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    world = tree.getroot().find("worldbody")
    extra = [b for b in world.findall("body") if b.get("mocap") == "true"] + world.findall("geom")
    if not extra:
        return path
    for b in extra:
        world.remove(b)
    out = os.path.splitext(path)[0] + "_robot_only.xml"
    tree.write(out)
    return out


def load_walk_run(walk_root, robot_mjcf=None):
    """The walker's run config (paths remapped, DR / noise / pushes off) and its facts;
    ``robot_mjcf`` replaces the run's robot model (same joint and link names)."""
    sys.path.insert(0, walk_root)
    from talos.rl.trainers import load_run_config
    sys.path.insert(0, os.path.join(walk_root, "deploy"))
    from talos_walk_policy import run_facts
    run_dir = os.path.join(walk_root, "logs", RUN[0], RUN[1])
    cfg = load_run_config(Path(run_dir))
    remap_run_paths(cfg, walk_root)
    cfg.reward["reward_scales"] = {}
    if "domain_rand" in cfg.env:
        cfg.env["domain_rand"] = {**cfg.env["domain_rand"], "enable": False}
    if "obs_noise" in cfg.obs:
        cfg.obs["obs_noise"] = {**cfg.obs["obs_noise"], "enable": False}
    cfg.env["studio_render"] = False      # the checker ground plane: walking reads against it
    cfg.env["push_interval_s"] = 0.0
    cfg.env["reset_joint_noise"] = 0.0
    cfg.env["episode_length_s"] = 1e6
    if robot_mjcf:
        cfg.env["mjcf_path"] = robot_only_mjcf(os.path.abspath(robot_mjcf))
    return run_dir, cfg, run_facts(walk_root, "/".join(RUN[:2]))


def use_plane_morph_for_ground(gs):
    """Genesis's bundled urdf/plane/plane.urdf declares XML version "0.0", which the
    XML parser of recent Pythons rejects; build the same infinite ground as the
    Plane morph (with its default checker look) instead."""
    class URDF(gs.morphs.URDF):        # a subclass, so Genesis's isinstance checks still hold
        def __new__(cls, *a, **kw):
            if kw.get("file") == "urdf/plane/plane.urdf":
                return gs.morphs.Plane()
            return super().__new__(cls)
    gs.morphs.URDF = URDF


# Camera in the robot's heading frame: front-left, so the hands and what they grasp stay
# in view through a turn (the ground's checker shows the turn itself).
CAMERA_OFFSET = (1.7, 1.3, 1.35)        # forward, left, up (m)


def follow(cam, pos, yaw):
    bx, by = float(pos[0]), float(pos[1])
    c, s = math.cos(yaw), math.sin(yaw)
    fx, fy, z = CAMERA_OFFSET
    cam.set_pose(pos=(bx + c * fx - s * fy, by + s * fx + c * fy, z), lookat=(bx, by, 0.75),
                 up=(0.0, 0.0, 1.0))     # else Genesis reuses the last up vector and the view rolls


def fell(pos_z, rpy):
    return pos_z < FALL_HEIGHT_M or max(abs(math.degrees(rpy[0])), abs(math.degrees(rpy[1]))) > FALL_TILT_DEG


# ----------------------------------------------------------------------------- g1_walk
def run_g1_walk(inp, args, rec):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    run_dir, cfg, facts = load_walk_run(args.walk_root, args.robot_mjcf)
    from talos.rl.envs.adapter_env import AdapterEnv
    from talos.rl import rotations as R
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    use_plane_morph_for_ground(gs)
    torch.manual_seed(0)
    env = AdapterEnv(1, cfg.env, cfg.obs, cfg.reward, cfg.command, cfg.clf,
                     show_viewer=False, record_camera=not args.no_video)
    runner = OnPolicyRunner(env, cfg.train, run_dir, device=gs.device)
    runner.load(os.path.join(run_dir, "model_%d.pt" % RUN[2]))
    policy = runner.get_inference_policy(device=gs.device)   # world model not attached: off, as deployed
    fingers = InspireFingers(env.robot) if args.robot_mjcf else None
    env._resample_commands = lambda idx: None
    env._resample_amo = lambda idx: None
    env.arm_motion = False
    dev, f = env.device, gs.tc_float
    box = facts.command_box
    h_rng, r_rng, p_rng, y_rng = (box[k] for k in ("height", "roll", "pitch", "yaw"))
    vx_rng, wz_rng = G1_WALK_VX_RANGE, G1_WALK_WZ_RANGE
    arm_lo, arm_hi = (np.asarray(v, float) for v in cfg.env["amo"]["arm_bounds"])
    arm_rate = float(facts.arm_rate)
    stand_vx = float(cfg.command.get("stand_thresh_vx", 0.05))
    stand_wz = float(cfg.command.get("stand_thresh_wz", 0.1))
    dt = float(env.dt)

    arm_obs = np.clip(inp["arm_wire"][0], arm_lo, arm_hi)
    env.amo_arm[:] = torch.tensor(arm_obs, device=dev, dtype=f)
    env.amo_arm_prev[:] = env.amo_arm
    env.amo_arm_goal[:] = env.amo_arm
    env.amo_arm_vel[:] = 0.0
    env.commands[:] = 0.0
    env.stand_mask[:] = True
    env.amo_rpy[:] = 0.0
    env.amo_h[:] = 0.0
    obs = env.reset()
    env.heading_on[:] = 1.0      # the deploy class holds heading; a reset may draw it off
    vx = wz = 0.0
    torso = np.zeros(4)
    if not args.no_video:
        env.cam.start_recording()
    every = max(1, int(round(1.0 / (dt * args.fps))))
    fall_tick = -1
    with torch.no_grad():
        for k in range(len(inp["walk_req"])):
            # g1_lower: clamp to the box, then slew
            want_vx = float(np.clip(inp["walk_req"][k, 0], *vx_rng))
            want_wz = float(np.clip(inp["walk_req"][k, 1], *wz_rng))
            vx = float(slew(vx, want_vx, WALK_ACCEL[0] * dt))
            wz = float(slew(wz, want_wz, WALK_ACCEL[1] * dt))
            req = inp["torso_req"][k]
            want = np.array([np.clip(req[0], *h_rng), np.clip(req[1], *r_rng),
                             np.clip(req[2], *p_rng), np.clip(req[3], *y_rng)])
            torso = np.array([slew(torso[0], want[0], TORSO_RATE[1] * dt)]
                             + [slew(torso[i], want[i], TORSO_RATE[0] * dt) for i in (1, 2, 3)])
            arm_obs = arm_obs + np.clip(np.clip(inp["arm_wire"][k], arm_lo, arm_hi) - arm_obs,
                                        -arm_rate * dt, arm_rate * dt)
            env.commands[0, 0] = vx
            env.commands[0, 1] = 0.0
            env.commands[0, 2] = wz
            env.stand_mask[0] = bool(abs(vx) < stand_vx and abs(wz) < stand_wz)
            env.amo_rpy[0] = torch.tensor(torso[1:4], device=dev, dtype=f)
            env.amo_h[0] = float(torso[0])
            env.base_height_cmd[0] = env.base_height_nominal + float(torso[0])
            zero = torch.zeros(1, device=dev, dtype=f)
            env.torso_pg_cmd[0] = R.projected_gravity(R.euler_xyz_to_quat(
                env.amo_rpy[0:1, 0], env.amo_rpy[0:1, 1], zero))[0]
            env.amo_arm[0] = torch.tensor(arm_obs, device=dev, dtype=f)
            env.amo_arm_goal[0] = env.amo_arm[0]
            if fingers is not None:
                fingers.apply(inp["hand_q01"][k])
            obs, _, dones, _ = env.step(policy(obs))
            pos = env.base_pos[0].cpu().numpy()
            quat = env.base_quat[0].cpu().numpy()
            rpy = quat_wxyz_to_rpy(quat)
            rec.add(t=k * dt, base_pos=pos, base_rpy=rpy, cmd_walk=(vx, wz), cmd_torso=torso,
                    q=env.dof_pos[0].cpu().numpy())
            if fingers is not None:
                rec.add(hand_cmd=inp["hand_q01"][k],
                        finger_q=env.robot.get_dofs_position(fingers.dofs).cpu().numpy().reshape(-1))
            if not args.no_video and k % every == 0:
                follow(env.cam, pos, rpy[2])
                env.cam.render()
            if bool(dones[0]) or fell(pos[2], rpy):
                fall_tick = k
                break
    if not args.no_video:
        env.cam.stop_recording(save_to_filename=args.video, fps=int(round(1.0 / (dt * every))))
    return fall_tick, dt, list(cfg.env["joint_names"])


# ----------------------------------------------------------------------------- htd_wbc
class HtdWbc:
    """``g1_wholebody._update_lower_body_state``: observation, policy, PD targets."""

    def __init__(self, wbc_dir):
        import yaml
        with open(os.path.join(wbc_dir, "g1_student_htd.yaml")) as fh:
            c = yaml.safe_load(fh)
        self.c = c
        self.policy = torch.jit.load(os.path.join(wbc_dir, "student_policy_jit.pt"), map_location="cpu")
        self.policy.eval()
        self.n = int(c["num_actions"])
        self.default = np.asarray(c["default_joint_pos"], np.float32)
        r = c["command_range"]
        names = ("lin_vel_x", "lin_vel_y", "ang_vel_z", "height", "body_roll", "body_pitch", "body_yaw")
        self.cmd_lo = np.array([r[k][0] for k in names], np.float32)
        self.cmd_hi = np.array([r[k][1] for k in names], np.float32)
        self.hist = np.zeros((int(c["history_length"]), int(c["num_obs"])), np.float32)
        self.action = np.zeros(self.n, np.float32)

    @staticmethod
    def gravity_orientation(q):
        """robot_control/g1_wbc_policy/rotation_helper.get_gravity_orientation (w, x, y, z)."""
        qw, qx, qy, qz = q
        return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx),
                         1 - 2 * (qw * qw + qz * qz)], np.float32)

    def step(self, quat_wxyz, gyro, q15, dq15, command7):
        c = self.c
        command = np.clip(np.asarray(command7, np.float32), self.cmd_lo, self.cmd_hi)
        obs = np.concatenate([np.asarray(gyro, np.float32) * c["ang_vel_scale"],
                              self.gravity_orientation(quat_wxyz), command,
                              (np.asarray(q15, np.float32) - self.default) * c["dof_pos_scale"],
                              np.asarray(dq15, np.float32) * c["dof_vel_scale"], self.action])
        self.hist[:-1] = self.hist[1:].copy()
        self.hist[-1] = obs
        x = torch.from_numpy(self.hist.reshape(1, -1)).clip(-100, 100)
        with torch.no_grad():
            self.action = self.policy(x).clip(-100, 100).numpy().squeeze().astype(np.float32)
        target = self.default + self.action * c["action_scale"]
        target[14] = np.clip(target[14], *c["waist_pitch_limits"])
        return target, command


def run_htd_wbc(inp, args, rec):
    import genesis as gs
    _, cfg, _ = load_walk_run(args.walk_root, args.robot_mjcf)   # the same G1 model, ground and physics step
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0)
    env_cfg = cfg.env
    dt = float(env_cfg["dt"])
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=env_cfg["substeps"]),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=bool(env_cfg.get("self_collision", False)),
            max_collision_pairs=300 if env_cfg.get("self_collision", False) else 200),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        show_viewer=False)
    scene.add_entity(gs.morphs.Plane())        # the walker's env's ground (see use_plane_morph_for_ground)
    robot = scene.add_entity(gs.morphs.MJCF(file=env_cfg["mjcf_path"]))
    cam = None if args.no_video else scene.add_camera(res=(1280, 720), pos=(2.5, -2.5, 1.8),
                                                      lookat=(0.0, 0.0, 0.7), fov=40, GUI=False)
    scene.build()
    names = list(env_cfg["joint_names"])
    if len(names) != 29:
        raise ValueError("expected the 29-joint G1, got %d joints" % len(names))
    dofs = [robot.get_joint(n).dof_start for n in names]     # G1 motor order (legs, waist, arms)
    wbc = HtdWbc(args.wbc_dir)
    c = wbc.c
    fingers = InspireFingers(robot) if args.robot_mjcf else None
    kp = np.zeros(29)
    kd = np.zeros(29)
    kp[c["joint2motor_idx"]] = c["kps"]
    kd[c["joint2motor_idx"]] = c["kds"]
    kp[c["arm_joint2motor_idx"]] = c["arm_kps"]
    kd[c["arm_joint2motor_idx"]] = c["arm_kds"]
    robot.set_dofs_kp(kp, dofs)
    robot.set_dofs_kv(kd, dofs)
    if "torque_limits" in env_cfg:
        lim = np.asarray(env_cfg["torque_limits"], float)
        robot.set_dofs_force_range(-lim, lim, dofs)
    # move_to_default_pos's end state: legs and waist at the policy default, arms at the first command
    q0 = np.zeros(29)
    q0[c["joint2motor_idx"]] = wbc.default
    q0[15:] = inp["arm_ik"][0]
    robot.set_pos(np.asarray(env_cfg["base_init_pos"], float))
    robot.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
    robot.set_dofs_position(q0, dofs, zero_velocity=True)
    target = q0.copy()
    if cam is not None:
        cam.start_recording()
    every = max(1, int(round(1.0 / (dt * args.fps))))
    fall_tick = -1
    for k in range(len(inp["arm_ik"])):
        quat = robot.get_quat().cpu().numpy().reshape(-1)[:4]
        w_world = robot.get_ang().cpu().numpy().reshape(-1)[:3]
        rot = _rot_wxyz(quat)
        gyro = rot.T @ w_world                     # body-frame angular velocity, as the pelvis IMU
        q = robot.get_dofs_position(dofs).cpu().numpy().reshape(-1)
        dq = robot.get_dofs_velocity(dofs).cpu().numpy().reshape(-1)
        idx = c["joint2motor_idx"]
        vel = inp["vel_raw"][k]
        torso = inp["torso_raw"][k].copy()         # absolute height, roll, pitch, yaw
        torso[0] += args.wbc_height_offset
        lower, command = wbc.step(quat, gyro, q[idx], dq[idx],
                                  [vel[0], vel[1], vel[2], torso[0], torso[1], torso[2], torso[3]])
        target[idx] = lower
        target[15:] = inp["arm_ik"][k]
        robot.control_dofs_position(target, dofs)
        if fingers is not None:
            fingers.apply(inp["hand_q01"][k])
        scene.step()
        pos = robot.get_pos().cpu().numpy().reshape(-1)[:3]
        rpy = quat_wxyz_to_rpy(robot.get_quat().cpu().numpy().reshape(-1)[:4])
        rec.add(t=k * dt, base_pos=pos, base_rpy=rpy, cmd_walk=(command[0], command[2]),
                cmd_vy=command[1], cmd_torso=command[3:7], q=q, action=wbc.action, obs=wbc.hist[-1])
        if fingers is not None:
            rec.add(hand_cmd=inp["hand_q01"][k],
                    finger_q=robot.get_dofs_position(fingers.dofs).cpu().numpy().reshape(-1))
        if cam is not None and k % every == 0:
            follow(cam, pos, rpy[2])
            cam.render()
        if fell(pos[2], rpy):
            fall_tick = k
            break
    if cam is not None:
        cam.stop_recording(save_to_filename=args.video, fps=int(round(1.0 / (dt * every))))
    return fall_tick, dt, names


def _rot_wxyz(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, help="walker_inputs.py npz")
    ap.add_argument("--lower", choices=("g1_walk", "htd_wbc"), required=True)
    ap.add_argument("--walk_root", required=True, help="talosdeploy-g1/g1_walk (talos/, logs/, data/)")
    ap.add_argument("--wbc_dir", default=None, help="robot_control/g1_wbc_policy (for --lower htd_wbc)")
    ap.add_argument("--robot_mjcf", default=None,
                    help="G1 model to simulate instead of the walker's training model, e.g. the "
                         "teleop's assets/g1/g1_inspire.xml (Inspire hands, fingers driven from the "
                         "inputs' hand_q01). Default: the walker's g1_29dof.xml (rubber hands)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--wbc_height_offset", type=float, default=0.0,
                    help="m added to the height HTD's controller sends the WBC (htd_wbc only); 0 is "
                         "the controller as deployed. This dataset's heights are the g1_walk "
                         "walker's (stand 0.78 m), the WBC's training neutral is 0.70 m")
    ap.add_argument("--fps", type=float, default=25.0, help="video frame rate")
    ap.add_argument("--no_video", action="store_true")
    args = ap.parse_args(argv)
    if args.lower == "htd_wbc" and not args.wbc_dir:
        ap.error("--lower htd_wbc needs --wbc_dir")
    args.walk_root = os.path.abspath(args.walk_root)
    z = np.load(args.inputs)
    inp = {k: z[k] for k in z.files}
    tag = "%s_%s_%s" % (str(inp["episode"]), str(inp["source"]), args.lower)
    if args.lower == "htd_wbc" and args.wbc_height_offset:
        tag += "_h%+.2f" % args.wbc_height_offset
    os.makedirs(args.out, exist_ok=True)
    args.video = os.path.join(os.path.abspath(args.out), tag + ".mp4")
    rec = Recorder()
    run = run_g1_walk if args.lower == "g1_walk" else run_htd_wbc
    fall_tick, dt, names = run(inp, args, rec)
    a = rec.arrays()
    np.savez_compressed(os.path.join(args.out, tag + ".npz"), lower=args.lower, source=str(inp["source"]),
                        wbc_height_offset=args.wbc_height_offset,
                        episode=str(inp["episode"]), dt=dt, settle_s=float(inp["settle_s"]),
                        fall_tick=fall_tick, joint_names=np.array(names), inputs=os.path.abspath(args.inputs), **a)
    p = a["base_pos"]
    print("[rollout] %s: %d ticks (%.1f s)%s, base moved %.2f m, yaw change %.0f deg -> %s"
          % (tag, len(p), len(p) * dt,
             (", FELL at %.1f s" % (fall_tick * dt)) if fall_tick >= 0 else "",
             float(np.linalg.norm(p[-1, :2] - p[0, :2])),
             math.degrees(float(np.unwrap(a["base_rpy"][:, 2])[-1] - a["base_rpy"][0, 2])),
             args.video if not args.no_video else args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
