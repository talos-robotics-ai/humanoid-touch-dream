"""A talos_walk run (AnyTracker / AnyAdapter) as a deployable policy.

This copy adds what a run trained with a REDUCED action space needs (v4.2,
``policy_action_idx`` in its cfg): the network outputs only those joints
(the 12 legs), the last-action slot of the observation is that wide, and
every other joint's PD target is its reference -- which for the arms IS the
operator's command, and for the head is the operator's own target (the host
overrides it; the policy reads the head at its default pose, ``mask_aux``,
forced on for a run trained with ``head_motion``). PD targets are clipped
to joint travel minus the run's ``clamp_margin``. ``heading_hold=False``
feeds 0 in the heading-error slot, as training did in its hold-off episodes.

The deploy-side twin of talos_walk's training environments
(``talos/rl/envs``: ``HzdEnv -> AnyTrackEnv -> AdapterEnv``). A run there is
``logs/<robot>/<exp>/{cfgs.pkl, model_<n>.pt}`` and nothing else, so this class
rebuilds from those two files everything the environment computed around the
network every control step:

* the reference ``g_t`` -- HZD gait library, analytic turn layer, the stand
  cross-fade with its slewed gait speed and parked clock, the PhiWalk residual
  on legs and waist, the arms at the commanded pose with the command's own
  velocity as their reference velocity;
* the observation frame, stacked newest first::

      ang_vel | gravity | cmd*scale | q - default | dq*scale | last action |
      sin, cos | AMO [rpy, h, arm] | heading error* | measured height* |
      q_des - q | (qd_des - dq)*scale | base_height_cmd | ref_feet_h

  (the starred terms only on a run trained with them);
* for an adapter run, the proprioceptive history (oldest first) and the
  innovation channel -- world state minus the world model's one-step
  prediction of it -- with the prediction formed from the action about to be
  applied, as ``AdapterEnv.step`` does.

The gait library, PhiWalk and the kinematic tree are talos_walk's OWN modules,
imported from the tree the run lives in: a port of them would be a second
implementation to keep in step with training, and these three are torch and
numpy only. The networks are rebuilt here in plain torch because the training
classes subclass rsl_rl's, which a deploy host does not carry.

The surface a host drives: ``amo_arm`` / ``amo_rpy`` / ``amo_h`` written
before ``get_observation``, ``get_action`` returning the PD target relative to
the default pose, ``post_step_callback`` advancing the gait clock. In this
repository the host is ``robot_control/talos_walk_lower.py``.

Importing this module needs numpy only. torch and the talos_walk tree arrive in
the constructor.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass

import numpy as np

_CFG_NAMES = ("env", "obs", "reward", "command", "clf", "train")


# --------------------------------------------------------------- run on disk
def run_dir(root, run):
    """The directory a ``<robot>/<exp>`` run name resolves to under ``root``."""
    return os.path.join(root, "logs", *run.split("/"))


def read_run_cfg(root, run):
    """``{env, obs, reward, command, clf, train}`` from the run's ``cfgs.pkl``.

    Plain dicts, lists and tuples: it unpickles with the standard library
    alone, which is what lets a caller that never runs the network -- the arm
    engine's joint box, the gain tables -- read the checkpoint's own numbers
    instead of a second copy of them.
    """
    path = os.path.join(run_dir(root, run), "cfgs.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s not found. A talos_walk run is cfgs.pkl + model_<iter>.pt under "
            "<root>/logs/<robot>/<exp>; the controller's --talos_walk_root "
            "names the root."
            % path)
    with open(path, "rb") as f:
        cfgs = pickle.load(f)
    if len(cfgs) != len(_CFG_NAMES):
        raise ValueError("%s holds %d cfg blocks, expected %d (%s)"
                         % (path, len(cfgs), len(_CFG_NAMES),
                            ", ".join(_CFG_NAMES)))
    return dict(zip(_CFG_NAMES, cfgs))


def _asset(root, stored, *relative):
    """A file the run's cfg names by absolute path, found under ``root``.

    ``cfgs.pkl`` stores the paths of the machine that trained it. The layout
    under the workspace root is fixed (``talos/core/workspace.py``), so the
    path under THIS root is preferred and the stored one is only the fallback
    for a file kept outside the layout.
    """
    local = os.path.join(root, *relative)
    if os.path.exists(local):
        return local
    if stored and os.path.exists(stored):
        return stored
    raise FileNotFoundError(
        "%s not found (nor the run's own %r)" % (local, stored))


def joint_ranges(mjcf_path, joint_names):
    """``(lo, hi)`` arrays of the joints' travel, read off the MJCF the run
    was trained on -- the limits its PD targets and its arm command box were
    clipped to, which are not necessarily any other model's."""
    ranges = {}
    for j in ET.parse(mjcf_path).getroot().iter("joint"):
        name, rng = j.get("name"), j.get("range")
        if name and rng:
            lo, hi = (float(v) for v in rng.split())
            ranges[name] = (lo, hi)
    missing = [n for n in joint_names if n not in ranges]
    if missing:
        raise KeyError("%s states no range for %s" % (mjcf_path, missing))
    r = np.array([ranges[n] for n in joint_names], dtype=np.float64)
    return r[:, 0], r[:, 1]


def arm_command_box(env_cfg, dof_lo, dof_hi):
    """``(lo, hi)`` the arm command was sampled across in training, per joint.

    Two shapes, as ``HzdEnv._sample_arm`` has two: each joint's own travel
    inset by ``arm_margin`` of it (``arm_full_range``), or the symmetric
    ``arm_amp`` clipped to the travel.
    """
    amo = env_cfg["amo"]
    sl = slice(int(amo["n_lower"]), int(amo["n_lower"]) + int(amo["n_upper"]))
    lo, hi = np.asarray(dof_lo)[sl], np.asarray(dof_hi)[sl]
    if amo.get("arm_full_range", False):
        pad = float(amo.get("arm_margin", 0.05)) * (hi - lo)
        return lo + pad, hi - pad
    amp = np.asarray(amo["arm_amp"], dtype=np.float64)
    return np.maximum(-amp, lo), np.minimum(amp, hi)


@dataclass(frozen=True)
class RunFacts:
    """What a run's ``cfgs.pkl`` states about commanding it, with no network
    loaded: the shape a caller reads before (or without) building the policy."""
    joint_names: tuple
    default_pos: tuple
    dt: float
    kp: tuple
    kd: tuple
    arm_joints: tuple
    arm_lo: tuple
    arm_hi: tuple
    arm_rate: float          # rad/s the arm command moved at in training; 0 = it never moved
    vx_trained: tuple
    wz_trained: tuple
    command_box: dict        # the WALKING envelope of every torso axis
    stand_box: dict          # the wider envelope training drew while standing
    base_height_nominal: float
    observes_heading: bool
    observes_height: bool
    has_world_model: bool
    action_latency_trained: int


def run_facts(root, run):
    cfg = read_run_cfg(root, run)
    env, command = cfg["env"], cfg["command"]
    names = list(env["joint_names"])
    robot = env.get("robot", run.split("/")[0])
    mjcf = _asset(root, env.get("mjcf_path"), "talos", "robots", robot, "assets",
                  os.path.basename(env.get("mjcf_path", "")))
    dof_lo, dof_hi = joint_ranges(mjcf, names)
    amo = env["amo"]
    n_lower, n_upper = int(amo["n_lower"]), int(amo["n_upper"])
    arm_lo, arm_hi = arm_command_box(env, dof_lo, dof_hi)
    tp = env.get("teleop", {})
    stand = amo.get("stand", {})
    box = {axis: [float(v) for v in amo["%s_range" % axis]]
           for axis in ("height", "roll", "pitch", "yaw")}
    return RunFacts(
        joint_names=tuple(names),
        default_pos=tuple(float(env["default_joint_angles"][n]) for n in names),
        dt=float(env["dt"]),
        kp=tuple(float(v) for v in env["kp"]),
        kd=tuple(float(v) for v in env["kd"]),
        arm_joints=tuple(names[n_lower:n_lower + n_upper]),
        arm_lo=tuple(float(v) for v in arm_lo),
        arm_hi=tuple(float(v) for v in arm_hi),
        arm_rate=(float(max(tp.get("arm_rate", (0.0, 0.0))))
                  if tp.get("arm_motion", False) else 0.0),
        vx_trained=tuple(float(v) for v in command["lin_vel_x_range"]),
        wz_trained=tuple(float(v) for v in command["ang_vel_range"]),
        command_box=box,
        stand_box={axis: [float(v) for v in stand.get("%s_range" % axis, box[axis])]
                   for axis in box},
        base_height_nominal=float(cfg["reward"]["base_height_target"]),
        observes_heading=bool(tp.get("heading_obs", False)),
        observes_height=bool(tp.get("height_obs", False)),
        has_world_model=("adapter" in env
                         and int(env["adapter"].get("resid_len", 0)) > 0),
        action_latency_trained=1 if env.get("simulate_action_latency") else 0,
    )


# ------------------------------------------------------------------ rotations
def _projected_gravity(quat_xyzw):
    """Gravity direction in the base frame, ``R(q)^T [0, 0, -1]``."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([-2.0 * (x * z - w * y),
                     -2.0 * (y * z + w * x),
                     -(1.0 - 2.0 * (x * x + y * y))])


def _yaw(quat_xyzw):
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate(quat_xyzw, v):
    x, y, z, w = (float(c) for c in quat_xyzw)
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


# ------------------------------------------------------------------- networks
def _build_nets(torch, train_cfg, policy_dim, n_actions):
    """The actor of a run, in plain torch, with the parameter names its
    checkpoint was saved under (``mlp.*`` of the rsl_rl actor)."""
    nn = torch.nn
    actor = train_cfg["actor"]
    acts = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}
    try:
        act = acts[actor.get("activation", "elu")]
    except KeyError:
        raise ValueError("actor activation %r has no deploy twin here"
                         % actor.get("activation")) from None
    sizes = list(actor["hidden_dims"]) + [n_actions]
    adapter = "adapter" in str(actor.get("class_name", "")).lower()

    def stack(dims, last_act):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or last_act:
                layers.append(act())
        return nn.Sequential(*layers)

    if not adapter:
        # rsl_rl's MLP: Linear / activation alternating, saved as mlp.<i>.
        return stack([policy_dim] + sizes, last_act=False), None, None

    hist_len, hist_feat = int(actor["hist_len"]), int(actor["hist_feat"])
    resid_len, resid_feat = int(actor.get("resid_len", 0)), int(actor.get("resid_feat", 0))
    embed = int(actor.get("embed_dim", 128))

    class HistEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c, length = [], hist_feat, hist_len
            for f, k, s in zip((64, 64), (9, 6), (5, 3)):
                layers += [nn.Conv1d(c, f, kernel_size=k, stride=s), act()]
                c, length = f, (length - k) // s + 1
            self.conv = nn.Sequential(*layers)
            extra = 0
            self.resid_mlp = None
            if resid_len > 0:
                self.resid_mlp = nn.Sequential(
                    nn.Linear(resid_len * resid_feat, 128), act(),
                    nn.Linear(128, 64), act())
                extra = 64
            self.out = stack([c * length + extra, 256, 256, embed], last_act=False)

        def forward(self, hist, resid):
            x = hist.view(hist.shape[0], hist_len, hist_feat).transpose(1, 2)
            h = self.conv(x).reshape(hist.shape[0], -1)
            if self.resid_mlp is not None:
                h = torch.cat([h, self.resid_mlp(resid)], dim=-1)
            return self.out(h)

    class WorldModel(nn.Module):
        def __init__(self, action_dim, embed_dim, hidden, vel_scale, dt, **_):
            super().__init__()
            self.A = int(action_dim)
            world_dim, delta_dim = 7 + 2 * self.A, 4 + self.A
            self.net = stack([world_dim + int(embed_dim) + self.A] + list(hidden)
                             + [delta_dim], last_act=False)
            self.vel_scale, self.dt = float(vel_scale), float(dt)

        def forward(self, world, embedding, action):
            A = self.A
            delta = self.net(torch.cat([world, embedding, action], dim=-1))
            new_vel = world[..., 6 + A:6 + 2 * A] + delta[..., 3:3 + A]
            new_pos = world[..., 6:6 + A] + new_vel / self.vel_scale * self.dt
            new_h = world[..., 6 + 2 * A:] + delta[..., 3 + A:]
            new_gyro = world[..., :3] + delta[..., :3]
            g = world[..., 3:6]
            new_g = g - torch.cross(new_gyro / self.vel_scale, g, dim=-1) * self.dt
            new_g = new_g / (new_g + 1e-4).norm(dim=-1, keepdim=True)
            return torch.cat([new_gyro, new_g, new_pos, new_vel, new_h], dim=-1)

    class AdapterNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.ModuleList([nn.Linear(policy_dim, sizes[0])])
            self.adapter = nn.ModuleList([nn.Linear(embed, sizes[0])])
            for i in range(1, len(sizes)):
                self.base.append(nn.Linear(sizes[i - 1], sizes[i]))
                self.adapter.append(nn.Linear(sizes[i - 1], sizes[i]))
            self.act = act()
            self.hist_enc = HistEncoder()
            wm = actor.get("world_model_cfg")
            self.world_model = WorldModel(**wm) if wm else None

        def forward(self, obs, hist, resid):
            emb = self.hist_enc(hist, resid)
            b, a = self.base[0](obs), self.adapter[0](emb)
            for i in range(1, len(self.base)):
                out = self.act(b + a)
                b, a = self.base[i](out), self.adapter[i](out)
            return b + a

    net = AdapterNet()
    return net, net.hist_enc, net.world_model


def _quat_wxyz_matrix(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class _Kin:
    """talos_walk's kinematic tree, answered one pose at a time in numpy at an
    identity root -- the shape ``base_state.KinematicBaseHeight`` and the
    strategy read.

    The tree is the run's own ``kintree.npz`` as ``RobotKinematics`` loaded
    it, and the recursion is that class's (``T_body = T_parent . (body_pos,
    body_quat) . Rot(axis, q)`` about the joint's own point), written on
    rotation matrices instead of batched quaternions: the batched form spends
    3 ms of a 20 ms tick on Python overhead at batch size one.
    """

    def __init__(self, kin, feet_idx, sole_center):
        n = lambda a: np.asarray(a.cpu().numpy(), dtype=np.float64)
        self.link_names = list(kin.link_names)
        self.feet_idx = list(feet_idx)
        self.sole_center = np.asarray(sole_center, dtype=np.float64)
        self._parent = [int(v) for v in kin.body_parent]
        self._pos = n(kin.body_pos)
        self._rot = [_quat_wxyz_matrix(q) for q in n(kin.body_quat)]
        self._jnt = [int(v) for v in kin.body_jnt]
        axis = n(kin.body_jnt_axis)
        # The cross-product matrix of each joint axis, for Rodrigues' formula.
        self._K = [np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]],
                             [-a[1], a[0], 0.0]]) for a in axis]
        self._jnt_pos = n(kin.body_jnt_pos)
        self._has_off = bool(np.abs(self._jnt_pos).max() > 1e-9)
        self._link_body = [int(v) for v in kin.link_body]

    def forward(self, dof_pos):
        """``(link_pos [L,3], link_rot [L,3,3])`` in the root's frame."""
        q = np.asarray(dof_pos, dtype=np.float64)
        nb = len(self._parent)
        P, R = [None] * nb, [None] * nb
        eye = np.eye(3)
        for i in range(nb):
            par = self._parent[i]
            if par < 0:
                P[i], R[i] = np.zeros(3), eye
                continue
            p = P[par] + R[par] @ self._pos[i]
            r = R[par] @ self._rot[i]
            j = self._jnt[i]
            if j >= 0:
                K = self._K[i]
                rj = eye + math.sin(q[j]) * K + (1.0 - math.cos(q[j])) * (K @ K)
                if self._has_off:
                    off = self._jnt_pos[i]
                    p = p + r @ (off - rj @ off)
                r = r @ rj
            P[i], R[i] = p, r
        return (np.array([P[b] for b in self._link_body]),
                np.array([R[b] for b in self._link_body]))

    def link_positions(self, dof_pos):
        return self.forward(dof_pos)[0]

    def sole_positions(self, dof_pos):
        """The point of each foot the run's measured height is taken to: the
        foot link's origin plus ``sole_center`` in that link's own frame."""
        p, r = self.forward(dof_pos)
        return np.array([p[i] + r[i] @ self.sole_center for i in self.feet_idx])


class _Ref:
    """What the strategy reads off ``policy.ref``."""

    def __init__(self, kin, base_height_nominal):
        self.kin = kin
        self.feet_idx = kin.feet_idx
        self.base_height_nominal = float(base_height_nominal)
        self.sole_positions = kin.sole_positions


# --------------------------------------------------------------------- policy
class TalosWalkPolicy:
    """One talos_walk run, ticked at its trained ``dt``.

    ``vx_range`` / ``wz_range`` are the DEPLOYED velocity box and must lie
    inside what the run was trained across. ``world_model`` runs the adapter's
    innovation loop (the run was trained with it live; off, the channel is
    zeros). ``mask_aux`` reports the joints after the arm block -- a head --
    at their default pose and at rest: training never commanded them, so a
    host that drives them itself would otherwise show the network joint
    states it has not seen.
    """

    def __init__(self, root, run, checkpoint, *, device="cpu",
                 action_latency_steps=0, vx_range=None, wz_range=None,
                 vx_rate=0.5, wz_rate=1.0, world_model=True, mask_aux=False,
                 heading_hold=True):
        import torch

        self.torch = torch
        self.root, self.run = os.path.abspath(root), run
        self.device = torch.device(device)
        cfg = read_run_cfg(self.root, run)
        self.env_cfg, self.obs_cfg = cfg["env"], cfg["obs"]
        self.command_cfg, self.train_cfg = cfg["command"], cfg["train"]
        self.facts = run_facts(self.root, run)
        env, obs_cfg, command = self.env_cfg, self.obs_cfg, self.command_cfg
        if "anytrack" not in env or not env.get("amo", {}).get("enable"):
            raise ValueError(
                "%s is not an AnyTracker/AnyAdapter run with the AMO command; "
                "this class deploys that observation layout only" % run)

        self.robot = env.get("robot", run.split("/")[0])
        self.joint_names = list(env["joint_names"])
        self.num_actions = A = int(env["num_actions"])             # joints
        idx = env.get("policy_action_idx")
        self.policy_idx = np.asarray(idx if idx else range(A), dtype=np.int64)
        self.n_out = int(self.policy_idx.size)                     # network outputs
        tp0 = env.get("teleop", {})
        self.clamp_margin = float(tp0.get("clamp_margin", 0.0))
        self.heading_hold = bool(heading_hold)
        self.dt = float(env["dt"])
        self.default_pos = np.array(self.facts.default_pos, dtype=np.float64)
        self.default_dof_pos = self.default_pos
        self.action_scale = float(env["anytrack"]["action_scale"])
        self.action_clip = float(env["clip_actions"])
        self.action_latency_steps = int(action_latency_steps)
        self.clamp_pd_target = bool(env.get("clamp_pd_target", False))

        amo = env["amo"]
        self.n_lower, self.n_upper = int(amo["n_lower"]), int(amo["n_upper"])
        self.aux_idx = np.arange(self.n_lower + self.n_upper, A)
        self.mask_aux = (bool(mask_aux) or bool(tp0.get("head_motion", False))) \
            and self.aux_idx.size > 0
        self.amo_cmd_scale = np.asarray(amo["cmd_scale"], dtype=np.float64)
        self.phi_stand_zero = bool(amo.get("phi_stand_zero", False))
        self.stand_enable = bool(amo.get("stand", {}).get("enable", False))
        self.base_height_nominal = self.facts.base_height_nominal

        tp = env.get("teleop", {})
        self.heading_obs = self.facts.observes_heading
        self.height_obs = self.facts.observes_height
        self.heading_err_max = float(tp.get("heading_err_max", 0.5))
        self.stand_blend_s = float(tp.get("stand_blend_s", 0.0))
        self.gait_accel = float(tp.get("gait_accel", 2.0))
        self.arm_live_qd = bool(tp.get("arm_motion", False))
        self.stand_thr_vx = float(command.get("stand_thresh_vx", -1.0))
        self.stand_thr_wz = float(command.get("stand_thresh_wz", -1.0))
        self.turn_reference = bool(env.get("turn_reference", False))
        self.turn_step_max = float(command.get("turn_step_max", 10.0))

        self.ang_vel_scale = float(obs_cfg["ang_vel_scale"])
        self.dof_pos_scale = float(obs_cfg["dof_pos_scale"])
        self.dof_vel_scale = float(obs_cfg["dof_vel_scale"])
        self.command_scale = np.asarray(obs_cfg["command_scale"], dtype=np.float64)
        self.frame_stack = max(1, int(obs_cfg.get("history_len", 1)))

        self.vx_range = self._deployed("vx", vx_range, self.facts.vx_trained)
        self.wz_range = self._deployed("wz", wz_range, self.facts.wz_trained)
        self.vx_rate, self.wz_rate = float(vx_rate), float(wz_rate)

        # ---- talos_walk's own reference modules, from the run's tree
        self._import_talos()
        from talos.reference.phi import PhiWalk
        from talos.rl.gait_library import TorchGaitLibrary
        from talos.rl.kinematics import RobotKinematics

        gait_dir = _asset(self.root, env.get("gait_dir"), "gait_libraries", self.robot)
        phi = _asset(self.root, amo.get("phi_ckpt"), "data", self.robot,
                     os.path.basename(amo.get("phi_ckpt", "phi_walk.pt")))
        kintree = _asset(self.root, env.get("kintree_path"), "data", self.robot,
                         "kintree.npz")
        mjcf = _asset(self.root, env.get("mjcf_path"), "talos", "robots", self.robot,
                      "assets", os.path.basename(env.get("mjcf_path", "")))
        self.gait = TorchGaitLibrary(gait_dir, self.joint_names,
                                     self.default_pos.tolist(), "cpu")
        self.period = self.gait.period
        self.phi = PhiWalk.load(phi, input_dim=self.n_upper + 7,
                                output_dim=self.n_lower, device="cpu")
        kin = RobotKinematics(kintree, device="cpu", dtype=torch.float32)
        if list(kin.joint_names) != self.joint_names:
            raise ValueError("the run's kintree joint order is not its policy's")
        self.kin = _Kin(kin, env["anytrack"]["link_feet"],
                        tp.get("sole_center", [0.0, 0.0, 0.0]))
        self.ref = _Ref(self.kin, self.base_height_nominal)
        self.dof_lo, self.dof_hi = joint_ranges(mjcf, self.joint_names)
        self._hy_left = self.joint_names.index("left_hip_yaw_joint")
        self._hy_right = self.joint_names.index("right_hip_yaw_joint")
        self._default_t = torch.tensor(self.default_pos, dtype=torch.float32).unsqueeze(0)

        # ---- observation widths
        self.frame_dim = (11 + 2 * A + self.n_out + 4 + self.n_upper
                          + int(self.heading_obs) + int(self.height_obs)
                          + 2 * A + 1 + len(self.kin.feet_idx))
        self.policy_dim = self.frame_dim * self.frame_stack
        adapter = env.get("adapter")
        self.is_adapter = adapter is not None
        self.hist_len = int(adapter.get("hist_len", 79)) if adapter else 0
        self.hist_feat, self.world_dim = 6 + 2 * A, 7 + 2 * A
        self.resid_len = int(adapter.get("resid_len", 0)) if adapter else 0
        self.resid_clip = float(adapter.get("resid_clip", 5.0)) if adapter else 0.0
        self.wm_vel_scale = (float(adapter.get("vel_scale", self.dof_vel_scale))
                             if adapter else self.dof_vel_scale)

        self.net, self.hist_enc, self.world_model = _build_nets(
            torch, self.train_cfg, self.policy_dim, self.n_out)
        self.checkpoint = self._load_weights(int(checkpoint))
        self.use_world_model = bool(world_model) and self.world_model is not None \
            and self.resid_len > 0

        self.amo_rpy = np.zeros(3)
        self.amo_h = 0.0
        self.amo_arm_cfg = self.default_pos[self.n_lower:self.n_lower + self.n_upper].copy()
        self.reset()

    # ------------------------------------------------------------ construction
    def _deployed(self, axis, asked, trained):
        box = tuple(float(v) for v in (trained if asked is None else asked))
        if box[0] < trained[0] - 1e-9 or box[1] > trained[1] + 1e-9:
            raise ValueError(
                "the %s command range %s leaves what %s was trained on %s; a "
                "clamp wider than training commands the policy out of "
                "distribution" % (axis, box, self.run, tuple(trained)))
        return list(box)

    def _import_talos(self):
        """Put the run's own tree on the path and insist it is the one that
        answers ``import talos``: a different checkout of the same package --
        an editable install, another body's vendored copy -- would evaluate
        this run's reference with code it was not trained with."""
        if self.root not in sys.path:
            sys.path.append(self.root)
        import talos

        have = os.path.dirname(os.path.dirname(os.path.abspath(talos.__file__)))
        if os.path.realpath(have) != os.path.realpath(self.root):
            raise ImportError(
                "`import talos` resolves to %s, not to the run's own tree %s; "
                "run with that tree first on the path (or uninstall the other)"
                % (have, self.root))

    def _load_weights(self, checkpoint):
        torch = self.torch
        directory = run_dir(self.root, self.run)
        if checkpoint < 0:
            its = sorted(int(f[6:-3]) for f in os.listdir(directory)
                         if f.startswith("model_") and f.endswith(".pt"))
            if not its:
                raise FileNotFoundError("no model_*.pt in %s" % directory)
            checkpoint = its[-1]
        path = os.path.join(directory, "model_%d.pt" % checkpoint)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        actor = sd.get("actor_state_dict") or sd.get("model_state_dict")
        weights = {k[len("mlp."):]: v for k, v in actor.items() if k.startswith("mlp.")}
        self.net.load_state_dict(weights, strict=True)
        self.net.eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.policy_file = path
        return checkpoint

    # ---------------------------------------------------------------- lifecycle
    def reset(self):
        A = self.num_actions
        self.timestep = 0
        self.vx = self.vy = self.wz = 0.0
        self.amo_arm = self.amo_arm_cfg.copy()
        self.last_action = np.zeros(self.n_out)
        self._q_des = self.default_pos.copy()
        self._action_delay = deque(maxlen=max(self.action_latency_steps, 0) + 1)
        self._frames = deque(maxlen=self.frame_stack)
        # The reference's own state, as HzdEnv._reset_idx leaves it for a zero
        # command: settled in the stand, gait clock at its start.
        self._fresh = True
        self.phase_offset = 0.0
        self.stand_alpha = 1.0 if self.stand_enable else 0.0
        self.gait_vx = 0.0
        self.heading_target = 0.0
        self.heading_err = 0.0
        self._arm_prev = None
        self._arm_vel = np.zeros(self.n_upper)
        # adapter
        self._hist = deque(maxlen=max(self.hist_len, 1))
        self._resid = deque(maxlen=max(self.resid_len, 1))
        self._pred_world = None
        self._prev_world = None
        self._prev_latent = None
        self._last_world = None

    def post_step_callback(self, commands=None):
        self.timestep += 1

    # ---------------------------------------------------------------- reference
    def _t(self, v):
        return self.torch.as_tensor(np.asarray(v, dtype=np.float32)).reshape(1, -1)

    def _phi_delta(self, arm, rpy, h, phase, vx, stand):
        t = self.torch
        st = None
        if self.phi_stand_zero and stand is not None:
            st = t.tensor([bool(stand)])
            if stand:
                vx = 0.0
        return self.phi.residual(self._t(arm), self._t(rpy), t.tensor([float(h)]),
                                 t.tensor([float(phase)]), t.tensor([float(vx)]), st)

    def _amo_apply(self, q, qd, phase, vx, stand, arm, rpy, h):
        """``HzdEnv._amo_apply``: PhiWalk residual on the lower block, the arm
        command on the arm block and its velocity as that block's reference."""
        lo, hi = self.n_lower, self.n_lower + self.n_upper
        q[:, :lo] = q[:, :lo] + self._phi_delta(arm, rpy, h, phase, vx, stand)
        q[:, lo:hi] = self._t(arm)
        if qd is not None:
            qd[:, lo:hi] = self._t(self._arm_vel) if self.arm_live_qd else 0.0
        return q, qd

    def _turn_layer(self, q, qd, phase, domain, wz):
        """``HzdEnv._turn_layer``: yaw the body over the planted foot."""
        tau = (phase - 0.5) / 0.5 if domain == 1 else phase / 0.5
        t_half = 0.5 * self.period
        psi = float(np.clip(wz * t_half, -self.turn_step_max, self.turn_step_max))
        a, b = 0.25, 0.85
        u = float(np.clip((tau - a) / (b - a), 0.0, 1.0))
        s = u * u * (3.0 - 2.0 * u)
        ds_dt = 6.0 * u * (1.0 - u) / ((b - a) * t_half)
        stance, swing = -psi * s, -psi * (1.0 - s)
        d_stance, d_swing = -psi * ds_dt, psi * ds_dt
        right = self._hy_right if domain == 0 else self._hy_left
        left = self._hy_left if domain == 0 else self._hy_right
        q[0, right] += stance
        q[0, left] += swing
        qd[0, right] += d_stance
        qd[0, left] += d_swing
        return q, qd

    def _gait(self, vx, time_s):
        t = self.torch
        q, qd, phase, domain = self.gait.reference(
            t.tensor([float(vx)]), t.tensor([float(time_s)]))
        return q, qd, float(phase[0]), int(domain[0])

    def _reference(self, commands, stand, arm, rpy, h):
        """This tick's ``(q_des, qd_des, phase)``, advancing the reference's
        own slewed state exactly as ``HzdEnv._update_reference`` does."""
        vx, wz = float(commands[0]), float(commands[2])
        # the gait clock in float32, as the env computes it (episode_length * dt +
        # phase_offset on float32 tensors): in float64 the clock lands on the other side
        # of a support switch at some cycle wraps, one tick of mirrored reference
        f32 = np.float32
        steps = f32(f32(self.timestep) * f32(self.dt))
        if self.stand_enable and self.stand_blend_s > 0.0:
            rate = self.dt / self.stand_blend_s
            a_new = self.stand_alpha + float(np.clip(float(stand) - self.stand_alpha,
                                                     -rate, rate))
            a_dot = (a_new - self.stand_alpha) / self.dt
            self.stand_alpha = a_new
            dv = self.gait_accel * self.dt
            self.gait_vx += float(np.clip(vx - self.gait_vx, -dv, dv))
            full = a_new >= 1.0 - 1e-6
            if full:
                self.phase_offset = -steps
            q_w, qd_w, phase, domain = self._gait(self.gait_vx, f32(steps + f32(self.phase_offset)))
            if self.turn_reference:
                q_w, qd_w = self._turn_layer(q_w, qd_w, phase, domain, wz)
            q_s = self._default_t.clone()
            qd_s = self.torch.zeros_like(q_s)
            q_w, qd_w = self._amo_apply(q_w, qd_w, phase, self.gait_vx, None, arm, rpy, h)
            q_s, qd_s = self._amo_apply(q_s, qd_s, 0.0, 0.0, True, arm, rpy, h)
            q = (1.0 - a_new) * q_w + a_new * q_s
            qd = (1.0 - a_new) * qd_w + a_new * qd_s + a_dot * (q_s - q_w)
            if full:
                phase = 0.0
        else:
            q, qd, phase, domain = self._gait(vx, f32(steps + f32(self.phase_offset)))
            if self.turn_reference:
                q, qd = self._turn_layer(q, qd, phase, domain, wz)
            if self.stand_enable and stand:
                q, qd, phase = self._default_t.clone(), self.torch.zeros_like(qd), 0.0
            q, qd = self._amo_apply(q, qd, phase, vx,
                                    stand if self.stand_enable else None, arm, rpy, h)
        return (q[0].numpy().astype(np.float64), qd[0].numpy().astype(np.float64),
                float(phase))

    def get_init_dof_pos(self):
        """The standing reference at the resting arm pose and the current
        torso command: the pose the reference holds at a zero base command,
        which is where a handover wants the robot before the network takes
        it. Leaves the reference's own state untouched."""
        q = self._default_t.clone()
        if self.stand_enable:
            q, _ = self._amo_apply(q, None, 0.0, 0.0, True,
                                   self.amo_arm_cfg, self.amo_rpy, self.amo_h)
        else:
            q, _, _, _ = self._gait(0.0, 0.0)
            q, _ = self._amo_apply(q, None, 0.0, 0.0, None,
                                   self.amo_arm_cfg, self.amo_rpy, self.amo_h)
        return q[0].numpy().astype(np.float32)

    # ----------------------------------------------------------------- commands
    def _ramp(self, cur, target, rate):
        step = rate * self.dt
        return float(cur + np.clip(target - cur, -step, step))

    def _get_commands(self, ctrl_data):
        """The base command from a stick deflection, clamped to the deployed
        box and ramped. ``LeftY`` is forward as a fraction of the forward
        bound, ``RightX`` is turn (right positive) as a fraction of the turn
        bound."""
        vx_t = wz_t = 0.0
        jc = (ctrl_data or {}).get("JoystickCtrl")
        if jc is not None and "axes" in jc:
            ax = jc["axes"]
            vx_t = float(ax.get("LeftY", 0.0)) * self.vx_range[1]
            wz_t = -float(ax.get("RightX", 0.0)) * self.wz_range[1]
        self.vx = self._ramp(self.vx, float(np.clip(vx_t, *self.vx_range)), self.vx_rate)
        self.wz = self._ramp(self.wz, float(np.clip(wz_t, *self.wz_range)), self.wz_rate)
        self.vy = 0.0
        return np.array([self.vx, self.vy, self.wz], dtype=np.float64)

    # -------------------------------------------------------------- observation
    def measure_height(self, quat_xyzw, dof_pos):
        """Pelvis height above the lower sole point, from attitude and joints:
        ``HzdEnv._measure_height`` on flat ground."""
        soles = self.kin.sole_positions(dof_pos)
        z = [_rotate(quat_xyzw, s)[2] for s in soles]
        return float(-min(z))

    def get_observation(self, env_data, ctrl_data):
        A = self.num_actions
        commands = self._get_commands(ctrl_data)
        quat = np.asarray(env_data.base_quat, dtype=np.float64)        # xyzw
        ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float64)
        dof_pos = np.array(env_data.dof_pos, dtype=np.float64)
        dof_vel = np.array(env_data.dof_vel, dtype=np.float64)
        if self.mask_aux:
            dof_pos[self.aux_idx] = self.default_pos[self.aux_idx]
            dof_vel[self.aux_idx] = 0.0
        grav = _projected_gravity(quat)
        base_pos = env_data.get("base_pos") if hasattr(env_data, "get") else None
        h_meas = (float(np.asarray(base_pos)[2]) if base_pos is not None
                  else self.measure_height(quat, dof_pos))

        # Heading: the commanded yaw rate integrated into a target, the error
        # clamped and the target re-anchored to the clamp (_update_heading).
        yaw = _yaw(quat)
        if self._fresh:
            self.heading_target, self.heading_err = yaw, 0.0
        else:
            d = self.heading_target + commands[2] * self.dt - yaw
            e = math.atan2(math.sin(d), math.cos(d))
            e = float(np.clip(e, -self.heading_err_max, self.heading_err_max))
            self.heading_err, self.heading_target = e, yaw + e

        # The arm command's own velocity (_advance_arm_command).
        arm = np.asarray(self.amo_arm, dtype=np.float64)
        if self._arm_prev is None:
            self._arm_vel = np.zeros(self.n_upper)
        else:
            self._arm_vel = (arm - self._arm_prev) / self.dt
        self._arm_prev = arm.copy()

        stand = (self.stand_enable and abs(commands[0]) < self.stand_thr_vx
                 and abs(commands[2]) < self.stand_thr_wz)
        rpy = np.asarray(self.amo_rpy, dtype=np.float64)
        h = float(self.amo_h)
        q_des, qd_des, phase = self._reference(commands, stand, arm, rpy, h)
        self._q_des = q_des
        ref_feet_h = self.kin.link_positions(q_des)[self.kin.feet_idx, 2]
        arg = 2.0 * math.pi * phase

        parts = [
            ang_vel * self.ang_vel_scale,
            grav,
            commands * self.command_scale,
            (dof_pos - self.default_pos) * self.dof_pos_scale,
            dof_vel * self.dof_vel_scale,
            self.last_action,
            [math.sin(arg), math.cos(arg)],
            rpy * self.amo_cmd_scale[:3],
            [h * self.amo_cmd_scale[3]],
            arm,
        ]
        if self.heading_obs:
            parts.append([self.heading_err if self.heading_hold else 0.0])
        if self.height_obs:
            parts.append([h_meas])
        q_des_obs = q_des
        if self.mask_aux:                      # the head as training showed it: default pose
            q_des_obs = q_des.copy()
            q_des_obs[self.aux_idx] = self.default_pos[self.aux_idx]
        parts += [
            (q_des_obs - dof_pos) * self.dof_pos_scale,
            (qd_des - dof_vel) * self.dof_vel_scale,
            [self.base_height_nominal + h],
            ref_feet_h,
        ]
        frame = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1)
                                for p in parts]).astype(np.float32)
        if frame.shape[0] != self.frame_dim:
            raise ValueError("observation frame is %d wide, the run's is %d"
                             % (frame.shape[0], self.frame_dim))
        if len(self._frames) == 0:
            for _ in range(self.frame_stack):
                self._frames.append(frame)
        else:
            self._frames.appendleft(frame)                # newest first
        obs = np.concatenate(list(self._frames))
        extras = {"commands": commands, "phase": phase, "stand": bool(stand),
                  "q_des": q_des, "heading_err": self.heading_err, "h_meas": h_meas}

        if self.is_adapter:
            obs = self._adapter_latent(obs, ang_vel, grav, dof_pos, dof_vel, h_meas)
            if self.resid_len > 0:
                extras["wm_residual"] = self._resid[-1].copy()
        self._fresh = False
        return obs, extras

    def _adapter_latent(self, obs, ang_vel, grav, dof_pos, dof_vel, base_z):
        """``AdapterEnv._compute_observations``: history, innovation, and the
        latent the next prediction is formed from."""
        torch = self.torch
        w = np.concatenate([
            ang_vel * self.wm_vel_scale, grav,
            (dof_pos - self.default_pos) * self.dof_pos_scale,
            dof_vel * self.wm_vel_scale, [base_z]]).astype(np.float32)
        f = w[:self.hist_feat]
        if len(self._hist) == 0:
            for _ in range(self.hist_len):
                self._hist.append(f)                      # oldest first
            for _ in range(max(self.resid_len, 1)):
                self._resid.append(np.zeros(self.world_dim, dtype=np.float32))
        else:
            self._hist.append(f)
            r = np.zeros(self.world_dim, dtype=np.float32)
            if self.use_world_model and self._pred_world is not None:
                r = np.clip(w - self._pred_world, -self.resid_clip,
                            self.resid_clip).astype(np.float32)
            self._resid.append(r)
        self._last_world = w
        hist = np.concatenate(list(self._hist))
        parts = [obs, hist]
        resid = None
        if self.resid_len > 0:
            resid = np.concatenate(list(self._resid))
            parts.append(resid)
        if self.use_world_model:
            with torch.no_grad():
                self._prev_latent = self.hist_enc(
                    torch.from_numpy(hist).unsqueeze(0),
                    torch.from_numpy(resid).unsqueeze(0))
            self._prev_world = torch.from_numpy(w).unsqueeze(0)
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------- action
    def get_action(self, obs):
        """The PD target relative to the default pose: the reference plus the
        scaled network output, clipped to the joints' travel when the run was
        trained with its targets clipped there."""
        torch = self.torch
        x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            if self.is_adapter:
                o, hd = self.policy_dim, self.hist_len * self.hist_feat
                a_t = self.net(x[:, :o], x[:, o:o + hd], x[:, o + hd:])
            else:
                a_t = self.net(x)
            a_t = torch.clamp(a_t, -self.action_clip, self.action_clip)
            if self.use_world_model and self._prev_world is not None:
                self._pred_world = self.world_model(
                    self._prev_world, self._prev_latent, a_t)[0].numpy()
        a = a_t[0].numpy().astype(np.float64)
        self.last_action = a.copy()
        self._action_delay.append(a)
        if self.action_latency_steps > 0:
            # An episode starts with an empty action history, which training
            # reads as zero actions (HzdEnv.action_hist).
            a = (self._action_delay[-1 - self.action_latency_steps]
                 if len(self._action_delay) > self.action_latency_steps
                 else np.zeros_like(a))
        # the network drives policy_idx; every other joint takes its reference
        target = self._q_des.copy()
        target[self.policy_idx] += self.action_scale * a
        if self.clamp_pd_target:
            m = self.clamp_margin
            target = np.clip(target, self.dof_lo + m, self.dof_hi - m)
        return target - self.default_pos
