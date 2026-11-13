"""Put a reconstructed scene (a real2sim RGB-D output folder) into the G1 + Inspire MuJoCo scene.

    python g1_walk/scene_sim/build_mjcf.py --scene-dir ~/TalosRoboticsAI/real2sim/outputs/unified_ep30 \
        [--base g1_walk/teleop-cbf-perception/assets/g1/g1_inspire.xml] [--out g1_walk/scene_sim/out/unified_ep30]

Writes ``OUT/scene.xml`` (pass it to the sim host as PSI_SIM_SCENE), ``OUT/meshes/`` and ``OUT/scene_frame.json``.

Frames. The sim host stands the robot at the MuJoCo origin facing +x, its feet side by side under the pelvis. The
reconstruction's frame has the robot where the episode started (``scene.json["source"]["robot"]``: pelvis pose and
joints). So the MuJoCo world is that start frame, taken at the feet (``--align feet``, the default): ``T_mj_scene``
moves the midpoint of the recorded feet to the origin and turns their mean heading to zero, and every object and
collider is placed through it. The feet, not the pelvis, because HTD's state estimator expresses the torso and
hand poses -- what the policy reads and commands -- in the frame of the feet (feet midpoint, mean foot heading); in
episode 30 the real robot stood with its feet turned 5 deg right of its pelvis (hip yaw), so a scene aligned on the
pelvis puts every commanded hand pose 5 deg to the left of where the real hand went, 3 cm at the beaker
(``--align pelvis`` keeps that older placement). Its height zero is the level of the robot's soles at the start (forward kinematics of the
recorded joints at the recorded pelvis), which need not be the reconstruction's floor: in the recording the camera
is higher above the floor it sees than the robot is tall, by ``feet_above_scene_floor`` (a platform under the robot,
or the depth sensor reading the far floor too deep; the nearby table, rail and objects are placed by the same
transform and keep their place relative to the robot either way). ``scene_frame.json`` records it for the renderer.

What goes in:

* every object of ``layout/scene.json``: its convex collision parts as mesh geoms and its visual mesh (shown, not
  collided with); a free joint when it is dynamic, with its collision parts' density set so that it weighs the
  scene's mass (the inertia is then its own shape's), welded to the world when static;
* a slab for each surface a box-like object stands on (the table under the bin, the rail under the tray): the
  background's collision mesh is one triangle soup, which MuJoCo would replace by its convex hull, so the surfaces
  that matter are fitted as boxes instead: the collision mesh's vertices at that level around the object, their
  minimum-area rectangle, 4 cm thick, top at the level. A slab is never wider than ``--slab-reach`` from the
  object, so it cannot grow into another surface at the same height. The rectangle is then trimmed by the
  recording: the real robot was never inside the real surface, so wherever a recorded robot point lower than the
  slab's top falls inside the rectangle, the rectangle is too wide and that edge is pulled back (``--episode``).

The robot model, its actuators, sensors and keyframe are the base scene's; the keyframe is extended by the free
joints' start poses.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = HERE.parent / "teleop-cbf-perception" / "assets" / "g1" / "g1_inspire.xml"


def quat_wxyz_to_R(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float) / np.linalg.norm(q))
    return R.reshape(3, 3)


def R_to_quat_wxyz(R) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def pose_matrix(pose: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_R(pose["quat_wxyz"])
    T[:3, 3] = pose["pos"]
    return T


def recorded_start(scene: dict, base: Path):
    """The robot model posed as recorded at the start (recorded joints at the recorded pelvis pose, scene frame)."""
    r = scene["source"]["robot"]
    m = mujoco.MjModel.from_xml_path(str(base))
    d = mujoco.MjData(m)
    if m.nkey:
        d.qpos[:] = m.key_qpos[0]
    d.qpos[0:3], d.qpos[3:7] = r["pelvis_pose"]["pos"], r["pelvis_pose"]["quat_wxyz"]
    for n, q in zip(r["joint_names"], r["joint_pos"]):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_forward(m, d)
    return m, d


def feet_frame(scene: dict, base: Path) -> tuple[np.ndarray, float]:
    """Where the recorded feet stood at the start, in the scene frame: the midpoint of the two ankle-roll links
    (x, y) and the heading of their summed forward axes -- the frame HTD's feet-flat state estimator calls world."""
    m, d = recorded_start(scene, base)
    L, R = m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id
    x = d.xmat[L].reshape(3, 3)[:, 0] + d.xmat[R].reshape(3, 3)[:, 0]
    return 0.5 * (d.xpos[L, :2] + d.xpos[R, :2]), float(np.arctan2(x[1], x[0]))


def feet_level(scene: dict, base: Path) -> float:
    """Height of the robot's soles in the scene at the start: forward kinematics of the recorded joint angles at
    the recorded pelvis pose, lowest point of the feet's collision spheres."""
    m, d = recorded_start(scene, base)
    low = []
    for g in range(m.ngeom):
        if "ankle_roll" in m.body(m.geom_bodyid[g]).name and m.geom_contype[g]:
            radius = m.geom_size[g][0] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else m.geom_rbound[g]
            low.append(d.geom_xpos[g][2] - radius)
    return float(min(low))


def start_frame(scene: dict, feet_z: float = 0.0, base: Path | None = None, align: str = "feet") -> np.ndarray:
    """T_mj_scene: the scene frame seen from where the robot stood at the start, at the level of its soles: origin
    at the feet's midpoint, x along their heading (``align="feet"``, needs ``base``), or origin under the pelvis,
    x along its heading (``align="pelvis"``)."""
    if align == "feet":
        xy, yaw = feet_frame(scene, base)
    else:
        T = pose_matrix(scene["source"]["robot"]["pelvis_pose"])
        xy, yaw = T[:2, 3], np.arctan2(T[1, 0], T[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    T_scene_start = np.eye(4)
    T_scene_start[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T_scene_start[:2, 3] = xy
    T_scene_start[2, 3] = feet_z
    return np.linalg.inv(T_scene_start)


def fit_slab(points: np.ndarray, centre_xy, level: float, reach: float, band=0.015, thickness=0.04):
    """The box under ``centre_xy`` whose top is ``level``: (centre, yaw, size) or None."""
    near = points[(np.abs(points[:, 2] - level) < band) & (np.linalg.norm(points[:, :2] - centre_xy, axis=1) < reach)]
    if len(near) < 30:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(near[:, :2].astype(np.float32))
    return np.array([cx, cy, level - thickness / 2]), float(np.radians(ang)), np.array([w, h, thickness])


def recorded_robot_points(episode: Path, base: Path, T_mj_odom: np.ndarray, step=5) -> np.ndarray:
    """Where the real robot was during the recording: the centres of all its collision geoms at every ``step``-th
    frame (forward kinematics of the recorded joints at the recorded odometry), in the MuJoCo frame."""
    import pyarrow.parquet as pq

    meta = json.loads((episode / "ep.json").read_text())
    t = pq.read_table(episode / "frames.parquet", columns=["observation.state", "observation.odometry"])
    st = np.array(t["observation.state"].to_pylist(), float)
    od = np.array(t["observation.odometry"].to_pylist(), float)
    m = mujoco.MjModel.from_xml_path(str(base))
    d = mujoco.MjData(m)
    jmap = []
    for k, n in enumerate(meta["state_names"]):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n[2:]) if n.startswith("q.") else -1
        if j >= 0:
            jmap.append((k, m.jnt_qposadr[j]))
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_contype[g]]
    out = []
    for i in range(0, len(st), step):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = quat_wxyz_to_R(od[i, 9:13]), od[i, 0:3]
        T = T_mj_odom @ T
        if m.nkey:
            d.qpos[:] = m.key_qpos[0]
        d.qpos[0:3], d.qpos[3:7] = T[:3, 3], R_to_quat_wxyz(T[:3, :3])
        for k, adr in jmap:
            d.qpos[adr] = st[i, k]
        mujoco.mj_kinematics(m, d)
        out.append(d.geom_xpos[geoms].copy())
    return np.concatenate(out)


def trim_slab(c, yaw, size, robot: np.ndarray, margin=0.04):
    """Shrink a slab's rectangle until no recorded robot point lies inside the slab grown by ``margin`` on every
    side (the points higher than its top are a hand over the table and constrain nothing; the points well under
    it are knees under a conveyor's edge and constrain nothing either). The real robot was never inside the real
    surface, so a fitted rectangle that contains where its hand hung is too wide there. Each step moves the one
    edge that clears every offending point for the least area. Returns (centre, size, area cut) or None when
    nothing is left."""
    R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    top = c[2] + size[2] / 2
    P = robot[(robot[:, 2] < top - 0.01) & (robot[:, 2] > top - size[2] - margin)]
    lo, hi = -size[:2] / 2, size[:2] / 2  # in the slab's own axes, about its centre
    q = (P[:, :2] - c[:2]) @ R.T
    area0 = float(np.prod(hi - lo))
    for _ in range(8):
        inside = np.all((q > lo - margin) & (q < hi + margin), axis=1)
        if not inside.any():
            break
        best = None
        for ax in (0, 1):
            other = (hi - lo)[1 - ax]
            new_hi, new_lo = q[inside, ax].min() - margin, q[inside, ax].max() + margin
            for side, new, loss in (("hi", new_hi, (hi[ax] - new_hi) * other), ("lo", new_lo, (new_lo - lo[ax]) * other)):
                if best is None or loss < best[0]:
                    best = (loss, ax, side, new)
        _, ax, side, new = best
        if side == "hi":
            hi[ax] = new
        else:
            lo[ax] = new
        if np.any(hi - lo < 0.05):
            return None
    centre = np.r_[c[:2] + ((lo + hi) / 2) @ R, c[2]]
    return centre, np.r_[hi - lo, size[2]], area0 - float(np.prod(hi - lo))


def add_box_collider(body, name: str, extent, floor_top: float, wall: float, friction: float, rgba) -> None:
    """An open box as five slabs in the object's frame (centred, x along the long side): a floor whose top is at
    ``floor_top`` and four walls ``wall`` thick up to the rim. The object's own collision hulls come from a
    generated mesh pressed into the measured size: their inside is lumpy, and what stands in the box tilts and
    creeps. The measured box is flat where the real one is."""
    L, W, H = (float(v) for v in extent)
    floor_top = float(np.clip(floor_top, -H / 2 + 0.002, H / 2 - 0.005))
    parts = {"floor": ((L / 2, W / 2, (floor_top + H / 2) / 2), (0, 0, (floor_top - H / 2) / 2)),
             "wall_xp": ((wall / 2, W / 2, H / 2), (L / 2 - wall / 2, 0, 0)),
             "wall_xn": ((wall / 2, W / 2, H / 2), (-L / 2 + wall / 2, 0, 0)),
             "wall_yp": ((L / 2, wall / 2, H / 2), (0, W / 2 - wall / 2, 0)),
             "wall_yn": ((L / 2, wall / 2, H / 2), (0, -W / 2 + wall / 2, 0))}
    for part, (size, pos) in parts.items():
        body.add_geom(name=f"{name}_collision_{part}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                      rgba=[*rgba[:3], 0.0], group=3, density=_DENSITY0, friction=[friction, 0.02, 0.001])


def add_object(spec, scene_dir: Path, meshes: Path, o: dict, T_mj_scene: np.ndarray, grasp: bool,
               box_floor_top=None, wall: float = 0.005):
    name = o["name"]
    T = T_mj_scene @ pose_matrix(o["pose"])
    body = spec.worldbody.add_body(name=name, pos=T[:3, 3], quat=R_to_quat_wxyz(T[:3, :3]))
    if not o["static"]:
        body.add_freejoint(name=f"{name}_free")
    # visual: the textured mesh as plain geometry in its mean colour (MuJoCo's picture is only the spectator's;
    # the policy's camera is rendered elsewhere)
    vis = trimesh.load(scene_dir / o["visual_mesh"], force="mesh", process=False)
    try:
        rgba = np.asarray(vis.visual.vertex_colors if vis.visual.kind == "vertex"
                          else vis.visual.to_color().vertex_colors, float).mean(0) / 255.0
    except Exception:  # noqa: BLE001
        rgba = np.array([0.6, 0.6, 0.6, 1.0])
    rgba[3] = 1.0
    vfile = meshes / f"{name}_visual.obj"
    trimesh.Trimesh(vis.vertices, vis.faces, process=False).export(vfile)
    spec.add_mesh(name=f"{name}_visual", file=str(vfile))
    body.add_geom(name=f"{name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_visual", rgba=rgba,
                  contype=0, conaffinity=0, group=2, density=0)
    fr = float(o["friction"])
    if box_floor_top is not None:
        add_box_collider(body, name, o["extent"], box_floor_top, wall, fr, rgba)
        return name, T, body
    for k, c in enumerate(o["collision"]):
        cm = trimesh.load(scene_dir / c, force="mesh", process=False)
        cfile = meshes / f"{name}_collision_{k}.obj"
        cm.export(cfile)
        spec.add_mesh(name=f"{name}_collision_{k}", file=str(cfile))
        g = body.add_geom(name=f"{name}_collision_{k}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=f"{name}_collision_{k}", rgba=[*rgba[:3], 0.0], group=3, density=_DENSITY0,
                          friction=[max(fr, 1.0) if grasp else fr, 0.02, 0.001])
        if grasp:  # the thing the hand closes on: torsional friction too, a little compliance
            g.condim = 4
            g.solref = [0.004, 1.0]
    return name, T, body


_DENSITY0 = 1000.0  # collision parts start at this density; a dynamic object's are then scaled to its mass


def scale_to_mass(spec, body, mass: float) -> float:
    """Scale the densities of a body's collision geoms so that it weighs ``mass``: its inertia is then that of
    its own collision shape (the scene file's inertia tensor does not survive per-axis scaling). Returns the
    density."""
    model = spec.compile()
    got = float(model.body_mass[model.body(body.name).id])
    d = _DENSITY0 * mass / got
    for g in body.geoms:
        if g.group == 3:
            g.density = d
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--out", default=None)
    ap.add_argument("--episode", default=None, help="the recorded episode folder (default: the scene's source path)")
    ap.add_argument("--slab-reach", type=float, default=0.6)
    ap.add_argument("--robot-clearance", type=float, default=0.10,
                    help="a slab's edge keeps this far from every recorded robot point lower than its top: the "
                         "simulated robot does not stand exactly as the real one did, and a hand hanging a few "
                         "centimetres further out must not start under the table's edge")
    ap.add_argument("--no-slabs", action="store_true")
    ap.add_argument("--align", choices=("feet", "pelvis"), default="feet",
                    help="what of the recorded start stance goes to the MuJoCo origin facing +x (see Frames above)")
    ap.add_argument("--finger-torque", type=float, default=0.4,
                    help="limit of every finger joint's actuator, N m (0 = the base model's)")
    ap.add_argument("--shift", default=None, help="dx,dy,dz: move the whole scene against the robot (a what-if, "
                    "or a residual of the camera calibration)")
    ap.add_argument("--mesh-colliders", action="store_true",
                    help="keep the measured boxes' mesh collision hulls instead of flat slabs")
    ap.add_argument("--wall", type=float, default=0.005, help="wall thickness of a measured box's slabs (m)")
    a = ap.parse_args(argv)

    scene_dir = Path(a.scene_dir).expanduser().resolve()
    out = (Path(a.out).expanduser() if a.out else HERE / "out" / scene_dir.name).resolve()
    meshes = out / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    scene = json.loads((scene_dir / "layout" / "scene.json").read_text())
    base = Path(a.base).expanduser().resolve()
    feet_z = feet_level(scene, base)
    T_mj_scene = start_frame(scene, feet_z, base, a.align)
    if a.align == "feet":
        xy, yaw = feet_frame(scene, base)
        P = pose_matrix(scene["source"]["robot"]["pelvis_pose"])
        print(f"aligned on the recorded feet: heading {np.degrees(yaw - np.arctan2(P[1, 0], P[0, 0])):+.1f} deg from "
              f"the pelvis's, midpoint {np.round(np.linalg.inv(P)[:2, :2] @ (xy - P[:2, 3]) * 100, 1).tolist()} cm "
              "from it (pelvis frame)")
    if a.shift:  # the scene moved against the robot (m, in the robot's start frame: x forward, y left, z up)
        T_shift = np.eye(4)
        T_shift[:3, 3] = [float(v) for v in a.shift.split(",")]
        T_mj_scene = T_shift @ T_mj_scene
        print(f"scene shifted by {T_shift[:3, 3].tolist()} m against the robot")
    print(f"the robot's soles are {feet_z:.3f} m above the scene's floor at the start: the MuJoCo ground is that level")
    spec = mujoco.MjSpec.from_file(str(base))
    spec.meshdir = str((base.parent / spec.meshdir).resolve()) if spec.meshdir else str(base.parent)
    n_q0 = spec.compile().nq
    if a.finger_torque > 0:
        # the base model lets every finger joint push with 10 N m (some 100 N at a fingertip); an Inspire finger
        # presses with about 10 N. A light cup squeezed that hard is shot out of the hand instead of held
        n_lim = 0
        for act in spec.actuators:
            if re.match(r"(left|right)_(little|ring|middle|index|thumb)_\d_joint$", act.name or ""):
                act.forcelimited = True
                act.forcerange = [-a.finger_torque, a.finger_torque]
                n_lim += 1
        print(f"finger actuators limited to {a.finger_torque} N m: {n_lim}")

    placed, free_q = {}, []
    fits_path = scene_dir / "objects" / "fits.json"
    fits = json.loads(fits_path.read_text()) if fits_path.exists() else {}
    # the fits are in the recording's frame, the scene's poses in the levelled one
    T_so = np.asarray(scene["source"]["T_scene_odom"]) if fits else np.eye(4)

    def to_scene(xyz):
        return (T_so @ np.r_[np.asarray(xyz, float), 1.0])[:3]

    for o in scene["objects"]:  # a cup starts at the height it was measured at: the scene file has it lowered
        # onto its container's mesh, whose inside is not where the real floor is
        for place in fits.get(o["name"], {}).get("places", {}).values():
            pm, ps = to_scene(place["pose"]["pos"]), o["pose"]["pos"]
            if np.hypot(pm[0] - ps[0], pm[1] - ps[1]) < 0.01 and abs(pm[2] - ps[2]) < 0.03:
                print(f"{o['name']}: starts at its measured height, {(pm[2] - ps[2]) * 1000:+.0f} mm from the scene's")
                o["pose"]["pos"] = [ps[0], ps[1], float(pm[2])]
    for o in scene["objects"]:
        grasp = (not o["static"]) and o["mass"] < 0.3
        floor_top = None
        f = fits.get(o["name"], {})
        if f.get("kind") == "box" and o.get("container") and not a.mesh_colliders:
            # a measured open box: flat slabs. Its floor is where the depth saw it, or under what stands in it
            # (an object's start pose is measured better than the dark inside of the box it is in)
            Tb = pose_matrix(o["pose"])
            m = f["measured"]
            floor_top = o.get("inner_floor_z", 0.0)
            if "inner_floor_z" in m:
                floor_top = float(to_scene([*m["centre"], m["inner_floor_z"]])[2] - Tb[2, 3])
            for other in scene["objects"]:
                if other is o or other["static"]:
                    continue
                local = np.linalg.inv(Tb) @ np.r_[other["pose"]["pos"], 1.0]
                if abs(local[0]) < o["extent"][0] / 2 and abs(local[1]) < o["extent"][1] / 2 \
                        and abs(local[2]) < o["extent"][2] / 2 and fits.get(other["name"], {}).get("kind") == "cup":
                    floor_top = float(local[2])  # a cup's frame sits on its base
                    print(f"{o['name']}: floor under {other['name']}, {floor_top + o['extent'][2] / 2:.3f} m "
                          "above its underside")
        name, T, body = add_object(spec, scene_dir, meshes, o, T_mj_scene, grasp, floor_top, a.wall)
        placed[name] = {"static": bool(o["static"]), "T_mj": T.tolist(), "extent": o["extent"], "mass": o["mass"]}
        if not o["static"]:
            free_q.append(np.r_[T[:3, 3], R_to_quat_wxyz(T[:3, :3])])
            placed[name]["density"] = scale_to_mass(spec, body, float(o["mass"]))

    slabs = []
    coll_path = scene_dir / scene["background"].get("collision_mesh", "")
    if not a.no_slabs and fits_path.exists() and coll_path.is_file():
        T_scene_odom = np.asarray(scene["source"]["T_scene_odom"])
        pts = trimesh.load(coll_path, force="mesh", process=False).vertices
        pts = pts @ T_mj_scene[:3, :3].T + T_mj_scene[:3, 3]
        robot_pts = None
        episode = Path(a.episode).expanduser() if a.episode else Path(scene["source"].get("episode", ""))
        if (episode / "frames.parquet").is_file():
            robot_pts = recorded_robot_points(episode, base, T_mj_scene @ T_scene_odom)
            print(f"free space from the recording: {len(robot_pts)} robot points of {episode.name}")
        else:
            print(f"no recording at {episode}: the slabs are not checked against the robot's recorded motion "
                  "(pass --episode)")
        for name, f in fits.items():
            if f.get("kind") != "box" or name not in placed:
                continue
            m = f["measured"]
            lvl = (T_mj_scene @ T_scene_odom @ np.r_[m["centre"], m["stands_on_z"], 1.0])
            fit = fit_slab(pts, lvl[:2], float(lvl[2]), a.slab_reach)
            if fit is None:
                print(f"no surface found under {name} at z {lvl[2]:.3f}")
                continue
            c, yaw, size = fit
            if robot_pts is not None:
                trimmed = trim_slab(c, yaw, size, robot_pts, margin=a.robot_clearance)
                if trimmed is None:
                    print(f"the surface under {name} vanished when trimmed by the recorded robot: skipped")
                    continue
                c2, size2, cut = trimmed
                if cut > 1e-6:
                    print(f"slab under {name}: trimmed from {np.round(size[:2], 3).tolist()} to "
                          f"{np.round(size2[:2], 3).tolist()} m (the recorded robot was inside it)")
                c, size = c2, size2
            q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            spec.worldbody.add_geom(name=f"slab_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size / 2, pos=c, quat=q,
                                    rgba=[0.75, 0.75, 0.75, 1.0], friction=[0.6, 0.005, 0.0001])
            slabs.append({"under": name, "centre": c.tolist(), "yaw": yaw, "size": size.tolist()})
            print(f"slab under {name}: {np.round(size, 3).tolist()} m, top at z {lvl[2]:.3f}")

    for key in spec.keys:  # the base scene's keyframes do not know the new free joints (a compile pads them with
        # a pose at the origin): the robot's part stays, the objects start where the scene has them
        key.qpos = np.concatenate([np.asarray(key.qpos, float)[:n_q0]] + free_q)
    model = spec.compile()
    (out / "scene.xml").write_text(spec.to_xml())
    (out / "scene_frame.json").write_text(json.dumps(
        {"scene_dir": str(scene_dir), "base": str(base), "T_mj_scene": T_mj_scene.tolist(), "objects": placed,
         "slabs": slabs, "feet_above_scene_floor": feet_z, "robot_start_pelvis_scene": scene["source"]["robot"]["pelvis_pose"]}, indent=1))

    # a look at it: does it load from the file, and does anything start in contact?
    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    pen = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = (model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
        if c.dist < -0.001 and (b1 in placed or b2 in placed):
            pen.append((round(float(c.dist) * 1000, 1), b1, b2))
    print(f"{out / 'scene.xml'}: nq {model.nq} (robot {n_q0}), {model.ngeom} geoms, objects {list(placed)}")
    print("initial penetrations over 1 mm (mm, body, body):", sorted(set(pen))[:10] or "none")


if __name__ == "__main__":
    main()
