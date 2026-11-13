"""A MuJoCo model as a PhysX scene: the same bodies, joints, masses and collision shapes, on a USD stage.

    build(stage, model, data) -> Layout

Run inside Isaac Sim's python. ``model`` is the compiled scene the rest of the stack already uses
(``scene_sim/build_mjcf.py``: the G1 with Inspire hands, the measured objects, the slabs); ``data`` gives the
configuration to build it in (the bodies are placed where ``data.xpos`` has them). What is written:

* one rigid body per MuJoCo body, with its mass, centre of mass and principal inertia;
* the floating robot as one articulation: a revolute joint per hinge (axis, anchor, limits, armature) and a fixed
  joint per welded body; no drives (the host applies joint efforts and sets the damping itself);
* free bodies that are not the robot (the objects) as plain rigid bodies;
* every colliding geom as a collider on its body: meshes by their convex hull (what MuJoCo collides too), boxes,
  spheres, capsules, cylinders, and the ground plane; friction from the geom, combined by ``max`` as MuJoCo does;
* mocap bodies and geoms that collide with nothing are left out (they are markers and looks).

The robot does not collide with itself (most of its geoms cannot in the MuJoCo model either).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROOT = "/World/physx"


@dataclass
class Layout:
    """Where things went: prim paths by MuJoCo name, and which bodies belong to what."""
    robot_root: str = ""  # the articulation root link's prim
    body_path: dict = field(default_factory=dict)  # body name -> prim path
    hinge_joints: list = field(default_factory=list)  # MuJoCo joint names of the articulation's revolute joints
    free_bodies: list = field(default_factory=list)  # names of the free rigid bodies that are not the robot
    robot_bodies: list = field(default_factory=list)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_from_x_to(axis):
    """The rotation taking +X to ``axis`` (wxyz)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    c = float(x @ a)
    if c > 1 - 1e-9:
        return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0, 0, 1.0])
    v = np.cross(x, a)
    q = np.array([1 + c, *v])
    return q / np.linalg.norm(q)


def build(stage, model, data, self_collisions: bool = False, position_iterations: int = 12) -> Layout:
    import mujoco
    from pxr import Gf, PhysxSchema, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    def gq(q):
        return Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))

    def gqd(q):
        return Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))

    lay = Layout()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, ROOT)
    scene = UsdPhysics.Scene.Define(stage, f"{ROOT}/scene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(float(-model.opt.gravity[2]))
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateSolverTypeAttr("TGS")
    px.CreateTimeStepsPerSecondAttr(int(round(1.0 / model.opt.timestep)))
    px.CreateEnableCCDAttr(False)
    px.CreateBroadphaseTypeAttr("MBP")
    px.CreateEnableGPUDynamicsAttr(False)  # one robot: the CPU solver is faster and exact to read back

    # --- which bodies: the robot's tree hangs off the first free joint; other free bodies are objects
    mocap = {b for b in range(model.nbody) if model.body_mocapid[b] >= 0}
    free_root = [int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free_root:
        raise RuntimeError("the model has no floating base")
    robot_root = free_root[0]
    in_robot = {b for b in range(1, model.nbody) if model.body_rootid[b] == robot_root}
    materials = {}

    def material(mu: float):
        key = round(float(mu), 3)
        if key not in materials:
            mat = UsdShade.Material.Define(stage, f"{ROOT}/materials/mu_{str(key).replace('.', '_')}")
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(key)
            api.CreateDynamicFrictionAttr(key)
            api.CreateRestitutionAttr(0.0)
            pm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
            pm.CreateFrictionCombineModeAttr("max")
            pm.CreateRestitutionCombineModeAttr("min")
            materials[key] = mat
        return materials[key]

    def add_geoms(body: int, parent_path: str) -> int:
        n = 0
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body or (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0):
                continue
            gname = Tf.MakeValidIdentifier(model.geom(g).name or f"geom_{g}")
            path = f"{parent_path}/{gname}"
            t, size = int(model.geom_type[g]), model.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(model.geom_dataid[g])
                v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                f = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
                prim = UsdGeom.Mesh.Define(stage, path)
                prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v, np.float32)))
                prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
                prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(f, np.int32).reshape(-1)))
                prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("convexHull")
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                prim = UsdGeom.Cube.Define(stage, path)
                prim.CreateSizeAttr(1.0)
                UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionFloat, "geom").Set(
                    Gf.Vec3f(*(2 * size).tolist()))
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                prim = UsdGeom.Sphere.Define(stage, path)
                prim.CreateRadiusAttr(float(size[0]))
            elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                cls = UsdGeom.Capsule if t == mujoco.mjtGeom.mjGEOM_CAPSULE else UsdGeom.Cylinder
                prim = cls.Define(stage, path)
                prim.CreateRadiusAttr(float(size[0]))
                prim.CreateHeightAttr(float(2 * size[1]))
                prim.CreateAxisAttr("Z")
            elif t == mujoco.mjtGeom.mjGEOM_PLANE:
                prim = UsdGeom.Plane.Define(stage, path)
                prim.CreateAxisAttr("Z")
                prim.CreateWidthAttr(200.0)
                prim.CreateLengthAttr(200.0)
            else:
                continue
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            tr = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            ro = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            tr.Set(Gf.Vec3d(*model.geom_pos[g].tolist()))
            ro.Set(gqd(model.geom_quat[g]))
            xf.SetXformOpOrder([tr, ro] + list(ops))  # translate, orient, then the box's scale
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(
                material(model.geom_friction[g, 0]), UsdShade.Tokens.weakerThanDescendants, "physics")
            n += 1
        return n

    # --- the world's own colliders (ground, slabs)
    UsdGeom.Xform.Define(stage, f"{ROOT}/world")
    add_geoms(0, f"{ROOT}/world")

    # --- bodies
    UsdGeom.Xform.Define(stage, f"{ROOT}/robot")
    for b in range(1, model.nbody):
        if b in mocap:
            continue
        name = model.body(b).name or f"body_{b}"
        robot = b in in_robot
        if not robot and not any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[j] == b
                                 for j in range(model.njnt)) and model.body_parentid[b] == 0:
            # welded to the world: its colliders are static
            path = f"{ROOT}/world/{Tf.MakeValidIdentifier(name)}"
            xf = UsdGeom.Xform.Define(stage, path)
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*data.xpos[b].tolist()))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(gqd(data.xquat[b]))
            add_geoms(b, path)
            lay.body_path[name] = path
            continue
        path = f"{ROOT}/{'robot' if robot else 'objects'}/{Tf.MakeValidIdentifier(name)}"
        if not robot:
            UsdGeom.Xform.Define(stage, f"{ROOT}/objects")
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*data.xpos[b].tolist()))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(gqd(data.xquat[b]))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(max(model.body_mass[b], 1e-5)))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*model.body_ipos[b].tolist()))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*np.maximum(model.body_inertia[b], 1e-9).tolist()))
        mass.CreatePrincipalAxesAttr(gq(model.body_iquat[b]))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateLinearDampingAttr(0.0)
        rb.CreateAngularDampingAttr(0.0)
        rb.CreateMaxLinearVelocityAttr(1000.0)
        rb.CreateMaxAngularVelocityAttr(5730.0)
        rb.CreateSleepThresholdAttr(0.0)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        add_geoms(b, path)
        lay.body_path[name] = path
        (lay.robot_bodies if robot else lay.free_bodies).append(name)

    # --- the robot's joints
    root_path = lay.body_path[model.body(robot_root).name]
    lay.robot_root = root_path
    root_prim = stage.GetPrimAtPath(root_path)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    art.CreateEnabledSelfCollisionsAttr(bool(self_collisions))
    art.CreateSolverPositionIterationCountAttr(int(position_iterations))
    art.CreateSolverVelocityIterationCountAttr(1)
    art.CreateSleepThresholdAttr(0.0)
    UsdGeom.Xform.Define(stage, f"{ROOT}/joints")
    for b in sorted(in_robot):
        if b == robot_root:
            continue
        name, parent = model.body(b).name, int(model.body_parentid[b])
        p_path, c_path = lay.body_path[model.body(parent).name], lay.body_path[name]
        joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
        if len(joints) > 1:
            raise RuntimeError(f"body {name} has {len(joints)} joints; one hinge per body is supported")
        if not joints:
            jn = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/joints/fixed_{Tf.MakeValidIdentifier(name)}")
            anchor, q_axis = np.zeros(3), np.array([1.0, 0, 0, 0])
        else:
            j = joints[0]
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(f"joint {model.joint(j).name} is not a hinge")
            jname = model.joint(j).name
            jn = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/joints/{Tf.MakeValidIdentifier(jname)}")
            jn.CreateAxisAttr("X")
            if model.jnt_limited[j]:
                jn.CreateLowerLimitAttr(float(np.degrees(model.jnt_range[j, 0])))
                jn.CreateUpperLimitAttr(float(np.degrees(model.jnt_range[j, 1])))
            dof = int(model.jnt_dofadr[j])
            pj = PhysxSchema.PhysxJointAPI.Apply(jn.GetPrim())
            pj.CreateArmatureAttr(float(model.dof_armature[dof]))
            pj.CreateJointFrictionAttr(0.0)
            pj.CreateMaxJointVelocityAttr(1e5)
            # an effort-controlled joint still needs a drive for its damping to be set at run time
            drive = UsdPhysics.DriveAPI.Apply(jn.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(float(model.dof_damping[dof]) * np.pi / 180.0)
            drive.CreateMaxForceAttr(1e6)
            anchor, q_axis = model.jnt_pos[j], _quat_from_x_to(model.jnt_axis[j])
            lay.hinge_joints.append(jname)
        jn.CreateBody0Rel().SetTargets([Sdf.Path(p_path)])
        jn.CreateBody1Rel().SetTargets([Sdf.Path(c_path)])
        # the joint frame: at the anchor in the child, X along the axis; the same frame seen from the parent at
        # the joint's zero (the child's rest pose in the parent)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.body_quat[b])
        jn.CreateLocalPos0Attr(Gf.Vec3f(*(model.body_pos[b] + R.reshape(3, 3) @ anchor).tolist()))
        jn.CreateLocalRot0Attr(gq(_quat_mul(model.body_quat[b], q_axis)))
        jn.CreateLocalPos1Attr(Gf.Vec3f(*np.asarray(anchor).tolist()))
        jn.CreateLocalRot1Attr(gq(q_axis))
    return lay
