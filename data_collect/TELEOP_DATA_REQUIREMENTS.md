# Teleoperation data requirements for HTD

Use this document to check that a **custom teleoperation system** (not the
Vision Pro collector in this repo) records data that HTD can train on, and that
the resulting policy can be deployed with this repo's
[policy rollout](../policy_rollout/README.md).

It covers the data your teleop must produce: the commanded targets that become
the policy's **actions**. Sensors, tactile formats, file layout and validation
are covered in [TRAINING_DATA_REQUIREMENTS.md](TRAINING_DATA_REQUIREMENTS.md).

---

## 1. The core rule

The policy learns to output the same quantities you record as actions. At
deployment, [policy_rollout.py](../policy_rollout/policy_rollout.py)
(`_execute_g1_action`) converts those outputs into robot commands with a fixed
mapping:

```text
torso_height  = body[2] + 0.75
torso_rpy     = body[3:6]
arm_pos       = eef_to_body[:3] + (0.15, 0.0, 0.45)            # per arm
arm_rot       = R(eef_to_body[3:6]) @ REMAP_side               # per arm
hand_25x3     = fingertips placed at landmark rows 4,9,14,19,24, wrist row 0 = origin,
                rotated by INSPIRE_TO_UNIFIED.T                # per hand
velocity      = velocity_commands                              # [vx, vy, wz]
```

Rollout sends these to the controller in [robot_control](../robot_control/README.md)
as a `TeleopDataWB` DDS message.

**Your teleop must therefore record, at every frame, the inverse of this mapping
applied to the command it actually sent to the robot.** If it does, the trained
policy reproduces your teleop's commands exactly when deployed. If it does not,
the policy trains without errors but sends wrong commands on the robot. Neither
training nor the validator can detect this.

If you also replace the robot controller or rollout, the rule still applies.
Recording and deployment must use the same mapping, whatever it is.

---

## 2. Conventions

| Item | Convention |
| --- | --- |
| Units | meters, radians, m/s, rad/s |
| Rotation encoding | RPY `[roll, pitch, yaw]`, **ZYX order**: `R = Rz(yaw) · Ry(pitch) · Rx(roll)` (see `rpy_to_rot_mat` in the collector) |
| Pose vector | `[x, y, z, roll, pitch, yaw]` |
| Two-sided fields | **right first, then left** |
| Finger order | thumb, index, middle, ring, pinky |
| Rate | one sample per control tick at **30 Hz** |
| "Unified" frame | z up, x forward, fixed at **episode start** (calibration). Origin at the operator's head in the reference collector |

Rotation remaps from [policy_rollout.py](../policy_rollout/policy_rollout.py) (the
collector uses the same matrices):

```python
REMAP_left  = [[1, 0, 0], [0, 0, -1], [0,  1, 0]]   # unified_to_unitree_left_eef_rot_remap
REMAP_right = [[1, 0, 0], [0, 0,  1], [0, -1, 0]]   # unified_to_unitree_right_eef_rot_remap

INSPIRE_TO_UNIFIED_dftp = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]   # inspire_to_unified_eef[:3, :3]
INSPIRE_TO_UNIFIED_f1   = [[0,  0, 1], [0, 1,  0], [-1, 0, 0]]
```

---

## 3. Required teleop signals

These four action streams are what HTD learns to output. They are
**content-critical**: wrong values give a wrong policy.

### 3.1 Torso: `actions/body` `[N, 6]`

| Index | Content | How to compute from your command |
| --- | --- | --- |
| 0, 1 | x, y | **Masked out** of the loss (`masks/act_body = [F,F,T,T,T,T]`). Write any finite value. The reference collector writes the operator's horizontal head offset; zeros also work |
| 2 | height offset | `torso_height_cmd − 0.75` (the WBC default height is 0.75 m) |
| 3, 4, 5 | torso roll, pitch, yaw | The torso RPY command you sent to the WBC |

Stay inside the WBC command range
([g1_student_htd.yaml](../robot_control/g1_wbc_policy/g1_student_htd.yaml)):
height 0.35–0.8 m, roll ±0.5, pitch −0.52…1.22, yaw ±1.27 rad.

### 3.2 Arms: `actions/eef_to_body` `[N, 12]`

The command is `[right(6), left(6)]`, where each side is the wrist target in the
torso/body frame. For each arm, if you sent the IK target position `p_cmd` and
orientation `R_cmd` (Unitree end-effector convention, the frame the controller's
arm IK uses):

```python
pos = p_cmd - np.array([0.15, 0.0, 0.45])
rot = rot_mat_to_rpy(R_cmd @ REMAP_side.T)
eef_to_body_side = np.concatenate([pos, rot])
```

This is exactly how the robot's measured wrist pose is converted into
`observations/proprioceptions/eef_to_body_proprio`. The rotation of your action
and of the proprio therefore share one convention, and their positions differ
by a constant offset. Section 5 uses this to check your frames.

### 3.3 Hands: `actions/eef_joints_target` `[N, 150]`

The layout is `[right(75), left(75)]`. Each side is 25 hand landmarks × xyz,
**relative to the wrist** and expressed in the unified wrist frame. Only the
five fingertip rows are used, for training (preprocessing extracts them as
`actions/{right,left}_eef_link`, 15-D) and at rollout:

| Landmark row | Finger |
| --- | --- |
| 4 | thumb tip |
| 9 | index tip |
| 14 | middle tip |
| 19 | ring tip |
| 24 | pinky tip |

Row 0 (the wrist) must be the origin. The other rows are unused; fill them with
zeros or with your real landmarks.

How to produce the fingertips depends on how your teleop drives the hands:

- **You send 25-landmark arrays to this repo's hand driver.** Take rows
  4/9/14/19/24 of the array you sent (Inspire frame) and convert them:
  `tips_unified = (INSPIRE_TO_UNIFIED @ tips_inspire.T).T`.
- **You have human fingertip positions from another device** (glove, mocap,
  camera tracking). Express them relative to the wrist, in meters and at
  **human-hand scale**. At rollout the controller retargets them with DexPilot
  ([inspire_hand_dftp.yml](../robot_control/assets/inspire_hand/inspire_hand_dftp.yml)),
  which is tuned for human hand geometry. Use the same retargeter during
  teleop, so the robot hand moves during collection exactly as it will during
  rollout.
- **You command Inspire joint angles directly** (no fingertip retargeting).
  HTD's hand action cannot represent this directly. You have two options:
  (a) put this repo's retargeter in your teleop loop and record its fingertip
  input, or (b) change the action head and rollout to use joint targets.
  Robot fingertip positions computed by forward kinematics are **not** a valid
  substitute, because rollout retargets them as if they were human fingertips.

### 3.4 Locomotion: `actions/velocity_commands` `[N, 3]`

This is `[lin_vel_x (m/s), lin_vel_y (m/s), ang_vel_z (rad/s)]`, the **velocity
command the WBC actually executed** on that tick. If a joystick can override
your teleop (as in [g1_wholebody.py](../robot_control/g1_wholebody.py)), record
the effective command after the override, not the teleop request. Ranges:
±0.55 m/s and ±1.57 rad/s.

Record this stream even if the robot never walks (all zeros). For walk profiles
it must contain real walking and turning.

---

## 4. Timing contract

Each frame `t` must contain the command your teleop **sent on tick `t`** and
the observations **sampled on the same tick**. The reference collector does
this, in order, every 1/30 s:

1. Compute the targets from the operator input and publish them to the robot.
2. Read the latest robot state, hand state, tactile and camera frames.
3. Append the targets (actions) and the readings (observations) as row `t`.

Requirements:

- The loop runs at a steady **30 Hz**. Write `frequency = 30` and one
  `observations/head_cam_timestamp` per row, increasing monotonically.
- Every row has both an action and an observation. Do not log the two at
  different rates and align them afterwards.
- Do not leave gaps or pauses mid-episode. Training assumes rows are evenly
  spaced, because the 60-row action and dream horizon is treated as 2 s.
- Do not shift actions by a lag. The policy learns the controller's lag from
  the data; rollout sends `action[t]` on the tick after it observes row `t`.
- Episodes start at calibration (unified frame fixed) and end when the task is
  done. Discard failed demonstrations.

---

## 5. Verifying your teleop

### 5.1 Live checks during teleop

| Check | Expected |
| --- | --- |
| Torso height and pitch commands | Robot crouches and bends as the operator does. Height stays 0.35–0.8 m |
| Each arm, moved alone | The robot arm on the **same side** moves the same way |
| Open hand, then a fist | Inspire hand opens and closes fully. Every finger responds |
| Pinch thumb to each finger | The matching robot finger moves (checks finger order) |
| Joystick or teleop walk | `velocity_commands` in the robot state changes and the robot walks |
| Loop timing | No overrun warnings. Measured rate ≈ 30 Hz |

### 5.2 Offline checks on a recorded episode

Your actions and the robot's proprio share frames (section 3.2), so a short
script can catch most mapping errors. Run it on a few recorded episodes
**before** collecting a full dataset:

```python
"""Sanity-check a custom-teleop HTD episode: python check_teleop.py episode.hdf5"""
import sys, h5py, numpy as np

def best_corr(a, p, max_lag=30):
    """Max correlation of action[t] vs proprio[t + lag] over lags 0..max_lag."""
    if a.std() < 1e-6 or p.std() < 1e-6:
        return None
    c, k = max((np.corrcoef(a[:len(a) - k], p[k:])[0, 1], k) for k in range(max_lag + 1))
    return round(float(c), 3), k

with h5py.File(sys.argv[1], "r") as f:
    lengths = {}
    f.visititems(lambda n, d: lengths.update({n: d.shape[0]})
                 if isinstance(d, h5py.Dataset) and not n.startswith("masks") else None)
    print("frame counts:", sorted(set(lengths.values())), "(must be a single value)")
    print("frequency attr:", f.attrs.get("frequency"))

    ts = f["observations/head_cam_timestamp"][()]
    dt = np.diff(ts)
    print(f"dt median {np.median(dt):.4f}s (target 0.0333), max {dt.max():.4f}s, non-increasing {np.sum(dt <= 0)}")

    act = f["actions/eef_to_body"][()]
    pro = f["observations/proprioceptions/eef_to_body_proprio"][()]
    for name, i in [("R x", 0), ("R y", 1), ("R z", 2), ("L x", 6), ("L y", 7), ("L z", 8)]:
        print(f"arm {name}: (corr, lag) =", best_corr(act[:, i], pro[:, i]))
    print("mean pos offset proprio - action (R):", (pro[:, :3] - act[:, :3]).mean(0).round(3))

    body = f["actions/body"][()]
    bpro = f["observations/proprioceptions/body_proprio"][()]
    for name, i in [("height", 2), ("roll", 3), ("pitch", 4)]:
        print(f"torso {name}: (corr, lag) =", best_corr(body[:, i], bpro[:, i]))

    tips = f["actions/eef_joints_target"][()].reshape(-1, 2, 25, 3)[:, :, [4, 9, 14, 19, 24]]
    robot = f["observations/proprioceptions/other/eef_link_proprio"][()].reshape(-1, 2, 5, 3)
    cos = (tips * robot).sum(-1) / (np.linalg.norm(tips, axis=-1) * np.linalg.norm(robot, axis=-1) + 1e-9)
    print("fingertip direction cosine [right, left] x [thumb..pinky]:\n", cos.mean(0).round(2))
    print("wrist landmark row 0 max |value|:", np.abs(f["actions/eef_joints_target"][()].reshape(-1, 2, 25, 3)[:, :, 0]).max())

    vel = f["actions/velocity_commands"][()]
    print("velocity: frac nonzero", (np.abs(vel) > 0.1).any(1).mean().round(3), "min", vel.min(0), "max", vel.max(0))

    for side in ("left", "right"):
        t = f[f"observations/proprioceptions/other/{side}_touch"][()]
        print(f"{side} touch: frames identical to previous {np.mean(np.all(t[1:] == t[:-1], axis=1)):.2%}")
```

How to read the results (these are heuristics, not hard thresholds):

| Output | Healthy | Indicates a problem |
| --- | --- | --- |
| frame counts | one value | Streams logged at different rates or truncated |
| `dt` | median ≈ 0.033 s, no large max, 0 non-increasing | Loop not at 30 Hz, pauses, or clock errors |
| arm (corr, lag) | high correlation (≈0.8+) at a small lag (a few frames) on axes that moved | Low or negative correlation: wrong frame, swapped axes, or swapped left/right. A large lag means command latency in your stack |
| mean pos offset | roughly constant across episodes | Drifting offset: the action frame is not body-relative |
| torso (corr, lag) | high correlation on axes that moved | Height not offset by 0.75, or RPY order/sign wrong |
| fingertip cosine | clearly positive (fingertip vectors point the same way) | Near 0 or negative: wrong `INSPIRE_TO_UNIFIED`, wrong row order, or fingertips not wrist-relative |
| wrist row 0 | 0 | Fingertips are not expressed relative to the wrist |
| velocity | within ±0.55 / ±1.57; nonzero fraction > 0 for walk tasks | Out-of-range values, or joystick input not recorded |
| touch identical-to-previous | low | High: tactile stream frozen, so the collector keeps re-reading a stale sample |

Correlation is `None` when a signal did not move. Move each arm and the torso
along all axes in at least one test episode.

### 5.3 Final dataset check

After running the two processors, run the repo validator, as described in
[TRAINING_DATA_REQUIREMENTS.md §10](TRAINING_DATA_REQUIREMENTS.md#10-validate-before-training).
It checks the schema, shapes, masks, timing and tactile ranges. It **cannot**
check the action conventions above; only section 5.2 does that.

---

## 6. What else your recorder must write

To use this repo's processors, validator and trainer, write the full raw schema
from [TRAINING_DATA_REQUIREMENTS.md §5](TRAINING_DATA_REQUIREMENTS.md#5-recorded-fields-raw-collector-output),
along with its masks and file attributes (`embodiment = "g1_inspire_wb"`,
`frequency = 30`, `compress`, `hand_type`, `tactile_format`, …). Some teleop
fields are required by the schema but have no effect on HTD; any finite value of
the correct shape works for these:

| Field | Shape | Used by HTD? | Suggested value |
| --- | --- | --- | --- |
| `actions/body`, `eef_to_body`, `eef_joints_target`, `velocity_commands` | 6, 12, 150, 3 | **Yes** (section 3) | As specified |
| `actions/eef` | 12 | No | Wrist targets in the unified frame, or zeros |
| `actions/delta_body`, `delta_eef` | 6, 12 | No | Frame-to-frame differences of `body` / `eef`, or zeros |
| `actions/gripper`, `delta_gripper` | 2, 2 | No | zeros |
| `camera_poses/head_camera_to_init` | 6 | No | zeros |
| `observations/proprioceptions/gripper_proprio` | 2 | No | zeros (the validator warns `placeholder_gripper`; this is expected) |

**Observations** (robot state, hand state, tactile, images) come from the robot,
not the teleop. If you run this repo's robot controller and hand drivers, read
them from its DDS topics exactly as `update_embodiment_proprio_states` does in
[htd_data_collect.py](htd_data_collect.py). This includes the rotation remaps
and the Inspire-to-unified fingertip rotation; skipping them breaks the
action/proprio frame consistency that section 5.2 relies on. If you replace the
controller too, your observations must match the definitions in
[TRAINING_DATA_REQUIREMENTS.md §5.1](TRAINING_DATA_REQUIREMENTS.md#51-observations),
and rollout must compute them the same way.
