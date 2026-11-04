# talos_walk: G1 lower-body walking policy

This is the controller's default lower body. It is a walking and balancing policy for the Unitree G1
(29 DoF) that also follows a torso command (roll, pitch, yaw, height) and an operator-driven arm pose.
[`../talos_walk_lower.py`](../talos_walk_lower.py) drives it from
[`../g1_inspire_wb_controller.py`](../g1_inspire_wb_controller.py) (`--lower_body talos_walk`, the
default).

- Run `g1/g1_teleop_anyadapter`, checkpoint 9999 (AnyAdapter on an AnyTracker base, trained in Genesis).
- 50 Hz, CPU, one torch thread. About 2-4 ms per tick.
- World model off and no extra action latency. These are the robot settings.

## What is in this folder

```
deploy/talos_walk_policy.py   deploy class: rebuilds reference + observation, runs the actor, returns PD targets
deploy/manifest.json          every number of the run in machine-readable form (layout, scales, gains, boxes)
talos/                        the reference modules the deploy class imports (gait library, PhiWalk, FK)
talos/robots/g1/assets/       g1_29dof.xml: the joint ranges the run was trained with
data/g1/                      phi_walk.pt (+ meta), kintree.npz
gait_libraries/g1/            HZD gait library: -0.4, -0.2, +0.49, +0.64, +0.77 m/s and the march in place
logs/g1/g1_teleop_anyadapter/ cfgs.pkl (the run's config, plain dicts) and model_9999.pt
```

The folder is a talos_walk workspace root. The deploy class puts it on `sys.path` and refuses to run
if `import talos` resolves to a different copy. The asset paths stored in `cfgs.pkl` are those of the
training machine. The deploy class looks the same files up under this root first.

Both checkpoints load with `torch.load(..., weights_only=True)`.

## Joints, gains, outputs

The joint order is the Unitree motor order: left leg 0-5, right leg 6-11, waist yaw/roll/pitch 12-14,
left arm 15-21, right arm 22-28.

| Slots | kp / kd |
|---|---|
| hips (pitch, roll, yaw) | 100 / 2 |
| knees | 150 / 4 |
| ankles (pitch, roll) | 40 / 2 |
| waist (yaw, roll, pitch) | 100 / 2 |

The network outputs 29 values, and `target = q_des + 0.25 * action`, where `q_des` is the gait
reference. **In this repository only slots 0-14 are used.** The arms stay on the controller's IK, and
the policy's arm outputs are discarded. The policy still observes the IK arm target as its arm command
and reads its own 29 outputs in its last-action slot.

## How the controller drives it

`TalosWalkLowerBody` (in `../talos_walk_lower.py`) adds these around the deploy class:

| What | Setting | Why |
|---|---|---|
| Leg/waist PD target filter | `f += 0.3 * (a - f)`; the network's last-action slot stays unfiltered | Removed the stand-still chatter on the real G1 (`--talos_walk_filter`) |
| Waist roll/pitch velocity | differentiated from position at every `rt/lowstate`, low-passed at 40 Hz | The G1's reported dq of these two linkage joints has been seen to disagree with their motion |
| Arm command the policy sees | clipped to the run's `amo.arm_bounds`, slewed at the trained arm rate | The deploy class's own arm box is wider than training on the inward shoulder roll (±1.40 rad against ±0.077 trained) |
| Torso command | clipped to the walking envelope, slewed at 0.4 rad/s and 0.1 m/s | |
| Walk command | `vx` −0.4…0.8 m/s, `wz` −1.0…1.0 rad/s; clamped and ramped by the deploy class | `vy` does not exist: the run never walked sideways |
| Startup | glide to the policy's stance over 3 s with the leg/waist gains ramping from 20 % over the first 2 s, then a pose check (every slot within 0.15 rad) | A handover far from the stance hands the network a state it never saw |
| Handover | 5 s blend from the stance targets to the policy, gait clock frozen, walk command held at zero | |
| Faults | tilt above 57° or a non-finite action: damping (kp 0, kd 10) and the controller exits | |

The trained torso envelope while walking is roll ±0.3, pitch −0.3…0.6, yaw ±0.6 rad, and height
offset −0.12…+0.02 m (legs straight above +0.02).

## Height convention with HTD

`TeleopDataWB.torso_height` stays absolute, around 0.75 m. The controller sends
`amo_h = torso_height − 0.75` to the policy. Rollout sends `torso_height = body_pose[2] + 0.75`, so
**the recorded `body_pose[2]` must be the height offset that was commanded to talos_walk**, and
`body_pose[3:6]` must be the torso RPY command. A teleop that records anything else trains a policy
whose height and torso commands are wrong at deployment.

## Checks

```bash
cd robot_control
python talos_walk_lower.py        # offline: loads the run, ticks it on a fixed stance, no robot
```

## What training assumed (not measured on the robot)

| Assumption | Trained value |
|---|---|
| Command delay | 0-40 ms, fixed per episode |
| Sensor delay | 0-30 ms, fixed per episode |
| Joint velocity noise | 0.15 rad/s |
| Foot friction | ×0.4 to 2.0 |
| Torso payload | 0-11.7 kg |
| Pushes | up to 1.5 m/s every 4 s |
| Arm command rate | up to 4 rad/s |
| Lateral walking | none: vy is always 0 |

Arms driven only by IK while the network still reads its own arm outputs is a split the policy was
not trained on. Test it with the robot in the gantry before relying on it.
