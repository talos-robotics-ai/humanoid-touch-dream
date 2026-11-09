# Evaluation: validation data, simulation, tests

Three checks, in order of cost: open-loop replay on held-out episodes, an inference
check on the deployment machine, closed-loop replays in simulation.

## 1. Open-loop replay on the validation episodes

`policy_rollout/replay_eval.py` loads a checkpoint exactly as rollout does and runs it
on the held-out episodes recorded in the run's `dataset_split.json`. Every `--stride` frames
(default 5) it predicts an action chunk and compares it with what the operator actually
commanded over the next 60 steps. It needs the processed dataset at the paths recorded in the run (or
`--episodes` to name files), a CUDA GPU and the `htd_policy` or `htd_rollout` environment.

```bash
cd policy_rollout
python replay_eval.py --model_path ../runs/run_white_becker_HTD \
  --checkpoints policy_best.ckpt policy_last.ckpt --interval 45    # policy_best exists once training finished
```

Results go to `<model_path>/replay_eval/`: `summary.md`, `metrics.json`, and one plot
and `.npz` per episode and checkpoint. The summary reports, per checkpoint:

- errors in physical units at the first step of the chunk (`k0`) and averaged over the
  45 executed steps (`window`): hand position [cm], hand rotation [deg], fingertips
  [mm], torso height and rotation, vx and yaw rate;
- the same errors for a "hold" baseline that keeps repeating the current command,
  which gives the scale (a useful policy beats it over the window);
- precision and recall of commanding motion on each velocity axis;
- whole-episode execution on the rollout schedule: tracking over the task, whether each
  grasp and release happens and how early or late, when walking starts and stops, the
  command jump at chunk switches, and the measured inference time;
- a comparison of chunk schedules (`--schedules`, `--latency_ms`), see
  `policy_rollout/README.md`.

Reference, run 1 `policy_last.ckpt` on the nine validation episodes: left hand
3.4 cm at `k0` and 4.2 cm over the window against 4.3 cm for hold; fingertips 2.6 mm
against 6.3 mm; yaw-rate precision 0.74 and recall 0.83. About 1.5 s ahead the policy
is 1.5 to 2 times better than holding the current command.

This is open loop: it finds broken artifacts, wrong units and frames, and policies that
ignore their inputs. It cannot measure task success, because the policy never sees the
consequences of its own commands.

`--save_commands` also writes the command stream rollout would send for each episode,
to `<out>/<checkpoint>/<episode>_commands_<schedule>.npz`; the simulations below replay
these files.

## 2. Inference check on the deployment machine

```bash
cd policy_rollout
python check_inference.py --model_path RUN_DIR --save_reference ref.pt    # on the training machine
python check_inference.py --model_path RUN_DIR --reference ref.pt         # on the robot PC
```

Loads the checkpoint as rollout does, captures the CUDA graph, times inference against
the 30 Hz control budget and, with `--reference`, checks that the robot PC computes the
same actions as the training machine. Exits non-zero on any failure. Measured: 6.6 ms on
the L40S, 10.7 ms on an RTX 5070 Laptop GPU, within 1.3e-5 of each other.

## 3. Closed loop in simulation

The saved command streams are replayed through the same lower-body controller the data
was recorded with, in two simulators. Both live in `g1_walk/` and are described in
[`g1_walk/README.md`](../g1_walk/README.md); the short version:

- **MuJoCo** (`g1_walk/replay/sim_replay.sh --commands EP_commands_SCHEDULE.npz`):
  brings up the teleop's simulated G1, the walking policy and the HTD-side controller
  on one machine, replays the stream, and writes a video and a robot log. `--source
  recorded` replays the operator's commands of the same episode as the baseline;
  `plot_replay.py` compares the two. Run on mains power with the performance profile:
  on battery the stack misses its timing and arms and fingers under-track.
- **Genesis** (`g1_walk/genesis/`): needs a [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
  install and a checkout of the walking policy's training tree (`rollout.py --walk_root`,
  the `talosdeploy-g1/g1_walk` folder). `walker_inputs.py` turns a stream into per-tick
  controller inputs, `rollout.py --lower g1_walk|htd_wbc` rolls it out on either the
  teleop's walking policy or HTD's own whole-body controller, `compare.py`,
  `plot_commands.py`, `render_overlay.py` and `make_clips.py` produce the comparison.
  The operator's recorded commands reproduce the real robot's heading within a few
  degrees on all nine held-out episodes, so Genesis is the reference simulator for
  walking and turning. MuJoCo matches the robot for arms, fingers and turning but drifts
  forward after long turns, a simulator artefact.

What the simulations tell about a checkpoint: whether grasps happen at the right time
with the right hand pose (both runs do), and whether the walking commands reproduce the
recorded heading (run 2 at step 70k is off by -35 to +40 degrees on turns in Genesis).
Forward walking cannot be judged in MuJoCo.

## Unit tests

No robot, camera or dataset is needed. From the repository root, in the `htd_policy` or
`htd_rollout` environment:

```bash
(cd policy_rollout && python test_head_camera.py)        # camera frame layouts and the ZMQ client
(cd policy_rollout && python test_replay_eval.py)        # whole-episode metrics
(cd policy_rollout && python test_chunk_schedule.py)     # chunk blending and async scheduling
(cd policy_rollout && python test_inference_engine.py)   # CUDA graph inference; needs a GPU
python -m unittest discover -s g1_walk/tests -p 'test_*.py'   # command shaping, walk gate, hands
```

Each `test_*.py` also runs under `pytest`. The `g1_walk` tests need the module's
dependencies (`bash g1_walk/setup.sh`). The simulation smoke tests
(`g1_walk/tests/sim_e2e.sh`, `sim_stack.sh`) need the full MuJoCo stack.
