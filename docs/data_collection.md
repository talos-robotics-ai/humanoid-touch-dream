# Data collection: broadening the white-beaker demonstrations

Why: the first deployment (2026-11-09) stalled with the hand in the air whenever the beaker
was outside the narrow region the 99 demonstrations cover, and the policy has no recovery
or walk-to-reach behaviour because no demonstration contains one. This protocol adds those.

## Current coverage (99 episodes, measured)

- Grasp point relative to the torso: forward 0.33 m (std 4 cm), left 0.07 m (std 5 cm),
  up 0.18 m (std 2 cm). Full range about 20 cm in depth, 22 cm sideways.
- Every episode starts at the home pose within reach of the beaker; the arm moves at 1.7 s,
  the grasp happens at about 12 s, walking only after the grasp (carry to the bin).
- 45 mid-task pauses of 1 s or more (operator re-anchoring); the right arm never moves.
- For comparison, the HTD paper used 72 to 193 demonstrations per task with controlled
  object placement.

## Target: 200 to 300 new episodes, plus the existing 99

| Type | Episodes | Start | What to demonstrate |
|---|---|---|---|
| Nominal | 120 | Home pose, robot at the rail as before | Beaker across the whole reachable band: 3 depths (near, centre, far, about 10 cm apart) x 4 lateral positions across the tray, 10 episodes each. Reach, grasp, carry, place in the bin, release. |
| Walk-in | 60 | Home pose, robot 0.3 to 1.0 m back and/or rotated 30 to 45 degrees | Step and turn until the beaker is in reach, then the nominal task. |
| Recovery | 60 | A failure state: hand raised in the air above or beside the beaker, hand 5 to 10 cm left/right/high of it, or robot one step too far back | Correct the hand (or step in), then complete the task. One third of these should start exactly like the deployment stall: arm raised, hand left of and above the beaker. |

## Rules for every episode

- One continuous task, no mid-episode pauses. If you need to re-anchor the teleop reference,
  stop the recording, discard the episode and start a new one.
- Keep episodes under 60 s; end after the release with the arm still, not after a pause.
- Let the right arm rest a little differently from episode to episode (a few cm). Its 1 mm
  variance in the current data is what made normalisation fragile.
- Vary small things freely: tray position on the rail, distractor objects, your own speed.
  Keep the hardware fixed: same G1, Inspire DFTP hands, head camera through the Jetson
  server, same walker.
- Log one row per episode in `episodes.csv`: `episode, type (nominal|walkin|recovery),
  beaker_position (depth/lateral label), start_offset (free text), ok (y/n), notes`.

## After collection

1. Convert, process and validate as in `dataset.md`.
2. `filter_idle_frames.py --dry_run` on the new episodes: expect well under 1% dropped.
   More means pauses crept back in.
3. Check coverage before training: histogram of the grasp point per type and a contact sheet
   of first frames (the scripts used for the 99 episodes are in the deployment analysis notes).
4. Train on old plus new episodes with the same configs and `--norm_std_floor 0.01`
   (`training.md`). Evaluate open loop with `replay_eval.py`, then on the robot with the
   rollout recorder on (`evaluation.md`), starting from the same failure states as the
   recovery episodes.
