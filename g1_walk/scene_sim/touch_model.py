"""The Inspire hands' touch in simulation: what the real hands read when idle, plus a response to contact.

The sim's emulated hands answer their force and tactile registers with zeros. The real ones never do: untouched,
FORCE_ACT sits at an offset per motor (the left hand around [50, 95, 135, 40, 15, 400] counts) and some forty taxels
per hand read a few tens of counts. A policy trained on the real readings therefore sees inputs it never saw in
training before it touches anything. This module fills the registers:

    registers = idle baseline (+ a count or two of noise) + contact response

Build the model from recorded teleop episodes (``observation.hand_pressure``: left skin 1062 | right skin 1062 |
left FORCE_ACT 6 | right FORCE_ACT 6, raw counts):

    python g1_walk/scene_sim/touch_model.py --episodes DATA/task_dir --out g1_walk/scene_sim/out/touch_model.npz

* baseline: each episode's first second (hands open, touching nothing), its median over time. Which taxels read
  something drifts from one episode to the next, so the model keeps every episode's baseline and the one nearest
  to their median is used (``HTD_SIM_TOUCH_EPISODE`` picks another);
* per finger, a skin template: the mean over episodes of (skin while holding - baseline), "holding" being the frames
  where that hand's finger forces are well above idle; and ``skin_per_force``: skin counts (summed over the
  finger's taxels) per count of that finger's FORCE_ACT rise. A hand never used to hold in the recordings has
  baseline only.

At run time (``TouchSynth``, driven by ``scene_sim_host.py`` when ``HTD_SIM_TOUCH`` names the model): the normal
force MuJoCo reports between a finger's bodies and anything that is not the robot becomes a FORCE_ACT rise at
``COUNTS_PER_NEWTON`` (FORCE_ACT is in grams-force: 102 counts per newton), low-pass filtered like the real
reading, and the finger's skin template scaled by ``skin_per_force`` times that rise. The thumb's contact loads
both thumb motors. What this is not: a model of the taxel geometry. The pattern on the skin is the one the
recordings show for this task's grasp, scaled by how hard the simulated finger presses.
"""

import argparse
import json
from pathlib import Path

import numpy as np

N_SKIN, N_FORCE = 1062, 6
# the skin's slices per finger, in the register order (finger one = little ... five = thumb, then the palm), and
# the FORCE_ACT motors each finger loads
FINGERS = (("little", 0, 185, (0,)), ("ring", 185, 370, (1,)), ("middle", 370, 555, (2,)),
           ("index", 555, 740, (3,)), ("thumb", 740, 950, (4, 5)))
COUNTS_PER_NEWTON = 1000.0 / 9.81
FORCE_MAX, SKIN_MAX = 2000.0, 4095.0


def build(episodes_dir: Path, out: Path, idle_frames: int = 30, hold_rise: float = 600.0) -> dict:
    import pyarrow.parquet as pq

    base_skin, base_force, templates, ratios = [], [], {s: [] for s in "lr"}, {s: [] for s in "lr"}
    eps = sorted(p for p in episodes_dir.glob("episode_*") if (p / "frames.parquet").exists())
    for ep in eps:
        a = np.array(pq.read_table(ep / "frames.parquet", columns=["observation.hand_pressure"])[
            "observation.hand_pressure"].to_pylist(), dtype=np.float64)
        if a.shape[1] != 2 * N_SKIN + 2 * N_FORCE or len(a) <= idle_frames:
            continue
        skin = {"l": a[:, :N_SKIN], "r": a[:, N_SKIN:2 * N_SKIN]}
        force = {"l": a[:, 2 * N_SKIN:2 * N_SKIN + N_FORCE], "r": a[:, 2 * N_SKIN + N_FORCE:]}
        bs = {s: np.median(skin[s][:idle_frames], 0) for s in "lr"}
        bf = {s: np.median(force[s][:idle_frames], 0) for s in "lr"}
        base_skin.append(np.concatenate([bs["l"], bs["r"]]))
        base_force.append(np.concatenate([bf["l"], bf["r"]]))
        for s in "lr":
            rise = force[s] - bf[s]
            hold = rise[:, 1:5].sum(1) > hold_rise  # ring..thumb bend well above idle
            if hold.sum() < 30:
                continue
            d_skin = np.clip(np.mean(skin[s][hold], 0) - bs[s], 0.0, None)
            d_force = np.clip(np.mean(rise[hold], 0), 1.0, None)
            templates[s].append(d_skin)
            ratios[s].append([d_skin[a0:a1].sum() / d_force[list(m)].mean() for _, a0, a1, m in FINGERS])
    if not base_skin:
        raise SystemExit(f"no episodes with a {2 * N_SKIN + 2 * N_FORCE}-wide hand_pressure under {episodes_dir}")
    base_skin, base_force = np.array(base_skin), np.array(base_force)
    both = np.hstack([base_skin, base_force])
    typical = int(np.argmin(np.abs(both - np.median(both, 0)).sum(1)))  # the episode nearest to the median
    model = {"baseline_skin": base_skin, "baseline_force": base_force, "typical": typical}
    report = {"episodes": len(base_skin), "typical_episode": typical,
              "baseline_force": base_force[typical].round(1).tolist(),
              "baseline_skin_nonzero": int((base_skin[typical] > 0).sum())}
    for s in "lr":
        used = len(templates[s]) >= max(3, len(base_skin) // 10)  # a hand that holds in a few episodes: no model
        t = np.mean(templates[s], 0) if used else np.zeros(N_SKIN)
        for _, a0, a1, _ in FINGERS:  # each finger's template sums to one
            t[a0:a1] /= max(t[a0:a1].sum(), 1e-9) if t[a0:a1].sum() > 1.0 else np.inf
        model[f"template_{s}"] = t
        model[f"skin_per_force_{s}"] = np.median(ratios[s], 0) if used else np.zeros(len(FINGERS))
        report[f"hold_episodes_{s}"] = len(templates[s])
        report[f"skin_per_force_{s}"] = model[f"skin_per_force_{s}"].round(2).tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **model)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    return report


class TouchSynth:
    """Force and skin registers of both hands from a MuJoCo state. ``read(model, data)`` returns
    {"left": (force counts 6, skin counts 1062), "right": ...} as integers."""

    def __init__(self, model_path, mj_model, tau: float = 0.05, noise: float = 1.5, seed: int = 0, episode=None):
        import mujoco

        self.mj = mujoco
        m = np.load(model_path)
        e = int(m["typical"]) if episode is None else int(episode)
        self.base_skin = {"left": m["baseline_skin"][e, :N_SKIN], "right": m["baseline_skin"][e, N_SKIN:]}
        self.base_force = {"left": m["baseline_force"][e, :N_FORCE], "right": m["baseline_force"][e, N_FORCE:]}
        self.template = {"left": m["template_l"], "right": m["template_r"]}
        self.skin_per_force = {"left": m["skin_per_force_l"], "right": m["skin_per_force_r"]}
        self.tau, self.noise, self.rng = tau, noise, np.random.default_rng(seed)
        self.last_t = None
        self.rise = {s: np.zeros(len(FINGERS)) for s in ("left", "right")}  # filtered, counts
        self.newton = {s: np.zeros(len(FINGERS)) for s in ("left", "right")}  # latest raw contact force
        # body id -> (side, finger index); the robot's bodies (contacts between two of them are not touch)
        self.finger_of, self.robot_root = {}, mj_model.body_rootid[mj_model.body("pelvis").id]
        for b in range(1, mj_model.nbody):
            name = mj_model.body(b).name
            for side in ("left", "right"):
                for k, (finger, *_rest) in enumerate(FINGERS):
                    if name.startswith(f"{side}_{finger}_"):
                        self.finger_of[b] = (side, k)

    def _contact_newtons(self, model, data):
        out = {s: np.zeros(len(FINGERS)) for s in ("left", "right")}
        wrench = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            fa, fb = self.finger_of.get(a), self.finger_of.get(b)
            if (fa is None) == (fb is None):  # neither is a finger, or finger against finger
                continue
            other = b if fa is not None else a
            if other != 0 and model.body_rootid[other] == self.robot_root:
                continue
            self.mj.mj_contactForce(model, data, i, wrench)
            side, k = fa or fb
            out[side][k] += max(float(wrench[0]), 0.0)
        return out

    def read(self, model, data) -> dict:
        t = float(data.time)
        alpha = 1.0 if self.last_t is None else 1.0 - np.exp(-max(t - self.last_t, 0.0) / self.tau)
        self.last_t = t
        self.newton = self._contact_newtons(model, data)
        res = {}
        for side in ("left", "right"):
            self.rise[side] += alpha * (self.newton[side] * COUNTS_PER_NEWTON - self.rise[side])
            force, skin = self.base_force[side].copy(), self.base_skin[side].copy()
            for k, (_, a0, a1, motors) in enumerate(FINGERS):
                rise = self.rise[side][k]
                force[list(motors)] += rise
                skin[a0:a1] += self.template[side][a0:a1] * self.skin_per_force[side][k] * rise
            live = skin > 0  # the taxels that read something flicker by a count or two, like the real ones
            skin[live] += self.rng.normal(0.0, self.noise, int(live.sum()))
            force += self.rng.normal(0.0, self.noise, N_FORCE)
            res[side] = (np.clip(np.rint(force), -FORCE_MAX, FORCE_MAX).astype(int),
                         np.clip(np.rint(skin), 0, SKIN_MAX).astype(int))
        return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True, help="folder of episode_NNNNNN with frames.parquet")
    ap.add_argument("--out", required=True, help="touch_model.npz to write (a .json report goes beside it)")
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.episodes).expanduser(), Path(a.out)), indent=1))


if __name__ == "__main__":
    main()
