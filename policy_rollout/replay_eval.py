"""Open-loop replay evaluation of an MXT/HTD checkpoint on recorded episodes.

For each recorded frame the policy sees exactly what the training loader gives
it (head image, proprioception, tactile) and predicts its action chunk; the
prediction is compared with what the operator actually commanded next. The
model is loaded through ``mxt_loading.load_mxt_policy``, the same code path
``policy_rollout.py`` uses on the robot, so the checkpoint, normalisation
statistics and YAMLs are checked as deployment will use them.

Two views of the same predictions:
  chunk metrics      every --stride frames, error of the predicted chunk at chosen steps;
  whole episode      each episode played start to end on the rollout schedule (infer every
                     --interval steps, execute that many): tracking error over the whole task,
                     grasp/release timing, walking start/stop timing and commanded path,
                     command jumps at chunk switches, and measured inference latency.

This is open loop: the policy never sees the consequences of its own actions.
It catches broken artifacts, wrong frames/units and policies that ignore their
inputs; it cannot show closed-loop task success.

Outputs in --out:
  summary.md                   metric tables per checkpoint (plus a "freeze" baseline)
  metrics.json                 the same numbers, machine readable
  <ckpt>/episode_<n>.png       stitched predicted trajectory vs recording, error over time, path
  <ckpt>/episode_<n>.npz       stitched predictions and ground truth (for sim replay)

Run from policy_rollout/ in the htd_policy environment, e.g.:
  python replay_eval.py --model_path ../runs/rack_white_becker_HTD \\
      --checkpoints policy_best.ckpt policy_last.ckpt
"""
import argparse
import ast
import json
import math
import os
import sys
import time

import numpy as np
import torch

from chunk_schedule import ChunkScheduler
from mxt_loading import load_mxt_policy, resolve_run_configs

EMBODIMENT = 'g1_inspire_wb'
WALK_THRESHOLD = 0.1  # same threshold train_mxt uses for --vel_rebalance


# ---------------------------------------------------------------- geometry

def rpy_to_rot(rpy):
    """ZYX roll/pitch/yaw -> rotation matrices, as policy_rollout.rpy_to_rot_mat (batched)."""
    x, y, z = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    R = np.empty(rpy.shape[:-1] + (3, 3))
    R[..., 0, 0] = cy * cz
    R[..., 0, 1] = cz * sx * sy - cx * sz
    R[..., 0, 2] = cx * cz * sy + sx * sz
    R[..., 1, 0] = cy * sz
    R[..., 1, 1] = cx * cz + sx * sy * sz
    R[..., 1, 2] = cx * sy * sz - cz * sx
    R[..., 2, 0] = -sy
    R[..., 2, 1] = cy * sx
    R[..., 2, 2] = cx * cy
    return R


def rot_err_deg(rpy_a, rpy_b):
    Ra, Rb = rpy_to_rot(rpy_a), rpy_to_rot(rpy_b)
    tr = np.einsum('...ij,...ij->...', Ra, Rb)  # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


# ---------------------------------------------------------------- data

def load_split_episodes(model_path, which):
    with open(os.path.join(model_path, 'dataset_split.json')) as f:
        split = json.load(f)
    entries = split[which]
    if isinstance(entries, str):
        entries = ast.literal_eval(entries)
    return [(e['path'], int(e['frames'])) for e in entries]


def episode_len(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return int(f['/actions/body'].shape[0])


def stack(samples, key):
    return {k: torch.stack([s[key][k] for s in samples]) for k in samples[0][key]}


def predict(policy, obs, obs_mask, horizon_by_head):
    """Batched version of CrossTransformer.forward (which hard-codes batch 1)."""
    model = policy.model
    device = next(model.parameters()).device
    obs = {k: v.to(device) for k, v in obs.items()}
    obs_mask = {k: v.to(device) for k, v in obs_mask.items()}
    batch = next(iter(obs.values())).shape[0]
    feats = model.forward_features(obs, EMBODIMENT, obs_mask)
    out = {}
    for name, head in model.action_heads[EMBODIMENT].items():
        a = head(feats[name]).view(batch, horizon_by_head[name], -1)
        out[name] = model.normalizers[EMBODIMENT][name].unnormalize(a)
    return {k: v.float().cpu().numpy() for k, v in out.items()}


# ---------------------------------------------------------------- metrics

def inference_latency_ms(policy, obs, mask, n=20):
    """Wall time of one rollout-style forward (batch 1) on this machine: median, p95."""
    device = next(policy.model.parameters()).device
    obs = {k: v.to(device) for k, v in obs.items()}
    mask = {k: v.to(device) for k, v in mask.items()}
    times = []
    with torch.no_grad():
        for i in range(n + 3):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            policy(obs, EMBODIMENT, mask)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            if i >= 3:  # warm-up
                times.append(1000 * (time.perf_counter() - t0))
    return {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95))}


def head_errors(name, pred, gt):
    """Per-sample, per-step errors in physical units. pred/gt: (N, T, D)."""
    if name in ('body_pose', 'left_eef_to_body_pose', 'right_eef_to_body_pose'):
        e = {'pos_cm': 100 * np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1),
             'rot_deg': rot_err_deg(pred[..., 3:6], gt[..., 3:6])}
        if name == 'body_pose':
            e['height_cm'] = 100 * np.abs(pred[..., 2] - gt[..., 2])
        return e
    if 'eef_link' in name:
        d = np.linalg.norm((pred - gt).reshape(*pred.shape[:-1], -1, 3), axis=-1)
        return {'tip_mm': 1000 * d.mean(-1)}
    if name == 'velocity_commands':
        return {'vx_abs': np.abs(pred[..., 0] - gt[..., 0]),
                'vy_abs': np.abs(pred[..., 1] - gt[..., 1]),
                'wz_abs': np.abs(pred[..., 2] - gt[..., 2])}
    return {'abs': np.abs(pred - gt).mean(-1)}


def summarise(errs, valid, steps, interval):
    """Mean of each error at chosen chunk steps and over the executed window."""
    out = {}
    for metric, e in errs.items():
        row = {}
        for k in steps:
            m = valid[:, k]
            row[f'k{k}'] = float(e[m, k].mean()) if m.any() else float('nan')
        w = valid[:, :interval]
        row['window'] = float(e[:, :interval][w].mean()) if w.any() else float('nan')
        out[metric] = row
    return out


def walk_scores(pred_v, gt_v):
    """Precision/recall of 'commanding motion' per velocity axis at chunk step 0."""
    out = {}
    for i, axis in enumerate(('vx', 'vy', 'wz')):
        p, g = np.abs(pred_v[:, i]) > WALK_THRESHOLD, np.abs(gt_v[:, i]) > WALK_THRESHOLD
        tp = int((p & g).sum())
        out[axis] = {'gt_active_frac': float(g.mean()), 'pred_active_frac': float(p.mean()),
                     'precision': tp / p.sum() if p.sum() else float('nan'),
                     'recall': tp / g.sum() if g.sum() else float('nan')}
    return out


# ---------------------------------------------------------------- whole-episode execution
#
# Rollout infers at t = 0, interval, 2*interval, ... and executes the first
# `interval` steps of each chunk. Stitching those chunks gives the command
# trajectory the robot would receive over the whole recorded task; the metrics
# below compare it with what the operator commanded, end to end.

EVENT_MIN_S = 0.5    # open/close or walk/stop flickers shorter than this are ignored
EVENT_MATCH_S = 3.0  # a predicted event further than this from a recorded one is "missed"
GRASP_MIN_RANGE_MM = 10.0  # a hand whose fingertips move less than this never grasps


def fingertip_spread_mm(a):
    """Mean fingertip distance from the hand origin: an open/close proxy. a: (T, 15)."""
    return 1000 * np.linalg.norm(a.reshape(len(a), -1, 3), axis=-1).mean(-1)


def debounce(on, min_len):
    """Flip interior runs shorter than min_len frames (shortest first) until none remain."""
    on = np.asarray(on, dtype=bool).copy()
    while True:
        edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
        bounds = np.concatenate([[0], edges, [len(on)]])
        runs = [(e - s, s, e) for s, e in zip(bounds[:-1], bounds[1:]) if s > 0 and e < len(on)]
        short = [r for r in runs if r[0] < min_len]
        if not short:
            return on
        _, s, e = min(short)
        on[s:e] = not on[s]


def events(on):
    """Frames where a boolean state switches, as (frame, 'on' | 'off')."""
    idx = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    return [(int(i), 'on' if on[i] else 'off') for i in idx]


def match_events(gt_ev, pred_ev, max_gap):
    """Greedy one-to-one matching of same-kind events within max_gap frames."""
    free = list(pred_ev)
    offsets = []
    for t, kind in gt_ev:
        cands = [p for p in free if p[1] == kind and abs(p[0] - t) <= max_gap]
        if cands:
            best = min(cands, key=lambda p: abs(p[0] - t))
            free.remove(best)
            offsets.append(best[0] - t)
    return offsets, len(gt_ev) - len(offsets), len(free)


def integrate_path(v, dt):
    """Dead-reckon the commanded (vx, vy, wz) body-frame velocities into a planar path."""
    yaw = np.cumsum(v[:, 2]) * dt
    yaw0 = np.concatenate([[0.0], yaw[:-1]])
    dx = (v[:, 0] * np.cos(yaw0) - v[:, 1] * np.sin(yaw0)) * dt
    dy = (v[:, 0] * np.sin(yaw0) + v[:, 1] * np.cos(yaw0)) * dt
    return np.cumsum(dx), np.cumsum(dy), yaw, float(np.hypot(dx, dy).sum())


EPISODE_METRICS = (  # (label, head, metric from head_errors)
    ('left hand pos [cm]', 'left_eef_to_body_pose', 'pos_cm'),
    ('left hand rot [deg]', 'left_eef_to_body_pose', 'rot_deg'),
    ('left fingertips [mm]', 'left_eef_link_pose', 'tip_mm'),
    ('right hand pos [cm]', 'right_eef_to_body_pose', 'pos_cm'),
    ('right hand rot [deg]', 'right_eef_to_body_pose', 'rot_deg'),
    ('right fingertips [mm]', 'right_eef_link_pose', 'tip_mm'),
    ('torso height [cm]', 'body_pose', 'height_cm'),
    ('torso rot [deg]', 'body_pose', 'rot_deg'),
    ('vx [m/s]', 'velocity_commands', 'vx_abs'),
    ('vy [m/s]', 'velocity_commands', 'vy_abs'),
    ('wz [rad/s]', 'velocity_commands', 'wz_abs'),
)


def episode_execution(gt, pred, hold, interval, fps):
    """Compare the stitched command trajectory with the recording over a whole episode.

    gt/pred/hold: {head: (T, D)}. Returns (json-able metrics, per-frame errors for pooling).
    Heads whose recorded command never changes (e.g. an unused arm) are marked inactive.
    """
    T = len(next(iter(gt.values())))
    tail = max(1, int(round(fps)))  # "where it ends up" = final second
    out = {'frames': T, 'duration_s': T / fps, 'active': {}, 'errors': {}, 'hold_mean': {},
           'chunk_switch': {}, 'grasp': {}}
    per_frame = {}
    for k in gt:
        out['active'][k] = bool(np.nanstd(gt[k], axis=0).max() > 1e-6)
    for label, head, metric in EPISODE_METRICS:
        if head not in gt:
            continue
        e = head_errors(head, pred[head][None], gt[head][None])[metric][0]
        h = head_errors(head, hold[head][None], gt[head][None])[metric][0]
        per_frame[label] = (e, h)
        out['errors'][label] = {'mean': float(np.mean(e)), 'p95': float(np.percentile(e, 95)),
                                'max': float(np.max(e)), 'end': float(np.mean(e[-tail:]))}
        out['hold_mean'][label] = float(np.mean(h))

    # Command discontinuities when rollout switches from one chunk to the next.
    switches = np.arange(interval, T, interval)
    for side in ('left', 'right'):
        pos_k, tip_k = f'{side}_eef_to_body_pose', f'{side}_eef_link_pose'
        if switches.size == 0 or pos_k not in gt or not out['active'].get(pos_k):
            continue
        jump = {}
        for name, a, scale, unit in (('pred', pred, 100, 'cm'), ('recorded', gt, 100, 'cm')):
            p = scale * np.linalg.norm(a[pos_k][switches, :3] - a[pos_k][switches - 1, :3], axis=-1)
            jump[f'{name}_pos_{unit}'] = {'mean': float(p.mean()), 'max': float(p.max())}
            if tip_k in a:
                d = fingertip_spread_mm(a[tip_k])
                f = np.abs(d[switches] - d[switches - 1])
                jump[f'{name}_spread_mm'] = {'mean': float(f.mean()), 'max': float(f.max())}
        out['chunk_switch'][side] = jump

    # Grasp / release timing from the fingertip spread.
    min_len, max_gap = int(EVENT_MIN_S * fps), int(EVENT_MATCH_S * fps)
    for side in ('left', 'right'):
        key = f'{side}_eef_link_pose'
        if key not in gt:
            continue
        g, p = fingertip_spread_mm(gt[key]), fingertip_spread_mm(pred[key])
        lo, hi = np.percentile(g, 5), np.percentile(g, 95)
        if hi - lo < GRASP_MIN_RANGE_MM:
            continue
        thr = (lo + hi) / 2
        g_ev = [(t, 'grasp' if k == 'on' else 'release') for t, k in events(debounce(g < thr, min_len))]
        p_ev = [(t, 'grasp' if k == 'on' else 'release') for t, k in events(debounce(p < thr, min_len))]
        offs, missed, extra = match_events(g_ev, p_ev, max_gap)
        out['grasp'][side] = {'threshold_mm': float(thr), 'recorded': [(t / fps, k) for t, k in g_ev],
                              'predicted': [(t / fps, k) for t, k in p_ev],
                              'offsets_s': [o / fps for o in offs], 'missed': missed, 'extra': extra}

    # Walking: when the policy starts/stops moving, and where the commanded path ends up.
    if 'velocity_commands' in gt:
        vg, vp = gt['velocity_commands'], pred['velocity_commands']
        g_on = debounce((np.abs(vg) > WALK_THRESHOLD).any(-1), min_len)
        p_on = debounce((np.abs(vp) > WALK_THRESHOLD).any(-1), min_len)
        g_ev = [(t, 'start' if k == 'on' else 'stop') for t, k in events(g_on)]
        p_ev = [(t, 'start' if k == 'on' else 'stop') for t, k in events(p_on)]
        offs, missed, extra = match_events(g_ev, p_ev, max_gap)
        xg, yg, yawg, dist_g = integrate_path(vg, 1 / fps)
        xp, yp, yawp, dist_p = integrate_path(vp, 1 / fps)
        out['walk'] = {'recorded': [(t / fps, k) for t, k in g_ev], 'predicted': [(t / fps, k) for t, k in p_ev],
                       'offsets_s': [o / fps for o in offs], 'missed': missed, 'extra': extra,
                       'recorded_walking_s': float(g_on.sum() / fps), 'pred_walking_s': float(p_on.sum() / fps),
                       'recorded_dist_m': dist_g, 'pred_dist_m': dist_p,
                       'end_pos_gap_m': float(np.hypot(xp[-1] - xg[-1], yp[-1] - yg[-1])),
                       'end_heading_gap_deg': float(np.degrees(abs(np.angle(np.exp(1j * (yawp[-1] - yawg[-1])))))),
                       'path_recorded': np.stack([xg, yg], -1), 'path_pred': np.stack([xp, yp], -1)}
    return out, per_frame


# ---------------------------------------------------------------- rollout schedules
#
# Replays an episode through chunk_schedule.ChunkScheduler, the class
# policy_rollout.py uses, with a simulated inference time. Wall time is the
# recording frame. In blocking mode the control loop stops during inference:
# the robot holds its last command while the world (the recording) moves on.
# In async mode commands keep flowing and the result arrives `latency` frames
# after the observation it was computed from.

# The model's head names for chunk_schedule.G1_COMMAND_KINDS.
REPLAY_COMMAND_KINDS = {'body_pose': 'pose6', 'left_eef_to_body_pose': 'pose6', 'right_eef_to_body_pose': 'pose6',
                        'velocity_commands': 'switch'}


def schedule_latency_steps(args, measured_ms):
    """Inference time in control steps: --latency_ms if given, else the time measured here."""
    ms = args.latency_ms if args.latency_ms is not None else measured_ms['median']
    return int(math.ceil(ms * args.fps / 1000.0))


def schedule_set(args):
    if args.schedules.strip().lower() == 'none':
        return {}
    known = {'upstream': (False, 0), 'crossfade': (False, args.crossfade_steps),
             'async': (True, 0), 'async+crossfade': (True, args.crossfade_steps)}
    out = {}
    for name in args.schedules.split(','):
        name = name.strip()
        if name not in known:
            raise SystemExit(f'unknown schedule {name!r}; choose from {", ".join(known)} or none')
        out[name] = {'async': known[name][0], 'crossfade': known[name][1]}
    return out


def simulate_schedule(predict_at, length, interval, crossfade, async_mode, latency, keys):
    """Commands the rollout would send at each recording frame: ({key: (T, D)} with NaN
    before the first chunk, held mask = frames where the loop was blocked in inference)."""
    sched = ChunkScheduler(interval, crossfade, async_mode, kinds=REPLAY_COMMAND_KINDS)
    out, held = [], np.zeros(length, dtype=bool)
    c, inflight, freeze_until, last = 0, None, -1, None
    for w in range(length):
        if w < freeze_until:  # blocking inference: nothing new is sent
            out.append(last)
            held[w] = last is not None
            continue
        if inflight is not None and w >= inflight[3]:
            sched.submit(inflight[2], inflight[1], c, generation=inflight[0], latency_steps=latency)
            inflight = None
        if sched.should_request(c):
            gen = sched.mark_requested(c)
            chunk = predict_at(w)
            if async_mode:
                inflight = (gen, c, chunk, w + latency)
            else:
                sched.submit(chunk, c, c, generation=gen, latency_steps=latency)
                if latency > 0:
                    freeze_until = w + latency
                    out.append(last)
                    held[w] = last is not None
                    continue
        cur = sched.command(c)
        if cur is not None:
            last = cur
        out.append(cur)
        c += 1
    dims = {k: len(np.atleast_1d(predict_at(0)[k][0])) for k in keys}
    cmd = {k: np.full((length, dims[k]), np.nan) for k in keys}
    for w, o in enumerate(out):
        if o is not None:
            for k in keys:
                cmd[k][w] = o[k]
    return cmd, held, sched.overrun_steps


def schedule_episode(gt, cmd, held, interval, fps):
    """Per-episode tracking, grasp timing and smoothness of a simulated command stream."""
    valid = ~np.isnan(next(iter(cmd.values()))[:, 0])
    first = int(np.argmax(valid)) if valid.any() else len(valid)
    g = {k: v[first:] for k, v in gt.items()}
    p = {k: v[first:] for k, v in cmd.items()}
    x, per_frame = episode_execution(g, p, p, interval, fps)
    steps = {}
    for side in ('left', 'right'):
        pk, tk = f'{side}_eef_to_body_pose', f'{side}_eef_link_pose'
        if not x['active'].get(pk):
            continue
        s = {'policy_pos_cm': 100 * np.linalg.norm(np.diff(p[pk][:, :3], axis=0), axis=-1),
             'recording_pos_cm': 100 * np.linalg.norm(np.diff(g[pk][:, :3], axis=0), axis=-1)}
        if tk in g:
            s['policy_spread_mm'] = np.abs(np.diff(fingertip_spread_mm(p[tk])))
            s['recording_spread_mm'] = np.abs(np.diff(fingertip_spread_mm(g[tk])))
        steps[side] = s
    walk = x.get('walk')
    if walk:
        walk = {k: v for k, v in walk.items() if not k.startswith('path_')}
    return {'per_frame': per_frame, 'grasp': x['grasp'], 'walk': walk, 'steps': steps,
            'held': int(held.sum()), 'uncommanded': first, 'frames': len(held)}


def aggregate_schedule(eps, fps):
    res = {'tracking': {}, 'grasp': {}, 'steps': {}}
    for label, *_ in EPISODE_METRICS:
        es = [e['per_frame'][label][0] for e in eps if label in e['per_frame']]
        if es:
            e = np.concatenate(es)
            res['tracking'][label] = {'mean': float(e.mean()), 'p95': float(np.percentile(e, 95))}

    def events(items):
        offs = np.abs(np.concatenate([np.asarray(i['offsets_s'], dtype=float) for i in items] or [np.zeros(0)]))
        return {'found': sum(len(i['recorded']) - i['missed'] for i in items),
                'recorded': sum(len(i['recorded']) for i in items),
                'extra': sum(i['extra'] for i in items),
                'timing_median_s': float(np.median(offs)) if offs.size else float('nan'),
                'timing_max_s': float(offs.max()) if offs.size else float('nan')}
    for side in ('left', 'right'):
        items = [e['grasp'][side] for e in eps if side in e['grasp']]
        if items:
            res['grasp'][side] = events(items)
        parts = [e['steps'][side] for e in eps if side in e['steps']]
        if parts:
            res['steps'][side] = {k: {'p99': float(np.percentile(np.concatenate([p[k] for p in parts]), 99)),
                                      'max': float(np.concatenate([p[k] for p in parts]).max())}
                                  for k in parts[0]}
    walks = [e['walk'] for e in eps if e.get('walk')]
    if walks:
        res['walk'] = events(walks)
    frames = sum(e['frames'] for e in eps)
    res['held_pct'] = 100 * sum(e['held'] for e in eps) / frames
    res['start_delay_s'] = float(np.mean([e['uncommanded'] for e in eps])) / fps
    res['overrun_steps'] = int(sum(e.get('overrun', 0) for e in eps))
    return res


# ---------------------------------------------------------------- plots

def stitch(chunks_by_t, length, interval):
    """Rollout executes `interval` steps of each chunk, then re-infers."""
    any_chunk = next(iter(chunks_by_t.values()))
    traj = np.full((length, any_chunk.shape[-1]), np.nan)
    for t0 in range(0, length, interval):
        if t0 in chunks_by_t:
            n = min(interval, length - t0)
            traj[t0:t0 + n] = chunks_by_t[t0][:n]
    return traj


def plot_episode(path_png, title, gt, stitched, interval, fps, execution=None, per_frame=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(len(gt['body_pose'])) / fps
    rows = 4 if execution is not None else 3
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.3 * rows))
    for ax in axes[:3].ravel()[1:]:
        ax.sharex(axes[0, 0])
    fig.suptitle(f'{title}\nsolid = recording, dashed = policy (open-loop chunks of {interval} steps)')

    def lines(ax, key, dims, labels, scale=1.0, unit=''):
        for d, lab, c in zip(dims, labels, ('C0', 'C1', 'C2')):
            ax.plot(t, scale * gt[key][:, d], c=c, lw=1.2, label=lab)
            ax.plot(t, scale * stitched[key][:, d], c=c, lw=1.2, ls='--')
        ax.set_ylabel(unit)
        ax.legend(loc='upper right', fontsize=8)

    lines(axes[0, 0], 'left_eef_to_body_pose', (0, 1, 2), ('x', 'y', 'z'), 100, 'left hand pos [cm]')
    lines(axes[0, 1], 'right_eef_to_body_pose', (0, 1, 2), ('x', 'y', 'z'), 100, 'right hand pos [cm]')

    def closure(a):  # mean fingertip distance from the hand origin: a hand open/close proxy
        return 1000 * np.linalg.norm(a.reshape(len(a), 5, 3), axis=-1).mean(-1)
    for key, c, lab in (('left_eef_link_pose', 'C0', 'left'), ('right_eef_link_pose', 'C1', 'right')):
        axes[1, 0].plot(t, closure(gt[key]), c=c, label=lab)
        axes[1, 0].plot(t, closure(stitched[key]), c=c, ls='--')
    axes[1, 0].set_ylabel('mean fingertip dist [mm]')
    axes[1, 0].legend(loc='upper right', fontsize=8)

    lines(axes[1, 1], 'left_eef_to_body_pose', (3, 4, 5), ('roll', 'pitch', 'yaw'), 180 / np.pi, 'left hand rot [deg]')
    lines(axes[2, 0], 'body_pose', (2,), ('z',), 100, 'torso z cmd [cm]')
    lines(axes[2, 1], 'velocity_commands', (0, 1, 2), ('vx', 'vy', 'wz'), 1.0, 'velocity cmd')
    for ax in axes[2]:
        ax.set_xlabel('time [s]')

    if execution is not None:
        for side, c in (('left', 'C0'), ('right', 'C1')):  # grasp/release events on the spread plot
            g = execution['grasp'].get(side)
            if g:
                for ts, _ in g['recorded']:
                    axes[1, 0].axvline(ts, c=c, lw=0.8, alpha=0.5)
                for ts, _ in g['predicted']:
                    axes[1, 0].axvline(ts, c=c, lw=0.8, alpha=0.5, ls='--')
        ax = axes[3, 0]
        for label, c in (('left hand pos [cm]', 'C0'), ('left fingertips [mm]', 'C1')):
            if per_frame and label in per_frame:
                e, h = per_frame[label]
                ax.plot(t, e, c=c, label=f'{label} policy')
                ax.plot(t, h, c=c, alpha=0.4, ls=':', label=f'{label} freeze baseline')
        for s in np.arange(interval, len(t), interval) / fps:
            ax.axvline(s, c='0.85', lw=0.6, zorder=0)
        ax.set_ylabel('error vs recording')
        ax.set_xlabel('time [s]  (grey lines = re-inference)')
        ax.legend(loc='upper right', fontsize=7)
        ax = axes[3, 1]
        w = execution.get('walk')
        if w is not None:
            ax.plot(w['path_recorded'][:, 0], w['path_recorded'][:, 1], c='k', label='recorded')
            ax.plot(w['path_pred'][:, 0], w['path_pred'][:, 1], c='C3', ls='--', label='policy')
            ax.plot(0, 0, 'ko', ms=4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]  (integrated velocity commands)')
            ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- main

def evaluate_checkpoint(args, ckpt, episodes, out_dir):
    from mxt_utils import EpisodicDataset

    policy, info = load_mxt_policy(args.model_path, args.embodiment_config_path,
                                   args.trunk_config_path, args.device, checkpoint=ckpt)
    emb = info['emb_dict'][EMBODIMENT]
    horizon = {k: v['action_horizon'] for k, v in emb['action_head'].items()}
    chunk = info['saved_config']['chunk_size']
    if args.interval > chunk:
        raise SystemExit(f'--interval {args.interval} exceeds the chunk size {chunk}')

    paths = [p for p, _ in episodes]
    lens = [n for _, n in episodes]
    dataset = EpisodicDataset(paths, info['stats'], list(range(len(paths))), lens, chunk,
                              main_cam_size=info['main_cam_size'], wrist_cam_size=info['wrist_cam_size'])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])

    ckpt_dir = os.path.join(out_dir, os.path.splitext(os.path.basename(ckpt))[0])
    os.makedirs(ckpt_dir, exist_ok=True)
    action_keys = list(horizon)
    pooled = {k: {'pred': [], 'gt': [], 'pad': [], 'hold': []} for k in action_keys}
    checked = False
    result_episodes, ep_errors = [], {}
    schedules = schedule_set(args)
    schedule_eps = {name: [] for name in schedules}
    t_start = time.time()

    for ep_i, (path, length) in enumerate(episodes):
        frames = sorted(set(range(0, length, args.stride)) | set(range(0, length, args.interval)))
        preds = {k: {} for k in action_keys}
        gts = {k: {} for k in action_keys}
        for b0 in range(0, len(frames), args.batch_size):
            ts = frames[b0:b0 + args.batch_size]
            samples = [dataset[int(starts[ep_i] + t)][0] for t in ts]
            obs, mask, act = stack(samples, 'obs'), stack(samples, 'obs_mask'), stack(samples, 'action')
            is_pad = torch.stack([s['is_pad'] for s in samples]).numpy()
            with torch.no_grad():
                out = predict(policy, dict(obs), dict(mask), horizon)
                if not checked:  # the batched path must equal the rollout forward()
                    single = policy({k: v[:1] for k, v in obs.items()}, EMBODIMENT,
                                    {k: v[:1] for k, v in mask.items()})
                    diff = max(float(np.abs(single[k].float().cpu().numpy()[0] - out[k][0]).max())
                               for k in action_keys)
                    print(f'batched vs rollout forward: max |diff| = {diff:.2e}')
                    if diff > 1e-3:
                        raise SystemExit('batched prediction disagrees with policy.forward')
                    checked = True
                    latency_ms = inference_latency_ms(policy, {k: v[:1] for k, v in obs.items()},
                                                      {k: v[:1] for k, v in mask.items()})
            for k in action_keys:
                gt = act[k].numpy()
                pooled[k]['pred'].append(out[k])
                pooled[k]['gt'].append(gt)
                pooled[k]['pad'].append(is_pad)
                # "hold" baseline: keep sending the current recorded command for the whole
                # chunk. (Measured proprio is in a different frame from the commands.)
                pooled[k]['hold'].append(np.repeat(gt[:, :1], gt.shape[1], axis=1))
                for j, t in enumerate(ts):
                    preds[k][t] = out[k][j]
                    gts[k][t] = gt[j]

        # stitched trajectory, as rollout would command it
        import h5py
        from mxt_utils import KEY_PATH_MAP_G1_INSPIRE_WB as KEYS
        with h5py.File(path, 'r') as f:
            gt_full = {k: f[KEYS[k]][()].astype(np.float64) for k in action_keys}
        stitched = {k: stitch(preds[k], length, args.interval) for k in action_keys}
        # Freeze baseline on the same schedule: hold the command recorded at each re-inference.
        held = {k: stitch({t0: np.repeat(gt_full[k][t0:t0 + 1], args.interval, 0)
                           for t0 in range(0, length, args.interval)}, length, args.interval)
                for k in action_keys}
        execution, per_frame = episode_execution(gt_full, stitched, held, args.interval, args.fps)
        name = os.path.splitext(os.path.basename(path))[0]
        walk = execution.get('walk', {})
        paths = {f'path_{k}': walk.pop(f'path_{k}') for k in ('recorded', 'pred') if f'path_{k}' in walk}
        np.savez_compressed(os.path.join(ckpt_dir, f'{name}.npz'), interval=args.interval,
                            **{f'pred_{k}': v for k, v in stitched.items()},
                            **{f'gt_{k}': v for k, v in gt_full.items()}, **paths)
        if not args.no_plots:
            walk.update(paths)
            plot_episode(os.path.join(ckpt_dir, f'{name}.png'), f'{os.path.basename(ckpt)} - {name}',
                         gt_full, stitched, args.interval, args.fps, execution, per_frame)
            for k in paths:
                walk.pop(k)
        execution['episode'] = name
        result_episodes.append(execution)
        for label, (e, h) in per_frame.items():
            ep_errors.setdefault(label, ([], []))
            ep_errors[label][0].append(e)
            ep_errors[label][1].append(h)

        if schedules:
            latency = schedule_latency_steps(args, latency_ms)

            def predict_at(t, ep_i=ep_i):
                if t not in preds[action_keys[0]]:
                    s = dataset[int(starts[ep_i] + t)][0]
                    with torch.no_grad():
                        o = predict(policy, stack([s], 'obs'), stack([s], 'obs_mask'), horizon)
                    for k in action_keys:
                        preds[k][t] = o[k][0]
                return {k: preds[k][t] for k in action_keys}
            for sname, cfg in schedules.items():
                cmd, held_mask, overrun = simulate_schedule(predict_at, length, args.interval, cfg['crossfade'],
                                                            cfg['async'], latency, action_keys)
                ep = schedule_episode(gt_full, cmd, held_mask, args.interval, args.fps)
                ep['overrun'] = overrun
                schedule_eps[sname].append(ep)
                if args.save_commands:
                    # What rollout would send at each 30 Hz step (NaN before the first chunk),
                    # with the operator's recorded commands, for replay through a controller.
                    np.savez_compressed(os.path.join(ckpt_dir, f'{name}_commands_{sname}.npz'),
                                        fps=args.fps, interval=args.interval, latency_steps=latency,
                                        crossfade_steps=cfg['crossfade'], async_inference=cfg['async'],
                                        **{f'cmd_{k}': v for k, v in cmd.items()},
                                        **{f'gt_{k}': v for k, v in gt_full.items()})
        print(f'  {name}: {len(frames)} predictions ({time.time() - t_start:.0f}s)')

    steps = sorted({0, min(15, chunk - 1), args.interval - 1})
    result = {'checkpoint': info['checkpoint'], 'heads': {}, 'hold_baseline': {}, 'normalised_window': {}}
    stats = info['stats'][EMBODIMENT]['action']
    for k in action_keys:
        pred = np.concatenate(pooled[k]['pred']).astype(np.float64)
        gt = np.concatenate(pooled[k]['gt']).astype(np.float64)
        valid = ~np.concatenate(pooled[k]['pad'])
        result['heads'][k] = summarise(head_errors(k, pred, gt), valid, steps, args.interval)
        hold = np.concatenate(pooled[k]['hold']).astype(np.float64)
        result['hold_baseline'][k] = summarise(head_errors(k, hold, gt), valid, steps, args.interval)
        std = np.asarray(stats[k]['std'], dtype=np.float64)
        keep = std > 1e-6
        z = np.abs(pred - gt)[..., keep] / std[keep]
        w = valid[:, :args.interval]
        result['normalised_window'][k] = float(z[:, :args.interval][w].mean())
        if k == 'velocity_commands':
            result['walking'] = walk_scores(pred[:, 0], gt[:, 0])

    pooled_exec = {}
    for label, (es, hs) in ep_errors.items():
        e, h = np.concatenate(es), np.concatenate(hs)
        pooled_exec[label] = {'mean': float(e.mean()), 'p95': float(np.percentile(e, 95)),
                              'max': float(e.max()), 'hold_mean': float(h.mean())}
    result['execution'] = {'episodes': result_episodes, 'pooled': pooled_exec,
                           'inference_ms': latency_ms, 'interval': args.interval, 'fps': args.fps}
    if schedules:
        result['schedules'] = {
            'latency_steps': schedule_latency_steps(args, latency_ms),
            'crossfade_steps': args.crossfade_steps,
            'results': {n: {**cfg, **aggregate_schedule(eps, args.fps)}
                        for (n, cfg), eps in zip(schedules.items(), schedule_eps.values())}}
    policy = None  # release the model's GPU memory before the next checkpoint loads
    torch.cuda.empty_cache()
    return result


def execution_summary(x, f):
    """Markdown for the whole-episode execution results of one checkpoint."""
    eps, interval, fps = x['episodes'], x['interval'], x['fps']
    head_of = {label: head for label, head, _ in EPISODE_METRICS}
    active = {h for e in eps for h, on in e['active'].items() if on}
    lat = x['inference_ms']
    lines = [f'### Whole-episode execution ({len(eps)} episodes, '
             f'{sum(e["duration_s"] for e in eps):.0f} s of recording)', '',
             f'Each episode is played from start to end on the rollout schedule: infer at t = 0, {interval}, '
             f'{2 * interval}, ..., execute {interval} steps ({interval / fps:.1f} s) of each chunk. '
             'Still open loop: the policy sees the recorded observations, not the results of its own commands. '
             '"freeze" = hold the command recorded at each re-inference for the whole chunk.', '',
             f'Inference on this machine: {lat["median"]:.0f} ms median, {lat["p95"]:.0f} ms p95 '
             f'(rollout infers inside the {fps:.0f} Hz control loop, so commands pause for '
             f'~{lat["median"] / 1000 * fps:.1f} control steps at every re-inference).', '',
             '| quantity | mean | p95 | max | freeze mean |', '|---|---|---|---|---|']
    for label, row in x['pooled'].items():
        if head_of[label] not in active:
            continue
        fmt = '{:.3f}' if '/s' in label else '{:.1f}'
        lines.append(f"| {label} | {f(row['mean'], fmt)} | {f(row['p95'], fmt)} | {f(row['max'], fmt)} "
                     f"| {f(row['hold_mean'], fmt)} |")

    def offs(ev):
        o = np.abs(ev['offsets_s']) if ev['offsets_s'] else None
        return '-' if o is None else f'{np.median(o):.2f} / {o.max():.2f}'

    sides = [s for s in ('left', 'right') if any(s in e['grasp'] for e in eps)]
    for side in sides:
        lines += ['', f'{side.capitalize()} hand over each episode (grasp = fingertips closed past the midpoint '
                  'of the recorded open/close range; events matched within '
                  f'{EVENT_MATCH_S:.0f} s):', '',
                  '| episode | dur [s] | hand pos mean / max [cm] | hand pos at end [cm] | fingertips mean [mm] '
                  '| grasp+release found | extra | timing error median / max [s] |',
                  '|---|---|---|---|---|---|---|---|']
        for e in eps:
            g = e['grasp'].get(side)
            pos = e['errors'].get(f'{side} hand pos [cm]', {})
            tip = e['errors'].get(f'{side} fingertips [mm]', {})
            found = '-' if g is None else f"{len(g['recorded']) - g['missed']}/{len(g['recorded'])}"
            lines.append(f"| {e['episode']} | {e['duration_s']:.1f} | {f(pos.get('mean'))} / {f(pos.get('max'))} "
                         f"| {f(pos.get('end'))} | {f(tip.get('mean'))} | {found} "
                         f"| {'-' if g is None else g['extra']} | {'-' if g is None else offs(g)} |")
    if any('walk' in e for e in eps):
        lines += ['', 'Walking over each episode (commanded path = integrated velocity commands, not the '
                  'robot\'s measured path):', '',
                  '| episode | walking rec / policy [s] | start+stop found | extra | timing error median / max [s] '
                  '| path length rec / policy [m] | path end gap [m] | heading gap [deg] |',
                  '|---|---|---|---|---|---|---|---|']
        for e in eps:
            w = e.get('walk')
            if w is None:
                continue
            lines.append(f"| {e['episode']} | {w['recorded_walking_s']:.1f} / {w['pred_walking_s']:.1f} "
                         f"| {len(w['recorded']) - w['missed']}/{len(w['recorded'])} | {w['extra']} | {offs(w)} "
                         f"| {w['recorded_dist_m']:.2f} / {w['pred_dist_m']:.2f} | {w['end_pos_gap_m']:.2f} "
                         f"| {w['end_heading_gap_deg']:.0f} |")
    jumps = [e['chunk_switch'][s] for e in eps for s in e['chunk_switch']]
    if jumps:
        def agg(key, stat):
            v = [j[key][stat] for j in jumps if key in j]
            return (np.mean(v) if stat == 'mean' else np.max(v)) if v else float('nan')
        lines += ['', 'Command jump when switching chunks (policy vs the recording\'s normal frame-to-frame change):',
                  '', '| | policy mean / max | recording mean / max |', '|---|---|---|',
                  f"| hand position [cm] | {agg('pred_pos_cm', 'mean'):.2f} / {agg('pred_pos_cm', 'max'):.2f} "
                  f"| {agg('recorded_pos_cm', 'mean'):.2f} / {agg('recorded_pos_cm', 'max'):.2f} |",
                  f"| fingertip spread [mm] | {agg('pred_spread_mm', 'mean'):.1f} / {agg('pred_spread_mm', 'max'):.1f} "
                  f"| {agg('recorded_spread_mm', 'mean'):.1f} / {agg('recorded_spread_mm', 'max'):.1f} |"]
    return lines + ['']


def schedule_summary(s, interval, f):
    """Markdown comparing simulated rollout chunk schedules for one checkpoint."""
    res = s['results']
    lines = [f"### Rollout schedules (inference = {s['latency_steps']} control steps, switch at chunk step "
             f"{interval}, crossfade {s['crossfade_steps']} steps)", '',
             'Each episode replayed through the rollout\'s ChunkScheduler. Blocking = the control loop '
             'waits for inference and the robot holds its last command meanwhile; async = inference runs '
             'in the background. Step = largest command change between consecutive control steps.', '']
    sides = sorted({side for r in res.values() for side in r['steps']})
    head = '| schedule | hand pos mean / p95 [cm] | fingertips mean [mm] |'
    sep = '|---|---|---|'
    for side in sides:
        head += f' {side} grasp+release found, extra | timing median / max [s] | hand step p99 / max [cm] ' \
                f'| finger step p99 / max [mm] |'
        sep += '---|---|---|---|'
    head += ' held [%] |'
    sep += '---|'
    lines += [head, sep]
    rec = None
    for name, r in res.items():
        side0 = sides[0] if sides else 'left'
        pos = r['tracking'].get(f'{side0} hand pos [cm]', {})
        tip = r['tracking'].get(f'{side0} fingertips [mm]', {})
        row = f"| {name} | {f(pos.get('mean'))} / {f(pos.get('p95'))} | {f(tip.get('mean'))} |"
        for side in sides:
            g = r['grasp'].get(side)
            st = r['steps'].get(side, {})
            row += (f" {'-' if g is None else str(g['found']) + '/' + str(g['recorded']) + ', ' + str(g['extra'])} "
                    f"| {'-' if g is None else f(g['timing_median_s'], '{:.2f}') + ' / ' + f(g['timing_max_s'], '{:.2f}')} "
                    f"| {f(st.get('policy_pos_cm', {}).get('p99'), '{:.2f}')} / {f(st.get('policy_pos_cm', {}).get('max'), '{:.2f}')} "
                    f"| {f(st.get('policy_spread_mm', {}).get('p99'))} / {f(st.get('policy_spread_mm', {}).get('max'))} |")
            rec = rec or st
        row += f" {r['held_pct']:.1f} |"
        lines.append(row)
    if rec:
        row = '| recording (operator) | | |'
        for side in sides:
            st = next(r['steps'][side] for r in res.values() if side in r['steps'])
            row += (f" | | {f(st['recording_pos_cm']['p99'], '{:.2f}')} / {f(st['recording_pos_cm']['max'], '{:.2f}')} "
                    f"| {f(st.get('recording_spread_mm', {}).get('p99'))} / {f(st.get('recording_spread_mm', {}).get('max'))} |")
        lines.append(row + ' |')
    if any('walk' in r for r in res.values()):
        lines += ['', '| schedule | walk start+stop found, extra | timing median / max [s] | vx err mean [m/s] '
                  '| wz err mean [rad/s] |', '|---|---|---|---|---|']
        for name, r in res.items():
            w = r.get('walk')
            if w:
                lines.append(f"| {name} | {w['found']}/{w['recorded']}, {w['extra']} "
                             f"| {f(w['timing_median_s'], '{:.2f}')} / {f(w['timing_max_s'], '{:.2f}')} "
                             f"| {f(r['tracking'].get('vx [m/s]', {}).get('mean'), '{:.3f}')} "
                             f"| {f(r['tracking'].get('wz [rad/s]', {}).get('mean'), '{:.3f}')} |")
    over = {n: r['overrun_steps'] for n, r in res.items() if r['overrun_steps']}
    if over:
        lines += ['', 'Chunk ran out before the next one arrived (last command held): ' +
                  ', '.join(f'{n} {v} steps' for n, v in over.items())]
    return lines + ['']


def write_summary(results, path, interval, n_frames):
    def f(x, fmt='{:.1f}'):
        return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)

    rows = [
        ('left hand pos [cm]', 'left_eef_to_body_pose', 'pos_cm'),
        ('left hand rot [deg]', 'left_eef_to_body_pose', 'rot_deg'),
        ('left fingertips [mm]', 'left_eef_link_pose', 'tip_mm'),
        ('right hand pos [cm]', 'right_eef_to_body_pose', 'pos_cm'),
        ('right fingertips [mm]', 'right_eef_link_pose', 'tip_mm'),
        ('torso height [cm]', 'body_pose', 'height_cm'),
        ('torso rot [deg]', 'body_pose', 'rot_deg'),
        ('vx [m/s]', 'velocity_commands', 'vx_abs'),
        ('wz [rad/s]', 'velocity_commands', 'wz_abs'),
    ]
    lines = [f'# Replay evaluation ({n_frames} recorded frames, executed window = {interval} steps)', '',
             'Errors between the predicted action chunk and the recorded commands. '
             '`k0` = first step of the chunk, `window` = mean over the steps rollout executes '
             'before re-inferring. "hold" = keep sending the current command for the whole '
             'chunk (a policy that freezes; zero error at k0 by construction).', '']
    for r in results:
        lines += [f"## {os.path.basename(r['checkpoint'])}", '',
                  '| quantity | k0 | window | hold (window) |', '|---|---|---|---|']
        for label, head, metric in rows:
            h = r['heads'].get(head, {}).get(metric)
            if h is None:
                continue
            hb = r['hold_baseline'].get(head, {}).get(metric, {})
            lines.append(f"| {label} | {f(h['k0'], '{:.3f}' if 'vx' in metric or 'wz' in metric else '{:.1f}')} "
                         f"| {f(h['window'], '{:.3f}' if 'vx' in metric or 'wz' in metric else '{:.1f}')} "
                         f"| {f(hb.get('window'), '{:.3f}' if 'vx' in metric or 'wz' in metric else '{:.1f}')} |")
        lines += ['', 'Normalised error (window, in training std units): ' +
                  ', '.join(f'{k} {v:.2f}' for k, v in r['normalised_window'].items()), '']
        if 'walking' in r:
            lines += ['| velocity axis | recorded active | predicted active | precision | recall |',
                      '|---|---|---|---|---|']
            for ax, w in r['walking'].items():
                lines.append(f"| {ax} | {100 * w['gt_active_frac']:.1f}% | {100 * w['pred_active_frac']:.1f}% "
                             f"| {f(w['precision'], '{:.2f}')} | {f(w['recall'], '{:.2f}')} |")
            lines.append('')
        if 'execution' in r:
            lines += execution_summary(r['execution'], f)
        if 'schedules' in r:
            lines += schedule_summary(r['schedules'], interval, f)
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_path', required=True, help='training run directory')
    parser.add_argument('--checkpoints', nargs='+', default=['policy_last.ckpt'],
                        help='checkpoint files in --model_path (or paths) to compare')
    parser.add_argument('--embodiment_config_path', default=None,
                        help='default: the YAML recorded in all_configs.json')
    parser.add_argument('--trunk_config_path', default=None,
                        help='default: the YAML recorded in all_configs.json')
    parser.add_argument('--episodes', nargs='*', default=None,
                        help='HDF5 episodes to replay (default: the run\'s validation split)')
    parser.add_argument('--split', default='validation', choices=['validation', 'train'],
                        help='which recorded split to use when --episodes is not given')
    parser.add_argument('--stride', type=int, default=5, help='evaluate every N-th frame')
    parser.add_argument('--interval', type=int, default=45,
                        help='steps executed per chunk, as rollout --inference_interval')
    parser.add_argument('--fps', type=float, default=30.0, help='recording rate, for plot time axes')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default=None, help='default: <model_path>/replay_eval')
    parser.add_argument('--no_plots', action='store_true')
    parser.add_argument('--schedules', default='upstream,crossfade,async,async+crossfade',
                        help='rollout chunk schedules to simulate over whole episodes (comma list, or none)')
    parser.add_argument('--crossfade_steps', type=int, default=15, help='as rollout --crossfade_steps')
    parser.add_argument('--save_commands', action='store_true',
                        help='also save each schedule\'s per-step command stream (<ckpt>/<episode>_commands_<schedule>.npz)')
    parser.add_argument('--latency_ms', type=float, default=None,
                        help='inference time to simulate (default: measured on this machine)')
    args = parser.parse_args()

    # Same contract check rollout performs: the YAMLs must be the ones trained with.
    try:
        args.embodiment_config_path, args.trunk_config_path = resolve_run_configs(
            args.model_path, args.embodiment_config_path, args.trunk_config_path)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.episodes:
        episodes = [(p, episode_len(p)) for p in args.episodes]
    else:
        episodes = load_split_episodes(args.model_path, args.split)
    out_dir = args.out or os.path.join(args.model_path, 'replay_eval')
    os.makedirs(out_dir, exist_ok=True)
    print(f'{len(episodes)} episodes, {sum(n for _, n in episodes)} frames -> {out_dir}')

    results = []
    for ckpt in args.checkpoints:
        print(f'== {ckpt}')
        results.append(evaluate_checkpoint(args, ckpt, episodes, out_dir))
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'episodes': [p for p, _ in episodes], 'interval': args.interval,
                   'stride': args.stride, 'results': results}, fh, indent=2)
    print(write_summary(results, os.path.join(out_dir, 'summary.md'), args.interval,
                        sum(n for _, n in episodes)))


if __name__ == '__main__':
    sys.exit(main())
