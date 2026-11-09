"""Tests for the whole-episode metrics in replay_eval.py (no model or data needed).

Run: python test_replay_eval.py  (or pytest), in the htd_policy environment.
"""
import numpy as np

import replay_eval as re_

FPS = 30


def _hand(spread_m, T):
    """(T, 15) fingertips, all at distance spread_m from the hand origin along x."""
    a = np.zeros((T, 5, 3))
    a[..., 0] = spread_m[:, None]
    return a.reshape(T, 15)


def _episode(T=600, shift=0):
    """Grasp at 5 s, release at 12 s, walk forward at 0.3 m/s from 2 s to 4 s (all delayed by `shift`)."""
    t = np.arange(T)
    spread = np.where((t >= 150 + shift) & (t < 360 + shift), 0.05, 0.09)
    vel = np.zeros((T, 3))
    vel[(t >= 60 + shift) & (t < 120 + shift), 0] = 0.3
    pose = np.zeros((T, 6))
    pose[:, 0] = 0.3 + 0.001 * t
    return {'left_eef_link_pose': _hand(spread, T), 'right_eef_link_pose': np.zeros((T, 15)),
            'left_eef_to_body_pose': pose, 'right_eef_to_body_pose': np.zeros((T, 6)),
            'body_pose': np.zeros((T, 6)), 'velocity_commands': vel}


def test_debounce_removes_flicker_keeps_edges():
    on = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    out = re_.debounce(on, 2)
    assert out.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]


def test_match_events_offsets_missed_extra():
    gt = [(100, 'on'), (200, 'off'), (300, 'on')]
    pred = [(110, 'on'), (195, 'off'), (900, 'on'), (950, 'off')]
    offs, missed, extra = re_.match_events(gt, pred, max_gap=90)
    assert offs == [10, -5] and missed == 1 and extra == 2


def test_perfect_policy_scores_zero():
    gt = _episode()
    x, per_frame = re_.episode_execution(gt, gt, gt, 45, FPS)
    assert all(v['max'] == 0 for v in x['errors'].values())
    g = x['grasp']['left']
    assert [k for _, k in g['recorded']] == ['grasp', 'release'] and g['offsets_s'] == [0, 0]
    assert 'right' not in x['grasp'] and not x['active']['right_eef_to_body_pose']
    w = x['walk']
    assert w['offsets_s'] == [0, 0] and abs(w['recorded_dist_m'] - 0.6) < 1e-9 and w['end_pos_gap_m'] == 0


def test_late_policy_timing_measured():
    gt, late = _episode(), _episode(shift=15)  # policy acts 0.5 s late
    x, _ = re_.episode_execution(gt, late, gt, 45, FPS)
    assert np.allclose(x['grasp']['left']['offsets_s'], [0.5, 0.5])
    assert np.allclose(x['walk']['offsets_s'], [0.5, 0.5])
    assert x['walk']['end_pos_gap_m'] < 1e-9  # same distance, just later
    assert x['errors']['left fingertips [mm]']['max'] > 30


def test_chunk_switch_jump():
    gt = _episode()
    pred = {k: v.copy() for k, v in gt.items()}
    pred['left_eef_to_body_pose'][45:90, 0] += 0.05  # a 5 cm step at the 45 and 90 switches
    x, _ = re_.episode_execution(gt, pred, gt, 45, FPS)
    assert abs(x['chunk_switch']['left']['pred_pos_cm']['max'] - 5.1) < 1e-6
    assert 'right' not in x['chunk_switch']


def _oracle(gt, T, H=60):
    """A perfect policy: the chunk observed at t is exactly the recording from t on."""
    def predict_at(t):
        idx = np.minimum(np.arange(t, t + H), T - 1)
        return {k: v[idx] for k, v in gt.items()}
    return predict_at


def test_async_schedule_is_on_time_blocking_lags():
    T, L = 300, 10
    gt = _episode(T)
    keys, hand = list(gt), 'left_eef_to_body_pose'
    cmd, held, overrun = re_.simulate_schedule(_oracle(gt, T), T, 45, 15, True, L, keys)
    assert np.isnan(cmd[hand][:L]).all() and np.allclose(cmd[hand][L:], gt[hand][L:])
    assert not held.any() and overrun == 0
    # Blocking: inferences at wall 0, 55, 110, ... each freeze the loop for L frames,
    # and every chunk is played L frames after the moment it was observed.
    cmd, held, _ = re_.simulate_schedule(_oracle(gt, T), T, 45, 0, False, L, keys)
    assert np.allclose(cmd[hand][L], gt[hand][0]) and np.allclose(cmd[hand][65], gt[hand][55])
    assert held.sum() == 5 * L and held[55:65].all()
    m = re_.schedule_episode(gt, cmd, held, 45, FPS)
    assert m['uncommanded'] == L and m['held'] == 5 * L


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'{len(tests)} tests passed')
