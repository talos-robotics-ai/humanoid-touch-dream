"""Tests for chunk_schedule.py. Run: python test_chunk_schedule.py  (or pytest). Needs only numpy."""
import numpy as np

import chunk_schedule as cs

H = 60


def _chunk(value, t_obs=0, dims=3):
    """A chunk whose step i commands value + (t_obs + i) * 1e-3 in every dimension."""
    ramp = (t_obs + np.arange(H))[:, None] * 1e-3
    return {'x': np.full((H, dims), float(value)) + ramp}


def _rot(rpy):
    x, y, z = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def test_quaternion_roundtrip_matches_zyx_convention():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rpy = rng.uniform([-3, -1.4, -3], [3, 1.4, 3])
        assert np.allclose(cs.quat_to_rpy(cs.rpy_to_quat(rpy)), rpy, atol=1e-9)
        w, x, y, z = cs.rpy_to_quat(rpy)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(R, _rot(rpy), atol=1e-9)


def test_pose_blend_slerps_rotation_and_lerps_position():
    a = np.array([0.0, 0, 0, 0, 0, 0])
    b = np.array([0.2, 0, 0, 0, 0, np.pi / 2])
    mid = cs.blend('pose6', a, b, 0.5)
    assert np.allclose(mid, [0.1, 0, 0, 0, 0, np.pi / 4])
    # across the +-pi wrap the blend takes the short way round
    a[5], b[5] = np.radians(170), np.radians(-170)
    assert np.isclose(abs(cs.blend('pose6', a, b, 0.5)[5]), np.pi)


def test_blocking_without_crossfade_is_upstream_schedule():
    s = cs.ChunkScheduler(45, crossfade_steps=0, async_mode=False)
    sent = []
    for t in range(200):
        if s.should_request(t):
            s.submit(_chunk(100 * (t // 45), t), t_obs=t, t_now=t, generation=s.mark_requested(t))
        sent.append(s.command(t)['x'][0])
    expected = [100 * (t // 45) + t * 1e-3 for t in range(200)]
    assert np.allclose(sent, expected)


def test_crossfade_removes_the_jump():
    s = cs.ChunkScheduler(45, crossfade_steps=15)
    sent = []
    for t in range(120):
        if s.should_request(t):
            s.submit(_chunk(0 if t == 0 else 1, t), t_obs=t, t_now=t, generation=s.mark_requested(t))
        sent.append(s.command(t)['x'][0])
    steps = np.abs(np.diff(sent))
    assert steps.max() < 1 / 16 + 2e-3          # a 1.0 jump spread over 16 steps
    assert np.isclose(sent[44], 44e-3) and np.isclose(sent[60], 1 + 60e-3)  # before / after the fade


def test_crossfade_limited_by_remaining_chunk():
    s = cs.ChunkScheduler(55, crossfade_steps=15)
    s.submit(_chunk(0), 0, 0)
    assert s.crossfade == 5
    s = cs.ChunkScheduler(60, crossfade_steps=15)
    s.submit(_chunk(0), 0, 0)
    assert s.crossfade == 0


def test_async_requests_early_and_enters_chunk_at_current_time():
    s = cs.ChunkScheduler(45, crossfade_steps=0, async_mode=True)
    s.submit(_chunk(0), t_obs=0, t_now=10, generation=s.mark_requested(0), latency_steps=9.4)
    assert s.lead_steps == 11 and s.next_request == 34    # ceil(9.4) + 1 margin
    assert np.isclose(s.command(10)['x'][0], 10e-3)       # chunk index 10, not 0
    assert not s.should_request(33) and s.should_request(34)
    gen = s.mark_requested(34)
    s.submit(_chunk(5, 34), t_obs=34, t_now=44, generation=gen, latency_steps=9.4)
    assert np.isclose(s.command(44)['x'][0], 5 + 44e-3)


def test_reset_discards_inference_in_flight():
    s = cs.ChunkScheduler(45)
    gen = s.mark_requested(0)
    s.reset()
    assert not s.submit(_chunk(0), 0, 0, generation=gen)
    assert s.command(0) is None and s.should_request(0)


def test_late_inference_holds_last_step_and_counts_overrun():
    s = cs.ChunkScheduler(45)
    s.submit(_chunk(0), 0, 0)
    assert np.isclose(s.command(70)['x'][0], 59e-3) and s.overrun_steps == 1


def test_failed_inference_can_be_retried():
    s = cs.ChunkScheduler(45, async_mode=True)
    gen = s.mark_requested(0)
    assert not s.should_request(1)
    s.cancel_request(gen)
    assert s.should_request(1)


def test_walk_command_is_not_crossfaded():
    """The walk command switches to the new chunk's prediction at once; poses still fade."""
    s = cs.ChunkScheduler(45, crossfade_steps=15, kinds=cs.G1_COMMAND_KINDS)
    vel, pose = [], []
    for t in range(120):
        if s.should_request(t):
            v = 0.0 if t == 0 else 0.3
            chunk = {'velocity_commands': np.tile([v, 0.0, -v], (60, 1)),
                     'body_pose': np.tile([v, 0, 0, 0, 0, 0], (60, 1))}
            s.submit(chunk, t_obs=t, t_now=t, generation=s.mark_requested(t))
        cmd = s.command(t)
        vel.append(cmd['velocity_commands'].copy())
        pose.append(cmd['body_pose'][0])
    vel = np.array(vel)
    assert np.all(vel[:45] == [0.0, 0.0, 0.0]) and np.all(vel[45:] == [0.3, 0.0, -0.3])
    assert 0.0 < pose[50] < 0.3                 # the pose is mid-fade where the walk already switched


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'{len(tests)} tests passed')
