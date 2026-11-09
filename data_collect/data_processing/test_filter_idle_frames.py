import hashlib
import json
import os

import h5py
import numpy as np
import pytest

import filter_idle_frames as fif

FPS = 30
N = 300
# Source frames the default thresholds must drop: the 2 s plateau at 80..139 keeps its
# first 0.3 s (80..88).
PLATEAU_DROPPED = np.arange(89, 140)
GROUP = 'g1_inspire_wb/task__manipulation'


def _trajectory():
    """Wrist, fingertip, torso and base commands for the synthetic episode.

    Each stretch with the left arm still has exactly one other command moving and
    is longer than keep_s, so dropping any idle term drops frames from it.
    """
    rng = np.random.default_rng(0)
    step = 0.1 / FPS  # 0.1 m/s while moving
    wrist_v = np.zeros(N)
    right_v = np.zeros(N)
    finger_v = np.zeros(N)
    torso_v = np.zeros(N)
    vel = np.zeros((N, 3))
    wrist_v[45:80] = step          # 0..44 idle at the start (inside the 2 s edge)
    # 80..139 plateau: everything still apart from sub-threshold jitter
    wrist_v[140:150] = step
    finger_v[150:165] = 0.05 / FPS  # fingers closing at 5 cm/s
    right_v[165:180] = step         # right wrist moving
    wrist_v[180:190] = step
    vel[190:202, 0] = 0.3           # walking forward
    vel[202:214, 2] = 0.5           # turning
    vel[214:226, 1] = 0.3           # walking sideways
    torso_v[226:238] = 0.05 / FPS   # torso moving
    wrist_v[238:] = step
    left = np.zeros((N, 6))
    left[:, 0] = 0.3 + np.cumsum(wrist_v)
    left[80:140, 0:3] += rng.uniform(-1e-5, 1e-5, size=(60, 3))
    right = np.tile([0.3, -0.2, 0.1, 0, 0, 0], (N, 1))
    right[:, 0] += np.cumsum(right_v)
    body = np.tile([0, 0, 0.75, 0, 0, 0], (N, 1))
    body[:, 2] += np.cumsum(torso_v)
    tips = np.tile(np.linspace(0.0, 0.1, 15), (N, 1)) + np.cumsum(finger_v)[:, None]
    return left, right, body, tips, vel


def _write_episode(path, n=N, misaligned=None):
    left, right, body, tips, vel = (a[:n] for a in _trajectory())
    ts = np.arange(n) / FPS
    with h5py.File(path, 'w') as f:
        f.attrs['embodiment'] = 'g1_inspire_wb'
        f.attrs['frequency'] = FPS
        f.attrs['compress'] = True
        f.attrs['sim'] = False
        f.attrs['hand_type'] = 'dftp'
        f.create_dataset('actions/left_eef_to_body', data=left)
        f.create_dataset('actions/right_eef_to_body', data=right)
        f.create_dataset('actions/left_eef_link', data=tips)
        f.create_dataset('actions/right_eef_link', data=np.tile(np.linspace(0, 0.1, 15), (n, 1)))
        f.create_dataset('actions/body', data=body)
        f.create_dataset('actions/velocity_commands', data=vel)
        f.create_dataset('actions/eef_to_body', data=np.hstack([left, left]).astype(np.float32),
                         chunks=(64, 12), maxshape=(None, 12), compression='gzip', compression_opts=4)
        f['actions/eef_to_body'].attrs['units'] = 'm, rad'
        f.create_dataset('observations/proprioceptions/left_eef_to_body_proprio', data=left + 0.001)
        other = f.create_group('observations/proprioceptions/other')
        other.attrs['note'] = 'nested'
        other.create_dataset('robot_joint_proprio', data=np.arange(n * 58, dtype=np.float32).reshape(n, 58))
        f.create_dataset('observations/tactile/left_palm_tactile',
                         data=np.linspace(0, 1, n * 112, dtype=np.float32).reshape(n, 112))
        images = f.create_dataset('observations/images/main', (n,), dtype=h5py.vlen_dtype(np.uint8),
                                  chunks=(16,), maxshape=(None,))
        for i in range(n):
            images[i] = np.full(10 + i % 7, i % 251, dtype=np.uint8)
        f.create_dataset('observations/head_cam_timestamp', data=ts)
        f.create_dataset('camera_poses/head_camera_to_init', data=np.tile(ts[:, None], (1, 6)))
        f.create_dataset('masks/act_body', data=np.array([False, False, True, True, True, True]))
        f.create_dataset('masks/img_main', data=np.ones(1, dtype=bool))
        if misaligned:
            f.create_dataset(misaligned, data=np.zeros((n - 1, 3)))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temporal_paths(f):
    paths = []
    f.visititems(lambda name, obj: paths.append(name)
                 if isinstance(obj, h5py.Dataset) and name.startswith(fif.TEMPORAL_PREFIXES) else None)
    return sorted(paths)


@pytest.fixture
def task(tmp_path):
    src = tmp_path / 'processed' / 'task'
    (src / GROUP).mkdir(parents=True)
    _write_episode(src / GROUP / 'episode_1.hdf5')
    _write_episode(src / GROUP / 'episode_1_mirror.hdf5')
    _write_episode(src / GROUP / 'episode_1_features.hdf5')
    return src, tmp_path / 'processed' / 'task_filtered'


def test_drops_only_the_tail_of_the_long_pause(task, tmp_path):
    src, dst = task
    report = tmp_path / 'report.json'
    assert fif.main(['--src', str(src), '--dst', str(dst), '--report', str(report)]) == 0
    with h5py.File(dst / GROUP / 'episode_1.hdf5', 'r') as f:
        kept_frames = np.rint(f['observations/head_cam_timestamp'][()] * FPS).astype(int)
        idle = fif.idle_mask(f, FPS, fif.DEFAULT_THRESHOLDS)
    assert len(kept_frames) == N - len(PLATEAU_DROPPED)
    assert np.array_equal(np.setdiff1d(np.arange(N), kept_frames), PLATEAU_DROPPED)
    # the plateau now holds exactly keep_s * fps idle frames
    plateau = idle[80:80 + 12]
    assert plateau[:9].all() and not plateau[9:].any()
    data = json.loads(report.read_text())
    episode, = data['episodes']
    assert episode['frames'] == N and episode['kept'] == N - 51 and episode['dropped'] == 51
    assert episode['pauses_shortened'] == 1 and episode['dropped_ranges'] == [[89, 140]]
    assert data['thresholds']['keep_s'] == 0.3 and data['thresholds']['fps'] == 30


def test_every_temporal_dataset_is_cut_and_metadata_is_copied(task):
    src, dst = task
    assert fif.main(['--src', str(src), '--dst', str(dst)]) == 0
    keep = np.setdiff1d(np.arange(N), PLATEAU_DROPPED)
    with h5py.File(src / GROUP / 'episode_1.hdf5', 'r') as s, h5py.File(dst / GROUP / 'episode_1.hdf5', 'r') as d:
        assert dict(d.attrs) == dict(s.attrs)
        assert _temporal_paths(d) == _temporal_paths(s) and len(_temporal_paths(d)) == 13
        for path in _temporal_paths(s):
            a, b = s[path], d[path]
            assert b.shape == (len(keep),) + a.shape[1:], path
            assert b.dtype == a.dtype and b.compression == a.compression and dict(b.attrs) == dict(a.attrs), path
            assert (b.chunks is None) == (a.chunks is None), path
            if a.dtype.kind == 'O':
                assert h5py.check_vlen_dtype(b.dtype) == np.dtype(np.uint8)
                assert all(np.array_equal(b[j], a[i]) for j, i in enumerate(keep))
            else:
                assert np.array_equal(b[()], a[()][keep]), path
        assert d['actions/eef_to_body'].chunks == (64, 12) and d['actions/eef_to_body'].compression_opts == 4
        assert dict(d['observations/proprioceptions/other'].attrs) == {'note': 'nested'}
        for path in ('masks/act_body', 'masks/img_main'):
            assert np.array_equal(d[path][()], s[path][()]) and d[path].dtype == s[path].dtype
    assert sorted(p.name for p in (dst / GROUP).iterdir()) == ['episode_1.hdf5']


def test_source_is_byte_identical_afterwards(task):
    src, dst = task
    files = sorted((src / GROUP).iterdir())
    before = {p: (_sha256(p), p.stat().st_mtime_ns) for p in files}
    assert fif.main(['--src', str(src), '--dst', str(dst)]) == 0
    assert sorted((src / GROUP).iterdir()) == files
    assert {p: (_sha256(p), p.stat().st_mtime_ns) for p in files} == before


def test_dry_run_writes_nothing(task, capsys):
    src, dst = task
    assert fif.main(['--src', str(src), '--dst', str(dst), '--dry_run']) == 0
    assert not dst.exists()
    assert 'kept=249/300 dropped=51 pauses_shortened=1' in capsys.readouterr().out


def test_short_episode_inside_the_edges_is_kept_whole(tmp_path):
    src = tmp_path / 'src'
    (src / GROUP).mkdir(parents=True)
    _write_episode(src / GROUP / 'episode_2.hdf5', n=110)  # 80..109 plateau ends inside the last 2 s
    assert fif.main(['--src', str(src), '--dst', str(tmp_path / 'dst')]) == 0
    with h5py.File(tmp_path / 'dst' / GROUP / 'episode_2.hdf5', 'r') as f:
        assert f['actions/body'].shape[0] == 110


def test_misaligned_episode_is_an_error_and_leaves_no_output(tmp_path):
    src = tmp_path / 'src'
    (src / GROUP).mkdir(parents=True)
    _write_episode(src / GROUP / 'episode_3.hdf5', misaligned='observations/proprioceptions/other/imu_proprio')
    dst = tmp_path / 'dst'
    assert fif.main(['--src', str(src), '--dst', str(dst)]) == 1
    assert not any(p.is_file() for p in dst.rglob('*'))


def test_vlen_rows_of_equal_length_and_one_row_blocks_are_copied(tmp_path):
    # Real failure: episode_82 had a 257-frame kept segment, so its last block was one
    # JPEG; h5py turns such a slice into a 2-D array and refuses to write it as vlen rows.
    n = fif.COPY_BLOCK + 3
    with h5py.File(tmp_path / 'vlen.h5', 'w') as f:
        src = f.create_dataset('src', (n,), dtype=h5py.vlen_dtype(np.uint8))
        for i in range(n):
            src[i] = np.full(7, i % 256, dtype=np.uint8)
        dst = f.create_dataset('dst', (n - 1,), dtype=h5py.vlen_dtype(np.uint8))
        fif._copy_rows(src, dst, [(0, fif.COPY_BLOCK + 1), (fif.COPY_BLOCK + 2, n)])
        rows = [dst[i] for i in range(n - 1)]
    assert [len(r) for r in rows] == [7] * (n - 1)
    assert [int(r[0]) for r in rows] == [i % 256 for i in range(fif.COPY_BLOCK + 1)] + [(n - 1) % 256]


def _files_under(path):
    return [os.path.join(root, name) for root, _, names in os.walk(path) for name in names]


def test_failed_write_leaves_no_file_or_temporary(task, monkeypatch):
    src, dst = task

    def fail(*args):
        raise OSError('disk full')

    monkeypatch.setattr(fif, '_copy_rows', fail)
    assert fif.main(['--src', str(src), '--dst', str(dst)]) == 1
    assert _files_under(dst) == []


@pytest.mark.parametrize('defect', ['missing_dataset', 'short_dataset'])
def test_verify_rejects_an_incomplete_episode(task, monkeypatch, defect):
    src, dst = task
    write = fif._write_filtered

    def write_with_defect(src_file, dst_file, keep):
        write(src_file, dst_file, keep)
        if defect == 'missing_dataset':
            del dst_file['camera_poses/head_camera_to_init']
        else:
            dst_file['actions/eef_to_body'].resize(10, axis=0)

    monkeypatch.setattr(fif, '_write_filtered', write_with_defect)
    assert fif.main(['--src', str(src), '--dst', str(dst)]) == 1
    assert _files_under(dst) == []


@pytest.mark.parametrize('where', ['inside_src', 'nonempty'])
def test_refuses_unsafe_destinations(task, where):
    src, dst = task
    if where == 'inside_src':
        dst = src / 'filtered'
    else:
        dst.mkdir()
        (dst / 'old.txt').write_text('x')
    with pytest.raises(SystemExit):
        fif.main(['--src', str(src), '--dst', str(dst)])
    assert os.listdir(src) == ['g1_inspire_wb']
