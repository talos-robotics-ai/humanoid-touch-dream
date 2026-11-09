"""Write a copy of a processed HTD task directory with long idle pauses shortened.

A frame is idle when every command is still: both wrists, every fingertip, the
torso and the base. A run of idle frames longer than --keep_s keeps its first
--keep_s and drops the rest; the first and last --edge_s of an episode are never
dropped. The source is only read; each output episode keeps the source's paths,
dtypes, storage settings and attributes, with the dropped rows removed from every
temporal dataset.
"""

import argparse
import fnmatch
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

if __package__:
    from .processing_utils import atomic_write_hdf5
else:
    from processing_utils import atomic_write_hdf5

# Every dataset below these groups is per-frame (validate_htd_dataset.py uses the same rule).
TEMPORAL_GROUPS = ('actions', 'camera_poses', 'observations')
TEMPORAL_PREFIXES = tuple(f'{name}/' for name in TEMPORAL_GROUPS)
DEFAULT_THRESHOLDS = {
    'keep_s': 0.3, 'edge_s': 2.0,
    'arm_v': 0.01, 'finger_v': 0.02, 'torso_v': 0.01, 'vx': 0.05, 'wz': 0.1,
}
COPY_BLOCK = 256


def find_all_hdf5(dataset_dir):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir, followlinks=False):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename or 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return sorted(hdf5_files)


def _point_speeds(positions, fps):
    """[N, 3k] positions -> [N, k] speeds by backward differences; frame 0 copies frame 1."""
    points = np.asarray(positions, dtype=np.float64).reshape(len(positions), -1, 3)
    if len(points) < 2:
        return np.zeros(points.shape[:2])
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=-1) * fps
    return np.concatenate([speeds[:1], speeds])


def idle_mask(f, fps, thresholds):
    """Frames where every command moves slower than its threshold (NaN counts as moving)."""
    left = _point_speeds(f['actions/left_eef_to_body'][:, 0:3], fps)[:, 0]
    right = _point_speeds(f['actions/right_eef_to_body'][:, 0:3], fps)[:, 0]
    tips = np.concatenate([_point_speeds(f['actions/left_eef_link'][()], fps),
                           _point_speeds(f['actions/right_eef_link'][()], fps)], axis=1)
    torso = _point_speeds(f['actions/body'][:, 0:3], fps)[:, 0]
    velocity = np.abs(np.asarray(f['actions/velocity_commands'][()], dtype=np.float64))
    with np.errstate(invalid='ignore'):
        return ((left < thresholds['arm_v']) & (right < thresholds['arm_v'])
                & (tips.max(axis=1) < thresholds['finger_v'])
                & (torso < thresholds['torso_v'])
                # lateral velocity is held to the forward threshold
                & (velocity[:, 0] < thresholds['vx']) & (velocity[:, 1] < thresholds['vx'])
                & (velocity[:, 2] < thresholds['wz']))


def _runs(mask):
    """[(start, stop)] of the True runs in a boolean vector."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2])]


def drop_mask(idle, keep_frames, edge_frames):
    """Frames to drop and the shortened idle runs as source [start, stop)."""
    idle = idle.copy()
    if edge_frames:
        idle[:edge_frames] = False
        idle[-edge_frames:] = False
    drop = np.zeros(len(idle), dtype=bool)
    shortened = []
    for start, stop in _runs(idle):
        if stop - start > keep_frames:
            drop[start + keep_frames:stop] = True
            shortened.append([start, stop])
    return drop, shortened


def _hard_children(group):
    for name in group:
        link = group.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError(f'Unsupported {type(link).__name__} at {group.name}/{name}')
        yield name, group[name]


def _temporal_datasets(f):
    """{path: dataset} for every dataset under the temporal groups."""
    found = {}

    def visit(group, prefix):
        for name, obj in _hard_children(group):
            if isinstance(obj, h5py.Group):
                visit(obj, f'{prefix}{name}/')
            elif obj.external or obj.is_virtual:
                raise ValueError(f'Externally backed dataset is unsupported: {prefix}{name}')
            else:
                found[f'{prefix}{name}'] = obj

    for name, obj in _hard_children(f):
        if name in TEMPORAL_GROUPS:
            if not isinstance(obj, h5py.Group):
                raise ValueError(f'/{name} must be a group')
            visit(obj, f'{name}/')
    return found


def _copy_attrs(src, dst):
    for name in src.attrs:
        attr = src.attrs.get_id(name)
        dst.attrs.create(name, src.attrs[name], shape=attr.shape, dtype=attr.dtype)


def _copy_rows(src, dst, segments):
    # A slice of a variable-length dataset whose rows happen to have equal lengths (always
    # the case for a one-row block, e.g. a 257-frame segment) comes back from h5py as a 2-D
    # array that cannot be written into a vlen dataset, so those rows go one at a time.
    vlen = h5py.check_vlen_dtype(src.dtype) is not None
    out = 0
    for start, stop in segments:
        for block in range(start, stop, COPY_BLOCK):
            end = min(block + COPY_BLOCK, stop)
            if vlen:
                for i in range(block, end):
                    dst[out + i - block] = src[i]
            else:
                dst[out:out + end - block] = src[block:end]
            out += end - block


def _write_filtered(src, dst, keep):
    rows = int(keep.sum())
    segments = _runs(keep)
    _copy_attrs(src, dst)

    def copy_group(src_group, dst_group):
        _copy_attrs(src_group, dst_group)
        for name, obj in _hard_children(src_group):
            if isinstance(obj, h5py.Group):
                copy_group(obj, dst_group.create_group(name))
                continue
            chunks = obj.chunks
            maxshape = None
            if chunks is not None:
                first = None if obj.maxshape[0] is None or chunks[0] > rows else rows
                maxshape = (first,) + obj.maxshape[1:]
            # the source's creation property list carries its filters and fill value
            out = dst_group.create_dataset(name, shape=(rows,) + obj.shape[1:], dtype=obj.dtype,
                                           chunks=chunks, maxshape=maxshape,
                                           dcpl=obj.id.get_create_plist())
            _copy_attrs(obj, out)
            _copy_rows(obj, out, segments)

    for name, obj in _hard_children(src):
        if name in TEMPORAL_GROUPS:
            copy_group(obj, dst.create_group(name))
        else:
            src.copy(obj, dst, name=name)


def _verify(dst, src, rows):
    expected = _temporal_datasets(src)
    written = _temporal_datasets(dst)
    if sorted(written) != sorted(expected):
        raise RuntimeError('Temporal dataset paths differ from the source')
    wrong = [path for path, dataset in written.items() if dataset.shape[0] != rows]
    if wrong:
        raise RuntimeError(f'Temporal datasets without {rows} rows: {wrong}')
    for name, obj in _hard_children(src):
        if name not in TEMPORAL_GROUPS and name not in dst:
            raise RuntimeError(f'/{name} was not copied')


def filter_episode(src_path, dst_path, args):
    """Filter one episode; returns its report entry. dst_path None only measures."""
    with h5py.File(src_path, 'r') as src:
        frequency = src.attrs.get('frequency')
        if frequency is not None and not math.isclose(float(frequency), args.fps):
            raise ValueError(f'frequency attribute {frequency} differs from --fps {args.fps}')
        temporal = _temporal_datasets(src)
        frames = src['actions/left_eef_to_body'].shape[0]
        if frames == 0:
            raise ValueError('empty episode')
        unaligned = {path: list(d.shape) for path, d in temporal.items()
                     if not d.shape or d.shape[0] != frames}
        if unaligned:
            raise ValueError(f'temporal datasets without {frames} rows: {unaligned}')
        thresholds = vars(args)
        drop, shortened = drop_mask(idle_mask(src, args.fps, thresholds),
                                    int(round(args.keep_s * args.fps)),
                                    int(round(args.edge_s * args.fps)))
        keep = ~drop
        if dst_path is not None:
            atomic_write_hdf5(dst_path, lambda dst: _write_filtered(src, dst, keep),
                              verify=lambda dst: _verify(dst, src, int(keep.sum())))
    return {
        'frames': int(frames),
        'kept': int(keep.sum()),
        'dropped': int(drop.sum()),
        'pauses_shortened': len(shortened),
        'shortened_pauses': shortened,
        'dropped_ranges': [list(r) for r in _runs(drop)],
    }


def _is_within(path, parent):
    return path == parent or parent in path.parents


def main(argv=None):
    parser = argparse.ArgumentParser(description='Copy a processed HTD task directory with long idle pauses shortened')
    parser.add_argument('--src', required=True, help='processed task directory (read only)')
    parser.add_argument('--dst', required=True, help='new or empty directory for the filtered copy')
    parser.add_argument('--keep_s', type=float, default=DEFAULT_THRESHOLDS['keep_s'],
                        help='seconds of each long idle run that are kept')
    parser.add_argument('--edge_s', type=float, default=DEFAULT_THRESHOLDS['edge_s'],
                        help='seconds at each episode end that are never dropped')
    parser.add_argument('--arm_v', type=float, default=DEFAULT_THRESHOLDS['arm_v'],
                        help='wrist command speed below which an arm is still [m/s]')
    parser.add_argument('--finger_v', type=float, default=DEFAULT_THRESHOLDS['finger_v'],
                        help='fingertip command speed below which a hand is still [m/s]')
    parser.add_argument('--torso_v', type=float, default=DEFAULT_THRESHOLDS['torso_v'],
                        help='torso command speed below which the torso is still [m/s]')
    parser.add_argument('--vx', type=float, default=DEFAULT_THRESHOLDS['vx'],
                        help='|forward| (and |lateral|) base command below which the base is still [m/s]')
    parser.add_argument('--wz', type=float, default=DEFAULT_THRESHOLDS['wz'],
                        help='|yaw rate| base command below which the base is still [rad/s]')
    parser.add_argument('--fps', type=float, default=30.0, help='episode frame rate')
    parser.add_argument('--report', help='JSON file for the per-episode numbers')
    parser.add_argument('--dry_run', action='store_true', help='measure and print only')
    args = parser.parse_args(argv)
    for name in ('keep_s', 'edge_s', 'arm_v', 'finger_v', 'torso_v', 'vx', 'wz', 'fps'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == 'fps' and value == 0):
            parser.error(f'--{name} must be finite and {"positive" if name == "fps" else "nonnegative"}')
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        parser.error('--src must be an existing directory')
    if _is_within(dst, src) or _is_within(src, dst):
        parser.error('--src and --dst must not contain each other')
    if not args.dry_run and dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        parser.error('--dst must be a new or empty directory')

    hdf5_files = find_all_hdf5(src)
    if not hdf5_files:
        print('No eligible HDF5 episodes found')
        return 1
    episodes, errors = [], []
    for index, path in enumerate(hdf5_files, 1):
        relative = os.path.relpath(path, src)
        target = None
        if not args.dry_run:
            target = dst / relative
            target.parent.mkdir(parents=True, exist_ok=True)
        try:
            entry = filter_episode(path, target, args)
        except Exception as exc:
            errors.append({'episode': relative, 'error': f'{type(exc).__name__}: {exc}'})
            print(f'[{index}/{len(hdf5_files)}] Error filtering {relative}: {type(exc).__name__}: {exc}')
            continue
        episodes.append({'episode': relative, **entry})
        print(f'[{index}/{len(hdf5_files)}] {relative}: kept={entry["kept"]}/{entry["frames"]} '
              f'dropped={entry["dropped"]} pauses_shortened={entry["pauses_shortened"]}')

    frames = sum(e['frames'] for e in episodes)
    dropped = sum(e['dropped'] for e in episodes)
    summary = {
        'episodes': len(episodes),
        'episodes_with_drops': sum(1 for e in episodes if e['dropped']),
        'frames': frames,
        'kept': frames - dropped,
        'dropped': dropped,
        'dropped_fraction': dropped / frames if frames else 0.0,
        'pauses_shortened': sum(e['pauses_shortened'] for e in episodes),
        'errors': len(errors),
    }
    print(f'{"Dry run" if args.dry_run else "Filter"} summary: episodes={summary["episodes"]} '
          f'with_drops={summary["episodes_with_drops"]} frames={frames} kept={summary["kept"]} '
          f'dropped={dropped} ({100 * summary["dropped_fraction"]:.2f}%) '
          f'pauses_shortened={summary["pauses_shortened"]} errors={len(errors)}')
    if args.report:
        thresholds = {name: getattr(args, name) for name in (*DEFAULT_THRESHOLDS, 'fps')}
        report = {'src': str(src), 'dst': None if args.dry_run else str(dst), 'dry_run': args.dry_run,
                  'thresholds': thresholds, 'summary': summary, 'episodes': episodes, 'errors': errors}
        with open(args.report, 'w') as stream:
            json.dump(report, stream, indent=2)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
