import importlib.util
import json
import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mxt_utils import apply_std_floor

EMB = 'g1_inspire_wb'


def _stats():
    def entry(*std):
        return {'mean': np.full(len(std), 0.5, dtype=np.float32), 'std': np.array(std, dtype=np.float32)}
    return {EMB: {
        'obs': {
            'main_image': {'mean': None, 'std': None},
            'right_eef_to_body_pose_proprio': entry(0.001, 0.2, 0.0),
            'robot_joint_vel_proprio': entry(0.5, 0.004),
            'left_palm_tactile': entry(0.0, 0.003),
            'left_tactile_cap': entry(0.002),
        },
        'action': {
            'right_eef_to_body_pose': entry(0.0012, 0.3),
            'velocity_commands': entry(0.0, 0.1, 0.002),
        },
        'dream': {'left_eef_finger_force': entry(0.0, 0.001)},
    }}


def test_floors_proprio_and_action_std_only():
    stats = _stats()
    before = pickle.dumps(stats)
    floored = apply_std_floor(stats, 0.01)
    assert pickle.dumps(stats) == before  # input untouched
    obs, action = floored[EMB]['obs'], floored[EMB]['action']
    np.testing.assert_array_equal(obs['right_eef_to_body_pose_proprio']['std'], np.float32([0.01, 0.2, 0.01]))
    np.testing.assert_array_equal(obs['robot_joint_vel_proprio']['std'], np.float32([0.5, 0.01]))
    np.testing.assert_array_equal(action['right_eef_to_body_pose']['std'], np.float32([0.01, 0.3]))
    np.testing.assert_array_equal(action['velocity_commands']['std'], np.float32([0.01, 0.1, 0.01]))
    for group in (obs, action):
        for value in group.values():
            if value['std'] is not None:
                assert value['std'].dtype == np.float32
    original = stats[EMB]
    assert obs['main_image'] == {'mean': None, 'std': None}
    for key in ('left_palm_tactile', 'left_tactile_cap'):
        np.testing.assert_array_equal(obs[key]['std'], original['obs'][key]['std'])
    np.testing.assert_array_equal(floored[EMB]['dream']['left_eef_finger_force']['std'],
                                  original['dream']['left_eef_finger_force']['std'])
    for group in ('obs', 'action'):
        for key, value in original[group].items():
            if value['mean'] is not None:
                np.testing.assert_array_equal(floored[EMB][group][key]['mean'], value['mean'])


def test_zero_floor_is_the_unchanged_object():
    stats = _stats()
    assert apply_std_floor(stats, 0.0) is stats


@pytest.mark.parametrize('floor', [-0.01, float('nan'), float('inf')])
def test_rejects_invalid_floor(floor):
    with pytest.raises(ValueError):
        apply_std_floor(_stats(), floor)


@pytest.fixture
def train_mxt(monkeypatch):
    if importlib.util.find_spec('wandb') is None:  # train_mxt imports it at module level
        monkeypatch.setitem(sys.modules, 'wandb', types.ModuleType('wandb'))
    import train_mxt
    return train_mxt


@pytest.mark.parametrize('floor', [0.0, 0.01])
def test_floored_stats_reach_the_policy_and_dataset_stats(train_mxt, monkeypatch, tmp_path, floor):
    """main_train with the loader and the training loop stubbed (no GPU, no dataset)."""
    raw = _stats()
    raw_pickle = pickle.dumps(raw)
    loader = SimpleNamespace(dataset=SimpleNamespace(
        dataset_path_list=np.array([b'/data/episode_1.hdf5']), episode_ids=[0], episode_len=[10]))
    monkeypatch.setattr(train_mxt, 'load_data', lambda *a, **k: (loader, loader, raw, [EMB], False))
    seen = {}
    monkeypatch.setattr(train_mxt, 'train_bc', lambda train, val, config, policy_config:
                        seen.setdefault('policy_config', policy_config) and None)
    argv = ['--touch_dreaming', '--dataset_dir', str(tmp_path), '--ckpt_dir', str(tmp_path / 'run'),
            '--task_name', 't', '--norm_std_floor', str(floor)]
    parser = train_mxt.build_parser(argv=argv)
    args = vars(parser.parse_args(argv))
    train_mxt.main_train(args, experiment_suffix='HTD', model_configs=train_mxt._prepare_model_configs(args))
    run = tmp_path / 'run_t_HTD'
    with open(run / 'dataset_stats.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    policy_stats = seen['policy_config']['norm_stats']
    if floor == 0.0:
        assert (run / 'dataset_stats.pkl').read_bytes() == raw_pickle
        assert policy_stats is raw
    else:
        expected = apply_std_floor(raw, floor)
        assert pickle.dumps(saved) == pickle.dumps(expected) == pickle.dumps(policy_stats)
        assert saved[EMB]['action']['velocity_commands']['std'].min() == np.float32(0.01)
    assert json.loads((run / 'all_configs.json').read_text())['norm_std_floor'] == floor


def test_parser_rejects_a_negative_floor(train_mxt, tmp_path, capsys):
    argv = ['--touch_dreaming', '--dataset_dir', str(tmp_path), '--ckpt_dir', 'x', '--task_name', 't',
            '--norm_std_floor', '-0.1']
    parser = train_mxt.build_parser(argv=argv)
    with pytest.raises(SystemExit):
        train_mxt.main(argv, parser=parser)
    assert '--norm_std_floor must be finite and nonnegative' in capsys.readouterr().err


def test_resume_requires_the_original_floor(train_mxt, tmp_path):
    (tmp_path / 'all_configs.json').write_text(json.dumps({'policy_class': 'MXT', 'policy_config': {}}))
    argv = ['--touch_dreaming', '--dataset_dir', str(tmp_path), '--ckpt_dir', 'x', '--task_name', 't',
            '--resume_training', str(tmp_path), '--norm_std_floor', '0.01']
    args = vars(train_mxt.build_parser(argv=argv).parse_args(argv))
    with pytest.raises(ValueError, match='norm_std_floor'):
        train_mxt._prepare_model_configs(args)
