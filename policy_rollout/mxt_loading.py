"""Load a trained MXT/HTD checkpoint the way rollout runs it.

Shared by ``policy_rollout.py`` and ``replay_eval.py`` so that offline
evaluation exercises the exact deployment path: the saved normalisation
statistics, the embodiment/trunk YAMLs and strict checkpoint deserialisation.
This module needs no robot dependencies (DDS, cameras, keyboard).
"""
import copy
import json
import os
import pickle
import sys

import torch
import yaml

_POLICY_SOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'htd_policy')
if _POLICY_SOURCE_DIR not in sys.path:
    sys.path.insert(0, _POLICY_SOURCE_DIR)

DEFAULT_CHECKPOINT = 'policy_last.ckpt'


def mxt_inference_config(config):
    """Drop image-tokenizer settings that only matter for training.

    * ``gradient_checkpoint``: activation checkpointing.
    * ``weights``: the ImageNet initialisation of the ResNet. The checkpoint's strict load
      overwrites every parameter, and building with ``weights=None`` keeps rollout from
      downloading those weights (it must start on a robot network with no internet).
    """
    config = copy.deepcopy(config)
    if isinstance(config, dict):
        for embodiment in config.values():
            if not isinstance(embodiment, dict):
                continue
            tokenizers = embodiment.get('tokenizer', {})
            if isinstance(tokenizers, dict):
                for name, tokenizer in tokenizers.items():
                    if 'image' in name and isinstance(tokenizer, dict):
                        tokenizer.pop('gradient_checkpoint', None)
                        tokenizer['weights'] = None
    return config


def mxt_camera_size(config, camera):
    """Resolve saved (height, width) exactly as the MXT training loader does."""
    size = []
    for axis, fallback in (('height', 480), ('width', 640)):
        value = config.get(f'{camera}_cam_{axis}')
        if value is None:
            value = config.get(axis, fallback)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'invalid saved {camera} camera {axis}: {value!r}')
        size.append(value)
    return tuple(size)


def checkpoint_path(model_path, checkpoint=None):
    """A checkpoint file name inside model_path, or an explicit path."""
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    if os.path.isabs(checkpoint) or os.path.dirname(checkpoint):
        return checkpoint
    return os.path.join(model_path, checkpoint)


def resolve_run_configs(model_path, embodiment_config_path=None, trunk_config_path=None):
    """The embodiment and trunk YAMLs for a run, checked against the trained artifact.

    A path left as None defaults to the one recorded in the run's all_configs.json
    (relative paths there are relative to htd_policy/). Raises ValueError if either
    YAML differs from what the checkpoint was trained with.
    """
    with open(os.path.join(model_path, 'all_configs.json'), 'r') as f:
        saved = json.load(f)
    paths = {}
    for flag, given in (('embodiment_config_path', embodiment_config_path),
                        ('trunk_config_path', trunk_config_path)):
        path = given or saved[flag]
        paths[flag] = path if os.path.isabs(path) or given else os.path.join(_POLICY_SOURCE_DIR, path)
    with open(paths['embodiment_config_path'], 'r') as f:
        if mxt_inference_config(yaml.safe_load(f)) != mxt_inference_config(saved['policy_config']['embodiment_args_dict']):
            raise ValueError(f"{paths['embodiment_config_path']} does not match the checkpoint artifact")
    with open(paths['trunk_config_path'], 'r') as f:
        if yaml.safe_load(f) != saved['policy_config']['transformer_args']:
            raise ValueError(f"{paths['trunk_config_path']} does not match the checkpoint artifact")
    return paths['embodiment_config_path'], paths['trunk_config_path']


def load_mxt_policy(model_path, embodiment_config_path, trunk_config_path, device, checkpoint=None):
    """Build the MXT policy from a run directory and load its weights.

    Returns ``(policy, info)`` where info holds the saved config, the parsed
    embodiment YAML and the camera sizes recorded at training time.
    """
    from policy import make_policy

    with open(os.path.join(model_path, 'all_configs.json'), 'r') as f:
        saved_config = json.load(f)
    with open(embodiment_config_path, 'r') as f:
        emb_dict = mxt_inference_config(yaml.load(f, Loader=yaml.FullLoader))
    with open(trunk_config_path, 'r') as f:
        transformer_dict = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(model_path, 'dataset_stats.pkl'), 'rb') as f:
        stats_dict = pickle.load(f)

    policy = make_policy('MXT', {
        'norm_stats': stats_dict,
        'embodiment_args_dict': emb_dict,
        'transformer_args': transformer_dict,
        'share_cross_attn': False,
    })
    ckpt = checkpoint_path(model_path, checkpoint)
    policy.deserialize(torch.load(ckpt, map_location=device, weights_only=True), mode='strict')
    policy.to(device)
    policy.eval()
    info = {
        'saved_config': saved_config,
        'emb_dict': emb_dict,
        'stats': stats_dict,
        'checkpoint': ckpt,
        'main_cam_size': mxt_camera_size(saved_config, 'main'),
        'wrist_cam_size': mxt_camera_size(saved_config, 'wrist'),
    }
    return policy, info
