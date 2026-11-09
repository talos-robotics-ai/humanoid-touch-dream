"""Check that this machine runs an HTD/MXT checkpoint on its GPU fast enough for rollout.

Loads the checkpoint exactly as policy_rollout.py does (mxt_loading + FastInference),
captures and verifies the CUDA graph, and reports the inference time. Needs no robot,
camera or dataset: by default it feeds a fixed synthetic observation (a seeded random
image and proprioception drawn around the training statistics).

Comparing two machines (e.g. the training server and the robot laptop):

    # on machine A
    python check_inference.py --model_path RUN_DIR --save_reference ref.pt
    # copy ref.pt to machine B, then
    python check_inference.py --model_path RUN_DIR --reference ref.pt

The second run feeds A's exact input and reports how far B's actions are from A's.
Different GPUs and PyTorch versions differ by floating-point rounding only, far
below a millimetre.

Run from policy_rollout/ in the rollout environment. Exits non-zero if the GPU is
unusable, the outputs disagree with the reference, or inference is too slow for
--control_freq.
"""
import argparse
import platform
import sys

import numpy as np
import torch

from inference_engine import FastInference
from mxt_loading import checkpoint_path, load_mxt_policy, resolve_run_configs

EMBODIMENT = 'g1_inspire_wb'


def synthetic_observation(stats, main_cam_size, seed=0):
    """A deterministic batch-1 observation shaped like policy_rollout's (CPU tensors)."""
    g = torch.Generator().manual_seed(seed)
    obs, mask = {}, {}
    for key, s in stats[EMBODIMENT]['obs'].items():
        if 'image' in key:
            h, w = main_cam_size
            obs[key] = torch.randint(0, 256, (1, 3, h, w), generator=g, dtype=torch.uint8)
            mask[key] = torch.ones(1, 1, dtype=torch.bool)
        else:
            mean = torch.as_tensor(np.asarray(s['mean']), dtype=torch.float32)
            std = torch.as_tensor(np.asarray(s['std']), dtype=torch.float32)
            obs[key] = (mean + std * torch.randn(mean.shape, generator=g))[None]
            mask[key] = torch.ones(1, mean.numel(), dtype=torch.bool)
    return obs, mask


def describe_machine(device):
    return {'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'host': platform.node()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', required=True, help='run directory (policy_*.ckpt, all_configs.json, dataset_stats.pkl)')
    p.add_argument('--checkpoint', default='policy_last.ckpt')
    p.add_argument('--embodiment_config_path', default=None, help='default: the YAML recorded in all_configs.json')
    p.add_argument('--trunk_config_path', default=None, help='default: the YAML recorded in all_configs.json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cuda_graph', type=int, default=1, help='1 = as rollout (default), 0 = normal forward only')
    p.add_argument('--control_freq', type=float, default=30.0, help='rollout rate the timing is judged against')
    p.add_argument('--runs', type=int, default=50)
    p.add_argument('--save_reference', default=None, help='write the input and this machine\'s outputs here')
    p.add_argument('--reference', default=None, help='compare against a file written by --save_reference')
    p.add_argument('--tolerance', type=float, default=1e-3,
                   help='max allowed |difference| to the reference, in action units (m, rad, m/s)')
    args = p.parse_args()

    emb_cfg, trunk_cfg = resolve_run_configs(args.model_path, args.embodiment_config_path, args.trunk_config_path)
    policy, info = load_mxt_policy(args.model_path, emb_cfg, trunk_cfg, args.device, checkpoint=args.checkpoint)
    engine = FastInference(policy, EMBODIMENT, args.device, cuda_graph=bool(args.cuda_graph))
    machine = describe_machine(engine.device)
    print(f"checkpoint: {checkpoint_path(args.model_path, args.checkpoint)}")
    print(f"machine:    {machine['gpu']}, torch {machine['torch']} (CUDA {machine['cuda']}), host {machine['host']}")

    ref = torch.load(args.reference, weights_only=False) if args.reference else None
    if ref is not None:
        obs, mask = ref['obs'], ref['mask']
    else:
        obs, mask = synthetic_observation(info['stats'], info['main_cam_size'])

    status = engine.warmup(obs, mask)
    ms = sorted(engine.time_ms(obs, mask, n=1) for _ in range(args.runs))
    median, p95 = ms[len(ms) // 2], ms[int(0.95 * (len(ms) - 1))]
    out = {k: v.float().cpu() for k, v in engine(obs, mask).items()}
    print(f'inference:  {status}')
    print(f'time:       median {median:.1f} ms, p95 {p95:.1f} ms over {args.runs} runs '
          f'(control step at {args.control_freq:g} Hz = {1000 / args.control_freq:.1f} ms)')

    ok = True
    if ref is not None:
        print(f"reference:  {ref['machine']['gpu']}, torch {ref['machine']['torch']}")
        worst = 0.0
        for k, v in ref['out'].items():
            d = float((out[k] - v).abs().max())
            worst = max(worst, d)
            print(f'  {k:24s} max |diff| {d:.2e}')
        ok = worst <= args.tolerance
        print(f"{'PASS' if ok else 'FAIL'}: max difference {worst:.2e} (tolerance {args.tolerance:g})")
    if args.save_reference:
        torch.save({'obs': obs, 'mask': mask, 'out': out, 'machine': machine,
                    'checkpoint': args.checkpoint, 'status': status}, args.save_reference)
        print(f'saved reference to {args.save_reference}')
    # Rollout starts each inference early enough to absorb its time; past about one
    # chunk interval (45 steps = 1.5 s at 30 Hz) chunks would run out. Warn well before.
    budget_ms = 0.5 * 1000 * 45 / args.control_freq
    if p95 > budget_ms:
        print(f'FAIL: p95 inference {p95:.0f} ms is over {budget_ms:.0f} ms; rollout would lag')
        ok = False
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
