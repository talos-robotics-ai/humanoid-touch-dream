"""Tests for inference_engine.py with a small stand-in policy. Needs a CUDA GPU (skips otherwise).

Run: python test_inference_engine.py  (or pytest), in the htd_policy environment.
"""
import torch
import torch.nn as nn

import inference_engine as ie

EMB = 'g1_inspire_wb'


class _Model(nn.Module):
    """Shaped like MXT's model: uint8 image scaled inside, masks checked unless known present."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, stride=4)
        self.fc = nn.Linear(4 + 5, 6)
        self.masks_known_present = False
        self.saw_mask_check = 0

    def forward(self, obs, mask, infer_dream=False):
        img = obs['main_image'].to(torch.float32) / 255.0
        if not self.masks_known_present:
            self.saw_mask_check += 1
            if not bool(torch.any(mask['proprio'])):  # GPU -> CPU read, like MXT's tokenize
                return {'a': torch.zeros(1, 2, 3, device=img.device)}
        proprio = obs['proprio'].to(torch.float32)  # MXT converts every input to float32
        feat = torch.cat([self.conv(img).mean((2, 3)), proprio * mask['proprio']], -1)
        out = {'a': self.fc(feat).view(1, 2, 3)}
        return (out, {'d': out['a'] * 2}) if infer_dream else out


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Model()

    def forward(self, obs, embodiment, mask, infer_dream=False):
        return self.model(obs, mask, infer_dream)


def _inputs(seed, channels_last_view=True):
    g = torch.Generator().manual_seed(seed)
    hwc = torch.randint(0, 256, (48, 64, 3), generator=g, dtype=torch.uint8)
    img = hwc.permute(2, 0, 1)[None] if channels_last_view else hwc.permute(2, 0, 1).contiguous()[None]
    return {'main_image': img, 'proprio': torch.randn(1, 5, generator=g)}, \
        {'main_image': torch.ones(1, 1, dtype=torch.bool), 'proprio': torch.ones(1, 5, dtype=torch.bool)}


def _setup():
    torch.manual_seed(0)
    policy = _Policy().cuda().eval()
    return policy, ie.FastInference(policy, EMB, 'cuda:0')


def _reference(policy, obs, mask, infer_dream=False):
    obs = {k: v.contiguous().cuda() for k, v in obs.items()}
    with torch.no_grad():
        return policy(obs, EMB, {k: v.cuda() for k, v in mask.items()}, infer_dream=infer_dream)


def test_graph_matches_eager_on_new_inputs():
    policy, eng = _setup()
    assert eng.warmup(*_inputs(0)).startswith('CUDA graph')
    for seed in (1, 2, 3):
        obs, mask = _inputs(seed)
        assert torch.equal(eng(obs, mask)['a'], _reference(policy, obs, mask)['a'])


def test_dream_variant_captured_separately():
    policy, eng = _setup()
    eng.warmup(*_inputs(0))
    assert eng.warmup(*_inputs(0), infer_dream=True).startswith('CUDA graph')
    obs, mask = _inputs(4)
    out, dream = eng(obs, mask, infer_dream=True)
    ref, ref_d = _reference(policy, obs, mask, infer_dream=True)
    assert torch.equal(out['a'], ref['a']) and torch.equal(dream['d'], ref_d['d'])


def test_outputs_not_overwritten_by_next_call():
    _, eng = _setup()
    eng.warmup(*_inputs(0))
    first = eng(*_inputs(1))['a']
    keep = first.clone()
    eng(*_inputs(2))
    assert torch.equal(first, keep)


def test_falls_back_to_eager_when_input_changes():
    policy, eng = _setup()
    eng.warmup(*_inputs(0))
    obs, mask = _inputs(5)
    mask['proprio'][:] = False  # a modality missing: must take the model's own mask path
    before = policy.model.saw_mask_check
    assert torch.equal(eng(obs, mask)['a'], torch.zeros(1, 2, 3, device='cuda'))
    assert policy.model.saw_mask_check == before + 1
    obs, mask = _inputs(6)
    obs['proprio'] = obs['proprio'].double()  # different dtype: eager
    assert eng(obs, mask)['a'].shape == (1, 2, 3)


def test_channels_last_view_reordered_on_gpu():
    policy, eng = _setup()
    obs, mask = _inputs(7, channels_last_view=True)
    obs_c, _ = _inputs(7, channels_last_view=False)
    assert torch.equal(eng._to_gpu(obs['main_image']), obs_c['main_image'].cuda())
    eng.warmup(obs, mask)
    assert torch.equal(eng(obs, mask)['a'], _reference(policy, obs_c, mask)['a'])


def test_graph_disabled_is_eager():
    policy = _Policy().cuda().eval()
    eng = ie.FastInference(policy, EMB, 'cuda:0', cuda_graph=False)
    assert eng.warmup(*_inputs(0)).startswith('eager') and not eng.graphs


def test_require_cuda_rejects_cpu():
    for bad in ('cpu',):
        try:
            ie.require_cuda(bad)
        except RuntimeError:
            continue
        raise AssertionError(f'accepted {bad}')
    try:
        ie.require_cuda('cuda:0', _Policy())  # parameters still on the CPU
    except RuntimeError:
        return
    raise AssertionError('accepted a policy on the CPU')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('skipped: no CUDA GPU')
    else:
        tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
        for fn in tests:
            fn()
            print(f'ok  {fn.__name__}')
        print(f'{len(tests)} tests passed')
