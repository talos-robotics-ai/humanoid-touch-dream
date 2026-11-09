"""Fast, GPU-only MXT/HTD inference for real-time rollout.

One policy forward is ~1,200 small CUDA kernels; launching them from Python
costs far more than running them. ``FastInference`` captures the whole forward
once as a CUDA graph and then replays it, so one inference is one launch plus
copying the new observation into fixed GPU buffers. The graph is checked against
the normal (eager) forward at start-up and inference falls back to eager if the
capture fails or the outputs differ.

The model must be on a CUDA device; ``require_cuda`` refuses anything else.
"""
import time

import torch

GRAPH_TOLERANCE = 1e-4  # max |graph - eager| in output units (m, rad, m/s) to accept the graph


def require_cuda(device, policy=None):
    """Return the torch.device; raise unless it is an available CUDA device holding the policy."""
    device = torch.device(device)
    if device.type != 'cuda':
        raise RuntimeError(f'inference device is {device}; real-time rollout needs a CUDA GPU (--device cuda:0)')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available: check the NVIDIA driver and that PyTorch is a CUDA build '
                           '(RTX 50-series GPUs need PyTorch >= 2.7 built for CUDA 12.8)')
    if policy is not None:
        wrong = {str(p.device) for p in policy.parameters() if p.device.type != 'cuda'}
        if wrong:
            raise RuntimeError(f'policy parameters are on {sorted(wrong)}, not the GPU')
    return device


def _max_diff(a, b):
    if isinstance(a, dict):
        return max(_max_diff(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return max(_max_diff(x, y) for x, y in zip(a, b))
    return float((a.float() - b.float()).abs().max())


def _clone(out):
    if isinstance(out, dict):
        return {k: _clone(v) for k, v in out.items()}
    if isinstance(out, (tuple, list)):
        return type(out)(_clone(v) for v in out)
    return out.clone()


class FastInference:
    """``engine(obs, mask, infer_dream=False)`` -> what ``policy(obs, embodiment, mask, ...)`` returns.

    obs / mask: dicts of batch-1 tensors, on the CPU or the GPU. Images may be uint8
    (the model scales them on the GPU, as in training).
    """

    def __init__(self, policy, embodiment, device, cuda_graph=True):
        self.policy, self.embodiment = policy, embodiment
        self.device = require_cuda(device, policy)
        self.use_graph = bool(cuda_graph)
        self.graphs = {}   # infer_dream -> (graph, static_obs, static_mask, static_out)
        self.status = {}
        torch.backends.cudnn.benchmark = True  # fixed input shapes: pick the fastest conv algorithms once
        self.gpu_name = torch.cuda.get_device_name(self.device)

    @staticmethod
    def masks_all_present(mask):
        """True if every mask is all True. Check CPU tensors: on the GPU this read is a sync."""
        return all(bool(m.all()) for m in mask.values())

    def _forward(self, obs, mask, infer_dream, present):
        """policy(...) with the model told whether all masks are known to be True."""
        model = getattr(self.policy, 'model', None)
        if model is not None:
            model.masks_known_present = present
        try:
            return self.policy(obs, self.embodiment, mask, infer_dream=infer_dream)
        finally:
            if model is not None:
                model.masks_known_present = False

    def _to_gpu(self, v):
        """Copy to the GPU. A channels-first *view* of a channels-last CPU image is copied in
        its memory order and reordered on the GPU, which is much faster than on the CPU."""
        if v.device.type == 'cpu' and v.dim() == 4 and not v.is_contiguous() \
                and v.permute(0, 2, 3, 1).is_contiguous():
            return v.permute(0, 2, 3, 1).to(self.device, non_blocking=True).permute(0, 3, 1, 2)
        return v.to(self.device, non_blocking=True)

    def _eager(self, obs, mask, infer_dream):
        present = self.masks_all_present(mask)
        obs = {k: self._to_gpu(v) for k, v in obs.items()}
        mask = {k: v.to(self.device, non_blocking=True) for k, v in mask.items()}
        with torch.inference_mode():
            return self._forward(obs, mask, infer_dream, present)

    def warmup(self, obs, mask, infer_dream=False, eager_runs=3):
        """Run the eager forward a few times, then capture and verify the CUDA graph.

        Returns a one-line description of what inference will use.
        """
        for _ in range(eager_runs):
            ref = self._eager(obs, mask, infer_dream)
        torch.cuda.synchronize(self.device)
        if not self.use_graph:
            self.status[infer_dream] = 'eager (CUDA graph disabled)'
            return self.status[infer_dream]
        if not self.masks_all_present(mask):
            self.status[infer_dream] = 'eager (warm-up input has missing modalities)'
            return self.status[infer_dream]
        try:
            static_obs = {k: self._to_gpu(v).contiguous() for k, v in obs.items()}
            static_mask = {k: v.to(self.device).clone() for k, v in mask.items()}
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(2):
                    self._forward(static_obs, static_mask, infer_dream, True)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                static_out = self._forward(static_obs, static_mask, infer_dream, True)
            self.graphs[infer_dream] = (graph, static_obs, static_mask, static_out)
            got = self(obs, mask, infer_dream)
            diff = _max_diff(got, ref)
            if not diff <= GRAPH_TOLERANCE:
                del self.graphs[infer_dream]
                self.status[infer_dream] = f'eager (CUDA graph output differs by {diff:.2e})'
            else:
                self.status[infer_dream] = f'CUDA graph (matches eager to {diff:.1e})'
        except Exception as e:  # capture unsupported by some op: stay eager
            self.graphs.pop(infer_dream, None)
            self.status[infer_dream] = f'eager (CUDA graph capture failed: {type(e).__name__}: {str(e)[:120]})'
        return self.status[infer_dream]

    def __call__(self, obs, mask, infer_dream=False):
        entry = self.graphs.get(infer_dream)
        if entry is None:
            return self._eager(obs, mask, infer_dream)
        graph, static_obs, static_mask, static_out = entry
        if (obs.keys() != static_obs.keys() or mask.keys() != static_mask.keys()
                or any(obs[k].shape != static_obs[k].shape or obs[k].dtype != static_obs[k].dtype for k in obs)
                or not self.masks_all_present(mask)):
            return self._eager(obs, mask, infer_dream)  # input differs from the captured one: never feed the graph
        for k, v in obs.items():
            static_obs[k].copy_(self._to_gpu(v))
        for k, v in mask.items():
            static_mask[k].copy_(v, non_blocking=True)
        graph.replay()
        return _clone(static_out)

    def time_ms(self, obs, mask, infer_dream=False, n=20):
        """Median wall time of one inference including input copy and output read-back."""
        ts = []
        for _ in range(n):
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            out = self(obs, mask, infer_dream)
            first = out[0] if isinstance(out, tuple) else out
            next(iter(first.values())).cpu()
            ts.append(1000 * (time.perf_counter() - t0))
        return sorted(ts)[len(ts) // 2]
