# Checkpoints

The checkpoints are not stored in this repository (it is a public fork, and the weights are kept private). They are
on the Hugging Face Hub, in private repos of the Relo02 account: ask for access, then log in with `hf auth login`.

| run | Hub repo | what it is |
|---|---|---|
| `rack_filtered_white_becker_HTD` | [`Relo02/htd-white-becker-filtered`](https://huggingface.co/Relo02/htd-white-becker-filtered) (private) | HTD (MXT + touch dreaming), white beaker task, filtered teleop data (99 episodes), 80,000 steps |

Download a run into the folder layout rollout expects, and pass that folder as `--model_path`:

```bash
hf download Relo02/htd-white-becker-filtered --local-dir runs/rack_filtered_white_becker_HTD
```

It holds `policy_last.ckpt` (step 80,000, what rollout loads by default), `policy_best.ckpt` (lowest validation
loss), `dataset_stats.pkl` (normalization; must stay with the checkpoint), `all_configs.json` and `config.pkl`
(training configuration), `dataset_split.json`, the training state and loss curves, and the open-loop replay
evaluation. SHA-256 of `policy_last.ckpt`: `e58157073b875d2c0a761e722273c8aac91035705ce65f95e56834ba3c5d8bdb`.
