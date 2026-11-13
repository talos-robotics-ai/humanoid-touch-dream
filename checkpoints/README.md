# Checkpoints

The checkpoints are not stored in this repository (it is a public fork, and the weights are kept private). They
are in the private Hugging Face repo [`Relo02/htd-checkpoints`](https://huggingface.co/Relo02/htd-checkpoints),
one folder per training run: ask for access, then log in with `hf auth login`.

| folder | what it is |
|---|---|
| `rack_filtered_white_becker_HTD` | HTD (MXT + touch dreaming), white beaker task, filtered teleop data (99 episodes), 80,000 steps |

## Download a run

Into the folder layout rollout expects; pass the folder as `--model_path`:

```bash
hf download Relo02/htd-checkpoints --include "rack_filtered_white_becker_HTD/*" --local-dir runs
```

A run folder holds `policy_last.ckpt` (what rollout loads by default), `policy_best.ckpt` (lowest validation loss),
`dataset_stats.pkl` (normalization; must stay with the checkpoint), `all_configs.json` and `config.pkl` (training
configuration), `dataset_split.json`, the training state and loss curves, and the replay evaluation summary.

## Push a run

```bash
scripts/push_checkpoint.sh runs/<run_folder>          # -> Relo02/htd-checkpoints/<run_folder>/
scripts/push_checkpoint.sh runs/<run_folder> <name>   # -> Relo02/htd-checkpoints/<name>/
```

A new name is a new folder; pushing to an existing name replaces the files there (the Hub keeps earlier versions
in its history). The optimizer, `training_last.ckpt` and the intermediate step checkpoints are left out (`ALL=1`
uploads the whole folder). Then add the folder to the table here and in the Hub repo's README.
