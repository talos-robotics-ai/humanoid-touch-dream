#!/usr/bin/env bash
# Push one HTD training run to the private Hugging Face checkpoint repo, as a folder of its own.
#
#   scripts/push_checkpoint.sh RUN_DIR [NAME]        e.g. scripts/push_checkpoint.sh runs/rack_filtered_white_becker_HTD
#
# The run lands in <repo>/NAME/ (NAME: the run folder's name by default). Another NAME is another folder; the same
# NAME again replaces the files it uploads there (the Hub keeps the earlier versions in its history). Uploaded: what
# rollout loads (policy_last.ckpt, policy_best.ckpt, dataset_stats.pkl, all_configs.json, config.pkl) and what
# documents the run (dataset_split.json, training_state.json, the loss curves, a README.md, the replay evaluation's
# summary.md); not the optimizer, training_last.ckpt or the intermediate step checkpoints (ALL=1 uploads the whole
# folder). Needs the hf CLI logged in with write access (hf auth login). HTD_CKPT_REPO overrides the repo.
set -euo pipefail
REPO=${HTD_CKPT_REPO:-Relo02/htd-checkpoints}
RUN=${1:?usage: $0 RUN_DIR [NAME]}
NAME=${2:-$(basename "$(cd "$RUN" && pwd)")}
for f in policy_last.ckpt dataset_stats.pkl all_configs.json; do
    [[ -f "$RUN/$f" ]] || { echo "$RUN has no $f: not a training run folder" >&2; exit 2; }
done
command -v hf >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub" >&2; exit 1; }
FILTER=()
if [[ "${ALL:-0}" != 1 ]]; then
    for p in policy_last.ckpt policy_best.ckpt dataset_stats.pkl all_configs.json config.pkl dataset_split.json \
             training_state.json "*.png" README.md "replay_eval*/summary.md" "replay_eval*.md"; do
        FILTER+=(--include "$p")
    done
fi
step=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('step', '?'))" "$RUN/training_state.json" 2>/dev/null || echo "?")
echo "pushing $RUN to $REPO/$NAME/ (step $step)"
hf upload "$REPO" "$RUN" "$NAME" --repo-type model "${FILTER[@]}" --commit-message "$NAME: step $step"
echo "done: https://huggingface.co/$REPO/tree/main/$NAME"
