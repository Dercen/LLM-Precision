# Server hand-off (M7)

The laptop produced the complete result set on its own (see [RESULTS.md](RESULTS.md));
a server adds scale, not credibility, and needs no code change — extras rather than
platform markers select the torch index, `--shard k/n` splits a matrix across GPUs,
and rows merge by `run_id`. What follows is the plan's §11, kept as written.


Nothing here is needed until M5 is demonstrated, and nothing here requires a code change — it is why extras rather than platform markers select the torch index.

Setup, when access arrives:

```bash
nvidia-smi; ldd --version | head -1        # driver picks cu126 vs cu130; glibc must be >= 2.28
git clone <origin url> ~/ptq-bench && cd ~/ptq-bench
#   scripts/env.sh reads $SCRATCH when set and falls back to ~/ml locally, so it is used unedited
source scripts/env.sh
uv python install 3.12
uv sync --extra cu130 --extra hqq --extra dev    # or --extra cu126 on an older driver
uv run ptq env-check && uv run pytest -m smoke
```

Code moves by git; results return as JSON files: `rsync -av --exclude '*.log' user@server:~/ptq-bench/results/runs/ results/runs/`, and the aggregator dedupes by `run_id`. (v1's `pull_results.ps1` / `scp` workaround existed only because Git Bash ships no rsync; both machines are Linux now.)

Plain SSH box: `tmux new -s ptq`, `source scripts/env.sh`, `uv run ptq prefetch configs/experiments/streamed_llama.yaml`, then one loop per GPU: `CUDA_VISIBLE_DEVICES=$i nohup uv run --frozen --no-sync ptq run $CFG --shard $i/$N > results/logs/worker$i.log 2>&1 &`.

SLURM sketch (`source scripts/env.sh` in the login shell before `sbatch` so `results/slurm` exists; `--gres=gpu:1` vs `--gpus=1` is cluster-specific):

```bash
#!/bin/bash
#SBATCH --job-name=ptq --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00
#SBATCH --array=0-7 --output=results/slurm/%x_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; source scripts/env.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run --frozen --no-sync ptq run "$1" --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}"
```

`--frozen --no-sync` matters: a plain `uv run` locks and syncs first, which can touch the network on an offline compute node. `ptq run --list` prints the group count to size the array; `prefetch.sbatch` runs first on a node with internet. Resume is re-submitting the same command. On the server the streamed tier becomes resident again (24 GB+ cards), which is exactly the cross-machine check in M7's definition of done. `module load cuda/X.Y` is needed in `env.sh` only if GPTQModel's JIT is used there, and the module's nvcc major.minor must equal `torch.version.cuda`.

