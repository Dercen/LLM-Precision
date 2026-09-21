#!/usr/bin/env bash
# Source me:  source scripts/env.sh
# Shell-scoped environment for ptq-bench. Safe to source repeatedly.

export PATH="$HOME/.local/bin:$PATH"          # uv is not on PATH in a batch shell

# Cache locations. $SCRATCH is used when set (cluster), else ~/ml and ~/.cache (laptop).
_ptq_root="${SCRATCH:-$HOME/ml}"
export HF_HOME="${HF_HOME:-$_ptq_root/hf}"
export PTQ_CACHE_DIR="${PTQ_CACHE_DIR:-$_ptq_root/ptq-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:+$SCRATCH/uv-cache}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SCRATCH:+$SCRATCH/torch-ext}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch-ext}"
mkdir -p "$HF_HOME" "$PTQ_CACHE_DIR" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# DISABLE_CUDA is hqq's setup.py flag, NOT a torch flag. It stops the hqq sdist
# shelling out to a CUDA extension build on every lock and sync. torch.cuda is
# unaffected; `ptq env-check` asserts CUDA is live so this cannot regress unnoticed.
export DISABLE_CUDA=1

# uv run must never re-sync: without this, a `uv run` issued without the same
# --extra set re-resolves torch from PyPI (the CUDA 13.0 build on Linux).
export UV_NO_SYNC=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc --all 2>/dev/null | awk '{print int($1/2)}')}"
export TOKENIZERS_PARALLELISM=false

mkdir -p results/runs results/logs results/slurm 2>/dev/null || true
unset _ptq_root
