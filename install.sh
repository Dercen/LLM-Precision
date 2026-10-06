#!/usr/bin/env bash
# One-shot setup for ptq-bench. Run it from the project folder:
#
#   ./install.sh                # detect the NVIDIA driver and pick the matching torch build
#   ./install.sh --cpu          # no GPU, or force the CPU build
#   ./install.sh --cuda 126     # force a CUDA build (126 or 130)
#   ./install.sh --with-mlir    # also install the torch-mlir export add-on
#   ./install.sh --detect-only  # print the torch build this machine would get, and exit
#
# Safe to run again: every step is a no-op when already done. Afterwards `./ptq` opens
# the menu; `./ptq doctor` re-runs the checks.

set -euo pipefail
cd "$(dirname "$0")"

TORCH_EXTRA=""
WITH_MLIR=0
DETECT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) TORCH_EXTRA=cpu ;;
    --cuda) shift; TORCH_EXTRA="cu${1:-130}" ;;
    --with-mlir) WITH_MLIR=1 ;;
    --detect-only) DETECT_ONLY=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# The torch build depends on the NVIDIA driver: CUDA 13.0 wheels need driver 580 or
# newer, CUDA 12.6 wheels need 525 or newer, anything else (or no GPU) gets the CPU build.
pick_torch_extra() {
  local driver major
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo cpu; return
  fi
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)
  major=${driver%%.*}
  if [[ -z "$major" || ! "$major" =~ ^[0-9]+$ ]]; then
    echo cpu; return
  fi
  if (( major >= 580 )); then echo cu130
  elif (( major >= 525 )); then echo cu126
  else echo cpu
  fi
}

explain_extra() {
  case "$1" in
    cu130) echo "torch for CUDA 13.0 (NVIDIA driver 580 or newer)" ;;
    cu126) echo "torch for CUDA 12.6 (NVIDIA driver 525 to 579)" ;;
    cpu)   echo "torch for the CPU (no usable NVIDIA GPU, or driver older than 525)" ;;
    *)     echo "torch build '$1'" ;;
  esac
}

if [ -z "$TORCH_EXTRA" ]; then
  TORCH_EXTRA=$(pick_torch_extra)
fi
if [ "$DETECT_ONLY" = 1 ]; then
  echo "$TORCH_EXTRA"
  exit 0
fi

step() { printf '\n==> %s\n' "$*"; }

step "1/4  uv (the Python and environment manager)"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  echo "    already installed: $(uv --version)"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || { echo "uv did not install; see https://docs.astral.sh/uv/" >&2; exit 1; }
fi

step "2/4  Python packages: $(explain_extra "$TORCH_EXTRA")"
driver_line=$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
[ -n "$driver_line" ] && echo "    GPU: $driver_line"
echo "    this downloads about 3 GB the first time; later runs are quick"
# shellcheck disable=SC1091
source scripts/env.sh
unset UV_NO_SYNC
extras=(--extra "$TORCH_EXTRA" --extra hqq --extra dev)
[ "$WITH_MLIR" = 1 ] && extras+=(--extra mlir)
uv sync "${extras[@]}"

step "3/4  Launcher"
chmod +x ptq
echo "    ./ptq runs the program with the environment set up (no 'source' needed)"

step "4/4  Checks"
if ./ptq doctor; then
  printf '\nready: ./ptq opens the menu, ./ptq doctor repeats these checks\n'
else
  printf '\nsetup finished, but the checks above found something to fix first\n'
  exit 1
fi
