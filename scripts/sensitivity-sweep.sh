#!/usr/bin/env bash
# Per-layer sensitivity sweep for one model: quantize at 4, 3 and 2 bits with RTN and GPTQ,
# scoring every layer's output error on real text (--score-windows), then write the tables
# and the heatmap (ptq sensitivity). The exports land in exports/sensitivity/ (git-ignored,
# nn.Module + JSON only); the tables in results/sensitivity/<model>.md and .csv, the chart
# in results/plots/<model>_sensitivity.png, which the model's results page picks up.
#
#   scripts/sensitivity-sweep.sh smollm2-135m 64                      # model key, group size
#   EXTRA="--act-order" scripts/sensitivity-sweep.sh smollm2-135m 64  # GPTQ act-order, as the Llama rows use
#   WINDOWS=32 ALGOS="rtn gptq hqq" BITS="4 3" scripts/sensitivity-sweep.sh opt-125m 128
set -euo pipefail
model=${1:?usage: $0 <model key> [group size]}
gs=${2:-64}
windows=${WINDOWS:-16}      # text windows the layer outputs are scored on
algos=${ALGOS:-"rtn gptq"}
bits_list=${BITS:-"4 3 2"}
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/env.sh
out=exports/sensitivity
for algo in $algos; do
  for bits in $bits_list; do
    dir="$out/$model-$algo$bits-g$gs"
    if [ -f "$dir/layers.json" ]; then
      echo "have $dir"
      continue
    fi
    # shellcheck disable=SC2086
    uv run --no-sync ptq export-mlir --model "$model" --algo "$algo" --bits "$bits" --group-size "$gs" \
      --score-windows "$windows" --no-mlir --out "$dir" ${EXTRA:-}
  done
done
uv run --no-sync ptq sensitivity "$out/$model"-*-g"$gs"/layers.json
