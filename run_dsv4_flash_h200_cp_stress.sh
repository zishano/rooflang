#!/usr/bin/env bash
# Small-batch/long-context CP diagnostics, based on the H200 Pareto launcher.
# Never relax model validation or reuse the original Pareto output directory.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export MPLBACKEND=Agg
python_bin="${ROOFLANG_PYTHON:-$script_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" && -z "${ROOFLANG_PYTHON:-}" ]]; then
    python_bin=/mnt/nvme1n1/data/lmk/PROJECT/rooflang/.venv/bin/python
fi
[[ -x "$python_bin" ]] || { echo 'Set ROOFLANG_PYTHON to a working interpreter' >&2; exit 1; }
exec "$python_bin" -m rooflang.programs.experiments.cp_stress \
    --workers "${WORKERS:-2}" \
    --output-dir "${OUTPUT_DIR:-$script_dir/results/dsv4_flash_h200_cp_stress}" "$@"
