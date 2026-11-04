#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$script_dir/.cache/matplotlib"
exec "$script_dir/.venv/bin/python" -B -m rooflang.programs.experiments.architecture_one_ablation "$@"
