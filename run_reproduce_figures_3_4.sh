#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON_BIN:-$script_dir/.venv/bin/python}" -B \
  "$script_dir/programs/experiments/reproduce_figures_3_4.py" "$@"
