#!/usr/bin/env bash
set -euo pipefail
ROOFLANG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$(dirname -- "$ROOFLANG_DIR")${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$ROOFLANG_DIR/.cache/matplotlib"
export MPLBACKEND="${MPLBACKEND:-Agg}"
cd -- "$ROOFLANG_DIR"
python_bin="${ROOFLANG_PYTHON:-$ROOFLANG_DIR/.venv/bin/python}"
if [[ ! -x "$python_bin" && -z "${ROOFLANG_PYTHON:-}" ]]; then
  python_bin="/mnt/nvme1n1/data/lmk/PROJECT/rooflang/.venv/bin/python"
fi
exec "$python_bin" -m rooflang.programs.experiments.main "$@"

# ./run.sh \
#   --model dsv4_flash \
#   --hardware H200Cluster1Node \
#   --stage decode \
#   --batch-size 8 \
#   --cp 1 --dp 8 --ep 8 \
#   --pp-partition 43
