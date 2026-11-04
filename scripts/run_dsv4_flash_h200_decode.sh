#!/usr/bin/env bash
# Configurable DSv4 Flash decode runner with post-simulation timing analysis.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
runner=${ROOFLANG_RUNNER:-$repo_root/run.sh}

usage() {
    cat <<'EOF'
Usage:
  scripts/run_dsv4_flash_h200_decode.sh [main.py options]

Runs one DSv4 Flash decode simulation and always generates a timing report.
Current defaults (override by passing the same option later):
  model=dsv4_flash, hardware=B300Cluster1Node, batch=512, context=8192
  CP/DP/EP=1/8/8, PP=[43], report-window=measured

Useful overrides:
  --hardware H200Cluster1Node
  --batch-size 8
  --context 65536
  --report-dir PATH
  --report-root PATH
  --report-window measured|full
  --focus-layer N
  --focus-device DEVICE_OR_RANK

Outputs include manifest/summary JSON, operator/communication/layer CSV,
Perfetto trace JSON, and PNG/PDF timelines. Existing output directories are
never overwritten.
EOF
}

if (($#)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

[[ -f "$runner" ]] || {
    echo "error: RoofLang runner not found: $runner" >&2
    exit 2
}

exec bash "$runner" \
  --model dsv4_flash \
  --hardware B300Cluster1Node \
  --stage decode \
  --batch-size 512 \
  --context 8192 \
  --cp 1 --dp 8 --ep 8 \
  --pp-partition 43 \
  --measurement_start read_input \
  --report \
  --report-window measured \
  "$@"
