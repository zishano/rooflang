#!/usr/bin/env bash
# 图3/4的Flash专用实验：固定完整矩阵，仅允许调整运行方式和源码来源。
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
args=()
while (($#)); do
  case "$1" in
    --workers|--output-dir|--source)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        exit 2
      fi
      args+=("$1" "$2")
      shift 2
      ;;
    --dry-run|--resume|--plot-only)
      args+=("$1")
      shift
      ;;
    -h|--help)
      cat <<'HELP'
Usage: ./run_dsv4_flash_paper_figures.sh [options]
Fixed: DSV4 Flash; prefill+decode; 64K+1M; 8+64 GPUs;
       H200+GH200+B300+GB300; all legal balanced PP/CP/DP/EP;
       batch doubles until baseline OOM; ideal-overlap frontier only.
Options:
  --dry-run              Enumerate sweeps without simulation
  --workers N            CPU simulation processes (default 1)
  --output-dir PATH      New output directory (required for resume/plot-only)
  --resume               Resume the saved Flash experiment
  --plot-only            Plot saved results without simulation
  --source head|workspace  Source snapshot (default head, excludes uncommitted edits)
The preset does not accept batch caps or parallelism/model restrictions.
HELP
      exit 0
      ;;
    *) echo "Unsupported option: $1 (see --help)" >&2; exit 2 ;;
  esac
done
exec "$script_dir/run_reproduce_figures_3_4.sh" --flash-paper "${args[@]}"
