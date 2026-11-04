#!/usr/bin/env bash
# Compatibility wrapper for the canonical timing-report runner.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$repo_root/scripts/run_dsv4_flash_h200_decode.sh" "$@"
