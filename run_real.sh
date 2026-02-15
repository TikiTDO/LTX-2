#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

INPUT_JSON="${1:-input.example.json}"

if [[ ! -f "$ROOT/venv/bin/activate" ]]; then
  echo "Missing venv at: $ROOT/venv" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

# Be robust even if the packages were not installed editable in the venv.
export PYTHONPATH="$ROOT/packages/ltx-core/src:$ROOT/packages/ltx-pipelines/src:${PYTHONPATH:-}"

# Helps reduce CUDA allocator fragmentation in long inference runs (safe default; can override externally).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export LTX_LOG_FILE="${LTX_LOG_FILE:-$ROOT/output/latest.log}"
mkdir -p "$(dirname -- "$LTX_LOG_FILE")"

# Capture stdout/stderr for easy tailing while also writing structured python logging to the same file.
exec python "$ROOT/run_from_json.py" "$INPUT_JSON" 2>&1 | tee "$LTX_LOG_FILE"
