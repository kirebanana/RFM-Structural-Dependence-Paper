#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DATASET="${1:-${HF_DATASET_DIR:-}}"
OUT_DIR="${2:-${ROOT}/artifacts/clean_rewire_preprocessed}"

if [[ -z "$RAW_DATASET" ]]; then
  printf '%s\n' 'Pass a local rel-f1 directory or set HF_DATASET_DIR.' >&2
  exit 2
fi

cd "$ROOT"
uv run python vendor/relational-transformer/scripts/preprocess.py one \
  --dataset "$RAW_DATASET" \
  --out-dir "$OUT_DIR" \
  --embedding-model all-MiniLM-L12-v2
