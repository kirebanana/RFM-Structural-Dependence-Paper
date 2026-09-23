#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || {
  printf '%s\n' 'uv is required: https://docs.astral.sh/uv/' >&2
  exit 1
}

uv python install 3.12.13
export UV_PYTHON_PREFERENCE=only-managed
export PYO3_PYTHON="$(uv python find 3.12.13)"
uv sync --python 3.12.13 --extra dev --extra rt

printf '%s\n' 'Environment ready. Use `uv run ...` from this project.'
