#!/usr/bin/env bash
set -euo pipefail

SECRET_FILE="${CANN_BENCH_SECRET_FILE:-/home/wsx/AscendOpGenAgent/.secrets/cannbench.env}"
if [[ -z "${CANN_BENCH_TOKEN:-}" && -r "$SECRET_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$SECRET_FILE"
  set +a
fi
if [[ -z "${CANN_BENCH_TOKEN:-}" ]]; then
  echo "CANN_BENCH_TOKEN is unset or unavailable." >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "${script_dir}/cannbench_hidden_followup.py" "$@"
