#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
image="der-optuna-lgbm"
platform="linux/amd64"
competition_dir="/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems"
data_dir="$repo_root/data"
working_dir="$repo_root/kaggle-working"

usage() {
  cat <<EOF
Usage:
  ./run_docker_lgbm.sh [runner args]

Examples:
  ./run_docker_lgbm.sh
  ./run_docker_lgbm.sh --limit-rows 10000 --n-estimators 50

Build first:
  docker build --platform ${platform} -f Dockerfile.lgbm -t ${image} .

Expected local input files:
  data/train.csv
  data/test.csv

Output:
  kaggle-working/submission_LGBM.csv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$data_dir" ]]; then
  printf 'Missing data directory: %s\n' "$data_dir" >&2
  exit 1
fi

mkdir -p "$working_dir"

if ! command -v docker >/dev/null 2>&1; then
  printf 'docker was not found. Open Docker Desktop or install Docker first.\n' >&2
  exit 1
fi

if [[ -z "$(docker image ls -q "$image")" ]]; then
  printf 'Docker image %s was not found. Build it with:\n' "$image" >&2
  printf '  docker build --platform %s -f Dockerfile.lgbm -t %s .\n' "$platform" "$image" >&2
  exit 1
fi

exec docker run --rm \
  --platform "$platform" \
  -v "$repo_root:/workspace" \
  -v "$data_dir:$competition_dir:ro" \
  -v "$working_dir:/kaggle/working" \
  -w /workspace \
  "$image" \
  python -m src.optuna_lgbm_runner "$@"

