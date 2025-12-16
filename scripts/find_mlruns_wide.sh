#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Wide search for MLflow local tracking directories (mlruns) that contain RUN folders.
#   Useful when workspace mlruns is empty and project-local scan finds nothing.
#
# Usage:
#   bash /mnt/c/Users/lc999/RD-Agent-main/scripts/find_mlruns_wide.sh
#   SEARCH_ROOTS="/some/path /other/path" bash .../find_mlruns_wide.sh

DEFAULT_ROOTS=(
  "$PWD"
  "$HOME"
  "$HOME/.qlib"
  "$HOME/.cache"
  "$HOME/tmp"
  "/tmp"
  "/mnt/c/Users/lc999"
  "/mnt/c/Users/lc999/RD-Agent-main"
  "/mnt/c/Users/lc999/RD-Agent-main/git_ignore_folder/RD-Agent_workspace"
  "/mnt/c/Users/lc999/NewAIstock"
)

if [ -n "${SEARCH_ROOTS:-}" ]; then
  # shellcheck disable=SC2206
  ROOTS=( ${SEARCH_ROOTS} )
else
  ROOTS=("${DEFAULT_ROOTS[@]}")
fi

echo "[env] whoami=$(whoami)" >&2
echo "[env] HOME=$HOME" >&2
echo "[env] PWD=$PWD" >&2
echo "[env] MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-<unset>}" >&2
echo "[env] QLIB_LOG_DIR=${QLIB_LOG_DIR:-<unset>}" >&2
echo "" >&2

echo "[find_mlruns_wide] searching roots:" >&2
for r in "${ROOTS[@]}"; do
  echo "  - $r" >&2
done

echo "" >&2

declare -A seen=()
found_any=0

# Helper: print a tracking_dir summary
print_tracking_dir() {
  local tracking_dir="$1"
  local exp_cnt run_meta_cnt

  exp_cnt=$(find "$tracking_dir" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | wc -l | tr -d ' ')
  run_meta_cnt=$(find "$tracking_dir" -type f -path '*/mlruns/[0-9]*/[0-9a-f]*/meta.yaml' 2>/dev/null | wc -l | tr -d ' ')

  echo "===================="
  echo "tracking_dir=$tracking_dir"
  echo "experiments=$exp_cnt run_meta=$run_meta_cnt"

  # Show newest 3 run metas
  find "$tracking_dir" -type f -path '*/mlruns/[0-9]*/[0-9a-f]*/meta.yaml' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 3 \
    | while read -r _ p; do
        echo "---"
        echo "meta=$p"
        egrep 'run_id:|artifact_uri:|end_time:|start_time:|status:' "$p" || true
      done
}

for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue

  # Find run-level meta.yaml: .../mlruns/<exp_id>/<run_id>/meta.yaml
  while IFS= read -r meta; do
    tracking_dir="${meta%/mlruns/*}/mlruns"

    if [ -z "${seen[$tracking_dir]+x}" ]; then
      seen[$tracking_dir]=1
      found_any=1
      print_tracking_dir "$tracking_dir"
    fi
  done < <(
    find "$root" -type f -path '*/mlruns/[0-9]*/[0-9a-f]*/meta.yaml' ! -path '*/mlruns/[0-9]*/meta.yaml' 2>/dev/null
  )

done

if [ "$found_any" -eq 0 ]; then
  echo "No run-level meta.yaml found under provided roots." >&2
  echo "Hint: if tracking is remote (http/https) or a non-local store, local scanning will find nothing." >&2
  echo "      Please also check environment variables above and Qlib config exp_manager settings." >&2
  exit 2
fi
