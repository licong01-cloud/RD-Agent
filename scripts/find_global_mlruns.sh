#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Locate MLflow tracking directories (mlruns) that actually contain RUN subfolders.
#   This is useful when workspace/mlruns only has experiment meta.yaml but no runs.
#
# Usage:
#   bash scripts/find_global_mlruns.sh
#   SEARCH_ROOTS="/mnt/c/... /mnt/d/..." bash scripts/find_global_mlruns.sh

DEFAULT_ROOTS=(
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

echo "[find_global_mlruns] searching roots:" >&2
for r in "${ROOTS[@]}"; do
  echo "  - $r" >&2
done

echo "" >&2

# Under `set -u`, referencing an undeclared array (even for length) can error.
# Ensure the associative array is always initialized.
declare -A seen=()

for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue

  # Find run-level meta.yaml: .../mlruns/<exp_id>/<run_id>/meta.yaml
  # Exclude experiment-level meta: .../mlruns/<exp_id>/meta.yaml
  while IFS= read -r meta; do
    # Normalize to tracking dir (the directory named mlruns)
    tracking_dir="${meta%/mlruns/*}/mlruns"

    if [ -z "${seen[$tracking_dir]+x}" ]; then
      seen[$tracking_dir]=1

      # Count experiments and run-meta files (cheap summary)
      exp_cnt=$(find "$tracking_dir" -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null | wc -l | tr -d ' ')
      # run folder names are not guaranteed to be hex; match any subdir under mlruns/<exp_id>/
      run_meta_cnt=$(find "$tracking_dir" -type f -path '*/mlruns/[0-9]*/*/meta.yaml' ! -path '*/mlruns/[0-9]*/meta.yaml' 2>/dev/null | wc -l | tr -d ' ')

      echo "===================="
      echo "tracking_dir=$tracking_dir"
      echo "experiments=$exp_cnt run_meta=$run_meta_cnt"

      # Show newest 3 run metas to confirm activity + reveal artifact_uri
      find "$tracking_dir" -type f -path '*/mlruns/[0-9]*/*/meta.yaml' ! -path '*/mlruns/[0-9]*/meta.yaml' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 3 \
        | while read -r _ p; do
            echo "---"
            echo "meta=$p"
            egrep 'run_id:|artifact_uri:|end_time:|start_time:|status:' "$p" || true
          done
    fi
  done < <(
    find "$root" -type f -path '*/mlruns/[0-9]*/*/meta.yaml' ! -path '*/mlruns/[0-9]*/meta.yaml' 2>/dev/null
  )

done

if [ "${#seen[@]}" -eq 0 ]; then
  echo "No run-level meta.yaml found under provided roots." >&2
  exit 2
fi
