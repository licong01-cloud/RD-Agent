#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="/mnt/c/Users/lc999/RD-Agent-main/git_ignore_folder/RD-Agent_workspace"
ROOT="${ROOT:-$ROOT_DEFAULT}"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/inspect_ws_mlruns.sh <workspace_id> [workspace_id ...]"
  echo "Env: ROOT=/path/to/RD-Agent_workspace (default: $ROOT_DEFAULT)"
  exit 1
fi

for ws in "$@"; do
  echo "===================="
  echo "WS=$ws"

  ws_dir="$ROOT/$ws"
  if [ ! -d "$ws_dir" ]; then
    echo "NO workspace dir: $ws_dir"
    continue
  fi

  conf="$ws_dir/conf_combined_factors.yaml"
  res="$ws_dir/qlib_res.csv"

  if [ -f "$conf" ]; then
    echo "[static_path]"
    grep -n -E '^\s*static_path\s*:' "$conf" || true
  else
    echo "[static_path] conf_combined_factors.yaml NOT FOUND"
  fi

  if [ -f "$res" ]; then
    echo "[metrics]"
    awk -F, '
      $1=="1day.excess_return_with_cost.annualized_return"{print "ann=" $2}
      $1=="1day.excess_return_with_cost.information_ratio"{print "ir=" $2}
      $1=="IC"{print "IC=" $2}
      $1=="Rank IC"{print "RankIC=" $2}
      $1=="1day.excess_return_with_cost.max_drawdown"{print "mdd=" $2}
    ' "$res" || true
  else
    echo "[metrics] qlib_res.csv NOT FOUND"
  fi

  if [ ! -d "$ws_dir/mlruns" ]; then
    echo "[mlruns] NOT FOUND"
    continue
  fi

  exp_id="$(ls -1 "$ws_dir/mlruns" 2>/dev/null | grep -E '^[0-9]+$' | head -n 1 || true)"
  if [ -z "$exp_id" ]; then
    echo "[mlruns] no numeric exp_id under $ws_dir/mlruns"
    ls -lah "$ws_dir/mlruns" 2>/dev/null | head -n 50 || true
    continue
  fi
  echo "[mlruns] exp_id=$exp_id"

  exp_dir="$ws_dir/mlruns/$exp_id"
  echo "[mlruns] exp_dir listing (top 50)"
  ls -lah "$exp_dir" 2>/dev/null | head -n 50 || true

  latest_run_meta="$(
    find "$exp_dir" -maxdepth 2 -type f -name meta.yaml \
      ! -path "$exp_dir/meta.yaml" \
      -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | awk '{print $2}'
  )"

  if [ -z "$latest_run_meta" ]; then
    echo "[mlruns] NO run meta.yaml found (this exp has NO runs under workspace mlruns)"
    echo "        Likely MLflow tracking_uri points elsewhere (global mlruns) or runs were cleaned."
    continue
  fi

  echo "[latest run meta]"
  echo "meta=$latest_run_meta"
  egrep 'run_id:|artifact_uri:|end_time:|start_time:|status:' "$latest_run_meta" || true

  run_dir="$(dirname "$latest_run_meta")"
  echo "[artifacts list (top 50)]"
  ls -1 "$run_dir/artifacts" 2>/dev/null | head -n 50 || echo "NO artifacts dir"
done
