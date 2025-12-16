#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${1:-}"
WS_A="${2:-}"
WS_B="${3:-}"

if [[ -z "$LOG_DIR" ]]; then
  echo "Usage:"
  echo "  bash scripts/qlib_ws_compare.sh <LOG_DIR> [WS_A] [WS_B]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/qlib_ws_compare.sh /mnt/c/Users/lc999/RD-Agent-main/log/2025-12-15_07-50-56-750327"
  echo "  bash scripts/qlib_ws_compare.sh /mnt/c/.../log/... /mnt/c/.../RD-Agent_workspace/xxx /mnt/c/.../RD-Agent_workspace/yyy"
  exit 1
fi

if [[ ! -d "$LOG_DIR" ]]; then
  echo "ERROR: LOG_DIR not found: $LOG_DIR"
  exit 2
fi

echo "==== [1] Scan LOG_DIR for qrun workspaces ===="
echo "LOG_DIR=$LOG_DIR"

# Collect candidate log files (cap to avoid huge scans)
mapfile -t LOG_FILES < <(find "$LOG_DIR" -type f 2>/dev/null | head -n 5000)
if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
  echo "No files found under LOG_DIR."
  exit 3
fi

echo ""
echo "---- Found qrun entries (workspace + conf + log_file) ----"

tmp="/tmp/_qlib_ws_pairs_$$.txt"
: > "$tmp"

tmp_ws="/tmp/_qlib_ws_only_$$.txt"
: > "$tmp_ws"

tmp_conf="/tmp/_qlib_conf_only_$$.txt"
: > "$tmp_conf"

for f in "${LOG_FILES[@]}"; do
  # conf name(s) e.g. conf_baseline.yaml / conf_combined_factors_dynamic.yaml
  # Be tolerant: logs may wrap qrun entry in shell scripts, quotes, etc.
  confs=$(grep -aoE "conf_[A-Za-z0-9_\.\-]+\.ya?ml" "$f" 2>/dev/null | tail -n 50 | sed -E 's/^"+//; s/"+$//; s/[[:space:]]+$//' || true)

  # workspace path(s): do NOT rely on table formatting; match the workspace path pattern directly.
  wss=$(grep -aoE "[^[:space:]]*RD-Agent_workspace/[0-9a-f]{32}" "$f" 2>/dev/null | tail -n 200 | sed -E 's/^"+//; s/"+$//; s/[[:space:]]+$//' || true)

  if [[ -n "$wss" ]]; then
    printf "%s\n" "$wss" >> "$tmp_ws"
  fi
  if [[ -n "$confs" ]]; then
    printf "%s\n" "$confs" >> "$tmp_conf"
  fi

  if [[ -n "$wss" ]]; then
    if [[ -n "$confs" ]]; then
      while IFS= read -r c; do
        while IFS= read -r w; do
          if [[ -n "$c" && -n "$w" ]]; then
            echo -e "${w}\t${c}\t${f}" >> "$tmp"
          fi
        done <<< "$wss"
      done <<< "$confs"
    else
      while IFS= read -r w; do
        if [[ -n "$w" ]]; then
          echo -e "${w}\t(unknown_conf)\t${f}" >> "$tmp"
        fi
      done <<< "$wss"
    fi
  fi
done

if [[ ! -s "$tmp" ]]; then
  echo "No (workspace, conf) pairs found in the same log file."
  echo "This can happen when workspace paths and qrun command lines are logged in different files."
else
  echo "(Top 80) [count] workspace\tconf"
  awk -F'\t' '{print $1 "\t" $2}' "$tmp" | sort | uniq -c | sort -nr | head -n 80
fi

echo ""
echo "---- All workspaces found in logs (Top 80) ----"
if [[ -s "$tmp_ws" ]]; then
  sort -u "$tmp_ws" | head -n 80
else
  echo "(none)"
fi

echo ""
echo "---- All qrun conf names found in logs ----"
if [[ -s "$tmp_conf" ]]; then
  sort -u "$tmp_conf"
else
  echo "(none)"
fi

echo ""
echo "==== [2] Compare two workspaces ===="
if [[ -z "$WS_A" || -z "$WS_B" ]]; then
  echo "You didn't pass WS_A/WS_B. Copy two workspace paths from above summary and rerun:"
  echo "  bash scripts/qlib_ws_compare.sh \"$LOG_DIR\" <WS_A> <WS_B>"
  exit 0
fi

if [[ ! -d "$WS_A" ]]; then
  echo "ERROR: WS_A not found: $WS_A"
  exit 4
fi
if [[ ! -d "$WS_B" ]]; then
  echo "ERROR: WS_B not found: $WS_B"
  exit 5
fi

echo "WS_A=$WS_A"
echo "WS_B=$WS_B"
echo ""

show_file () {
  local fp="$1"
  if [[ -f "$fp" ]]; then
    stat -c "%y %s %n" "$fp" 2>/dev/null || ls -l "$fp"
  else
    echo "MISSING $fp"
  fi
}

hash_file () {
  local fp="$1"
  if [[ -f "$fp" ]]; then
    sha256sum "$fp" | awk '{print $1 "  " $2}'
  else
    echo "MISSING $fp"
  fi
}

echo "---- [2.1] Key files stat ----"
for name in \
  conf.yaml \
  conf_baseline.yaml \
  conf_combined_factors_dynamic.yaml \
  conf_combined_factors.yaml \
  qlib_res.csv \
  ret.pkl \
  combined_factors_df.parquet
do
  echo "[A] $name"; show_file "$WS_A/$name"
  echo "[B] $name"; show_file "$WS_B/$name"
  echo ""
done

echo "---- [2.2] Hash key artifacts (if exist) ----"
for name in \
  conf.yaml \
  conf_baseline.yaml \
  conf_combined_factors_dynamic.yaml \
  conf_combined_factors.yaml \
  qlib_res.csv \
  ret.pkl \
  combined_factors_df.parquet
do
  echo "[A] $name"; hash_file "$WS_A/$name"
  echo "[B] $name"; hash_file "$WS_B/$name"
  echo ""
done

echo "---- [2.3] Parquet quick compare (shape/cols/head-hash) ----"
WS_A="$WS_A" WS_B="$WS_B" python3 - <<'PY' || true
import os
import hashlib
import pandas as pd

ws_a = os.environ.get("WS_A")
ws_b = os.environ.get("WS_B")
pa = os.path.join(ws_a, "combined_factors_df.parquet")
pb = os.path.join(ws_b, "combined_factors_df.parquet")

def head_sig(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    cols = list(map(str, df.columns))
    head = df.head(5)
    h = hashlib.sha256(pd.util.hash_pandas_object(head, index=True).values.tobytes()).hexdigest()
    return {
        "shape": df.shape,
        "first_cols": cols[:30],
        "head_hash": h,
    }

print("WS_A parquet:", pa)
print("WS_B parquet:", pb)
print("A:", head_sig(pa))
print("B:", head_sig(pb))
PY

echo ""
echo "==== Done ===="
