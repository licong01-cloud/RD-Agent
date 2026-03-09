"""
SOTA Session Editor — 手动修改 session pickle 中 Loop 的 decision 状态

功能：
1. --list: 列出所有 Loop 的 decision 状态、假设摘要和关键指标
2. --accept N: 将 Loop N 标记为 ACCEPTED (decision=True)
3. --reject N: 将 Loop N 标记为 REJECTED (decision=False)

使用场景：
- 程序逻辑因 SOTA 阈值限制未能自动 ACCEPT 某个优秀 Loop
- 人工审查后决定采纳/拒绝特定 Loop 的因子

执行环境：必须在 WSL conda rdagent-gpu 中执行（pickle 由 WSL Python 序列化）

Usage:
    # 列出所有 Loop 的 decision 状态
    python scripts/sota_session_editor.py <task_id> --list

    # 将 Loop 7 标记为 ACCEPTED
    python scripts/sota_session_editor.py <task_id> --accept 7

    # 将 Loop 3 标记为 REJECTED
    python scripts/sota_session_editor.py <task_id> --reject 3

    # 自定义 log 目录
    python scripts/sota_session_editor.py <task_id> --accept 7 --log-dir /path/to/logs
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path, PurePath


# ---------------------------------------------------------------------------
# Cross-platform pickle compatibility (same pattern as rdagent_task_analyzer.py)
# ---------------------------------------------------------------------------
class _CompatUnpickler(pickle.Unpickler):
    """Handle pathlib cross-platform deserialization."""

    def find_class(self, module: str, name: str):
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return Path
        if module == "pathlib" and name in {"PurePosixPath", "PureWindowsPath"}:
            return PurePath
        return super().find_class(module, name)


def safe_pickle_load(path: Path):
    """Load a pickle file with compatible unpickler; return None on failure."""
    try:
        with path.open("rb") as f:
            return _CompatUnpickler(f).load()
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------
def load_latest_session(log_dir: Path, task_id: str):
    """Load the latest session pickle from __session__/ directory."""
    session_dir = log_dir / task_id / "__session__"
    if not session_dir.exists():
        print(f"[ERROR] Session directory not found: {session_dir}")
        return None, None

    # Get all loop directories, sorted by loop_id descending
    loop_dirs = []
    for d in session_dir.iterdir():
        if d.is_dir():
            try:
                loop_id = int(d.name)
                loop_dirs.append((loop_id, d))
            except ValueError:
                continue

    loop_dirs.sort(key=lambda x: x[0], reverse=True)

    if not loop_dirs:
        print(f"[ERROR] No loop directories found in {session_dir}")
        return None, None

    # Try each loop directory, starting from latest
    for loop_id, loop_dir in loop_dirs:
        # Prefer 3_feedback (contains complete trace with all decisions)
        for step_file in ["3_feedback", "2_running", "1_coding", "0_direct_exp_gen"]:
            pkl_path = loop_dir / step_file
            if pkl_path.exists():
                session = safe_pickle_load(pkl_path)
                if session is not None:
                    print(f"[OK] Loaded session from loop {loop_id}/{step_file}")
                    return session, pkl_path

    print(f"[ERROR] Could not load any session pickle from {session_dir}")
    return None, None


def get_trace(session):
    """Extract trace from session object."""
    if hasattr(session, "trace"):
        return session.trace
    # Some session types wrap trace differently
    for attr in ["_trace", "t"]:
        if hasattr(session, attr):
            return getattr(session, attr)
    return None


# ---------------------------------------------------------------------------
# List command
# ---------------------------------------------------------------------------
def cmd_list(session):
    """List all loops with their decision status, hypothesis summary, and metrics."""
    trace = get_trace(session)
    if trace is None:
        print("[ERROR] Cannot find trace in session object")
        return False

    hist = trace.hist
    if not hist:
        print("[INFO] trace.hist is empty — no loops recorded")
        return True

    print(f"\n{'='*80}")
    print(f"  SOTA Session Editor — Loop Decision List ({len(hist)} loops)")
    print(f"{'='*80}\n")

    # Header
    print(f"{'Loop':>5} | {'Decision':>10} | {'IC':>8} | {'AnnRet':>8} | {'MaxDD':>8} | Hypothesis")
    print(f"{'-'*5:>5}-+-{'-'*10:>10}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*8:>8}-+{'-'*40}")

    for i, (exp, feedback) in enumerate(hist):
        decision_str = "ACCEPTED" if feedback.decision else "rejected"

        # Extract hypothesis summary
        hyp_text = ""
        if hasattr(exp, "hypothesis") and exp.hypothesis:
            h = exp.hypothesis
            action = getattr(h, "action", "?")
            hyp_raw = getattr(h, "hypothesis", "")
            hyp_text = f"[{action}] {hyp_raw[:50]}"

        # Extract metrics from exp.result
        ic_str = "—"
        ret_str = "—"
        dd_str = "—"
        if hasattr(exp, "result") and exp.result is not None:
            try:
                result = exp.result
                ic_val = result.get("IC") if hasattr(result, "get") else result.loc.get("IC") if hasattr(result, "loc") else None
                if ic_val is None and hasattr(result, "loc"):
                    try:
                        ic_val = result.loc["IC"]
                    except (KeyError, TypeError):
                        pass
                if ic_val is not None:
                    ic_str = f"{float(ic_val):.4f}"

                ret_key = "1day.excess_return_with_cost.annualized_return"
                dd_key = "1day.excess_return_with_cost.max_drawdown"
                for key, target in [(ret_key, "ret"), (dd_key, "dd")]:
                    val = None
                    if hasattr(result, "get"):
                        val = result.get(key)
                    elif hasattr(result, "loc"):
                        try:
                            val = result.loc[key]
                        except (KeyError, TypeError):
                            pass
                    if val is not None:
                        if target == "ret":
                            ret_str = f"{float(val)*100:.1f}%"
                        else:
                            dd_str = f"{float(val)*100:.1f}%"
            except Exception:
                pass

        print(f"{i:>5} | {decision_str:>10} | {ic_str:>8} | {ret_str:>8} | {dd_str:>8} | {hyp_text}")

    # Summary
    accepted = sum(1 for _, fb in hist if fb.decision)
    print(f"\n  Total: {len(hist)} loops, {accepted} ACCEPTED, {len(hist) - accepted} rejected")

    # Show SOTA
    sota_hyp, sota_exp = trace.get_sota_hypothesis_and_experiment()
    if sota_hyp is not None:
        # Find which loop is SOTA (last accepted)
        for i in range(len(hist) - 1, -1, -1):
            if hist[i][1].decision:
                print(f"  Current SOTA: Loop {i}")
                break
    else:
        print("  Current SOTA: None (no accepted loops)")

    print()
    return True


# ---------------------------------------------------------------------------
# Accept / Reject commands
# ---------------------------------------------------------------------------
def cmd_modify(session, pkl_path: Path, loop_idx: int, new_decision: bool):
    """Modify the decision of a specific loop."""
    trace = get_trace(session)
    if trace is None:
        print("[ERROR] Cannot find trace in session object")
        return False

    hist = trace.hist
    if loop_idx < 0 or loop_idx >= len(hist):
        print(f"[ERROR] Loop index {loop_idx} out of range (0-{len(hist)-1})")
        return False

    exp, feedback = hist[loop_idx]
    old_decision = feedback.decision
    action_str = "ACCEPT" if new_decision else "REJECT"

    if old_decision == new_decision:
        print(f"[INFO] Loop {loop_idx} is already {'ACCEPTED' if new_decision else 'REJECTED'}, no change needed")
        return True

    # Show what we're about to change
    hyp_text = ""
    if hasattr(exp, "hypothesis") and exp.hypothesis:
        h = exp.hypothesis
        action = getattr(h, "action", "?")
        hyp_raw = getattr(h, "hypothesis", "")
        hyp_text = f"[{action}] {hyp_raw[:80]}"

    print(f"\n  Modifying Loop {loop_idx}:")
    print(f"    Hypothesis: {hyp_text}")
    print(f"    Decision:   {'ACCEPTED' if old_decision else 'rejected'} → {'ACCEPTED' if new_decision else 'rejected'}")

    # Create backup
    bak_path = pkl_path.with_suffix(pkl_path.suffix + ".bak")
    shutil.copy2(pkl_path, bak_path)
    print(f"    Backup:     {bak_path}")

    # Modify decision — feedback is a mutable object, tuple holds reference
    feedback.decision = new_decision

    # Save back
    with pkl_path.open("wb") as f:
        pickle.dump(session, f)
    print(f"    Saved:      {pkl_path}")

    # Verify
    verify_session = safe_pickle_load(pkl_path)
    if verify_session is not None:
        verify_trace = get_trace(verify_session)
        if verify_trace and len(verify_trace.hist) > loop_idx:
            verified = verify_trace.hist[loop_idx][1].decision
            if verified == new_decision:
                print(f"    Verified:   Loop {loop_idx} decision = {'ACCEPTED' if verified else 'rejected'} ✓")
            else:
                print(f"    [WARN] Verification failed! Expected {new_decision}, got {verified}")
                return False
    else:
        print("    [WARN] Could not verify — reload failed")

    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SOTA Session Editor — Manually accept/reject loop decisions in session pickle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2026-03-07_05-25-05-912419 --list
  %(prog)s 2026-03-07_05-25-05-912419 --accept 7
  %(prog)s 2026-03-07_05-25-05-912419 --reject 3
        """,
    )
    parser.add_argument("task_id", help="Task ID (directory name under log/)")
    parser.add_argument("--list", action="store_true", help="List all loops with decision status")
    parser.add_argument("--accept", type=int, metavar="N", help="Accept loop N (set decision=True)")
    parser.add_argument("--reject", type=int, metavar="N", help="Reject loop N (set decision=False)")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/mnt/f/Dev/RD-Agent-main/log",
        help="Log directory (default: /mnt/f/Dev/RD-Agent-main/log)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.list and args.accept is None and args.reject is None:
        parser.error("Must specify --list, --accept N, or --reject N")

    if args.accept is not None and args.reject is not None:
        parser.error("Cannot specify both --accept and --reject")

    log_dir = Path(args.log_dir)
    task_dir = log_dir / args.task_id
    if not task_dir.exists():
        print(f"[ERROR] Task directory not found: {task_dir}")
        sys.exit(1)

    # Load session
    session, pkl_path = load_latest_session(log_dir, args.task_id)
    if session is None:
        sys.exit(1)

    # Execute command
    if args.list:
        success = cmd_list(session)
    elif args.accept is not None:
        success = cmd_modify(session, pkl_path, args.accept, new_decision=True)
    elif args.reject is not None:
        success = cmd_modify(session, pkl_path, args.reject, new_decision=False)
    else:
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
