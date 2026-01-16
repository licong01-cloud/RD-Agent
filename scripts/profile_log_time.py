import argparse
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from rdagent.log.ui.utils import load_times_info


def fmt(td: timedelta) -> str:
    s = int(td.total_seconds())
    return str(timedelta(seconds=s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile RD-Agent log timing by parsing time_info entries")
    parser.add_argument(
        "--log_root",
        required=True,
        help="Path to a single run folder under ./log (e.g. ./log/2025-12-13_06-25-45-651001)",
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    if not log_root.exists():
        raise SystemExit(f"log_root does not exist: {log_root}")

    print(f"Log: {log_root}")
    print("=" * 80)

    # Important: load_times_info expects tags like `Loop_0.coding.time_info`.
    # Therefore, it must be called on the run root folder (the one that contains Loop_*).
    times_info = load_times_info(log_root)

    # Aggregate per loop
    loop_acc: dict[int, dict[str, timedelta]] = defaultdict(lambda: defaultdict(timedelta))
    for loop_id, loop_times in times_info.items():
        for step_name, step_time in loop_times.items():
            st = step_time.get("start_time")
            et = step_time.get("end_time")
            if st and et:
                loop_acc[int(loop_id)][step_name] += (et - st)

    # Print per-loop breakdown
    grand: dict[str, timedelta] = defaultdict(timedelta)
    for loop_id in sorted(loop_acc.keys()):
        acc = loop_acc[loop_id]
        print(f"Loop_{loop_id}", {k: fmt(v) for k, v in acc.items() if v.total_seconds()})
        for k, v in acc.items():
            grand[k] += v

    overall = sum(grand.values(), timedelta())
    breakdown = {
        k: f"{fmt(v)} ({(v.total_seconds() / overall.total_seconds()):.1%})"
        for k, v in grand.items()
        if overall.total_seconds() and v.total_seconds()
    }
    print("OVERALL", fmt(overall), breakdown)


if __name__ == "__main__":
    main()
