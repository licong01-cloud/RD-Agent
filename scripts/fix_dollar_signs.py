"""Fix: re-patch the trade_w_adj_price check with proper $-escaping."""
from pathlib import Path

EXCHANGE = Path(
    "/home/lc999/miniconda3/envs/rdagent-gpu/lib/python3.10/"
    "site-packages/qlib/backtest/exchange.py"
)
src = EXCHANGE.read_text(encoding="utf-8")

# The shell ate the $ signs. Find and fix.
BAD = '''                # trade_w_adj_price check (same as full-load path)
                if (self.quote_df[""].isna() & ~self.quote_df[""].isna()).any():'''

GOOD = '''                # trade_w_adj_price check (same as full-load path)
                if (self.quote_df["$factor"].isna() & ~self.quote_df["$close"].isna()).any():'''

if BAD in src:
    src = src.replace(BAD, GOOD)
    EXCHANGE.write_text(src, encoding="utf-8")
    print("Fixed $factor/$close references")
else:
    # Check if already correct
    if '"$factor"' in src and "trade_w_adj_price check" in src:
        print("Already correct, no fix needed")
    else:
        print("ERROR: Could not find pattern to fix")
        # Show the area
        idx = src.find("trade_w_adj_price check")
        if idx >= 0:
            print(repr(src[idx:idx+200]))
        raise SystemExit(1)
