"""Patch qlib exchange.py: add ensure_data_for_day + modify get_quote_from_qlib"""
import shutil
from pathlib import Path

EXCHANGE = Path(
    "/home/lc999/miniconda3/envs/rdagent-gpu/lib/python3.10/"
    "site-packages/qlib/backtest/exchange.py"
)

# Backup
bak = EXCHANGE.with_suffix(".py.bak_20260409")
if not bak.exists():
    shutil.copy2(EXCHANGE, bak)
    print(f"Backed up to {bak}")

src = EXCHANGE.read_text(encoding="utf-8")

# ── 1. Add imports: gc, os at top ──
if "import gc" not in src:
    src = src.replace(
        "import numpy as np\n",
        "import gc\nimport os\nimport numpy as np\n",
    )
    print("Added gc, os imports")

# ── 2. Add _MINUTE_BATCH_DAYS constant before class Exchange ──
BATCH_CONST = (
    "\n# ── Minute-frequency batch loading (memory optimization) ──\n"
    "# Load N trading days at a time instead of full backtest period (121GB -> 200MB)\n"
    "# Set QLIB_MINUTE_FULL_LOAD=1 to disable (for 128GB+ machines)\n"
    "# Set QLIB_MINUTE_BATCH_DAYS=N to change batch size (default 20)\n"
    '_MINUTE_BATCH_DAYS = int(os.environ.get("QLIB_MINUTE_BATCH_DAYS", "20"))\n'
    '_MINUTE_FULL_LOAD = os.environ.get("QLIB_MINUTE_FULL_LOAD", "") == "1"\n'
    "\n"
)

if "_MINUTE_BATCH_DAYS" not in src:
    src = src.replace("\nclass Exchange:", BATCH_CONST + "\nclass Exchange:")
    print("Added _MINUTE_BATCH_DAYS constant")

# ── 3. Replace get_quote_from_qlib() ──
OLD_GQFQ = (
    "    def get_quote_from_qlib(self) -> None:\n"
    "        # get stock data from qlib\n"
    "        if len(self.codes) == 0:\n"
    "            self.codes = D.instruments()\n"
    "        self.quote_df = D.features(\n"
    "            self.codes,\n"
    "            self.all_fields,\n"
    "            self.start_time,\n"
    "            self.end_time,\n"
    "            freq=self.freq,\n"
    "            disk_cache=True,\n"
    "        )\n"
    "        self.quote_df.columns = self.all_fields"
)

NEW_GQFQ = (
    "    def get_quote_from_qlib(self) -> None:\n"
    "        # get stock data from qlib\n"
    "        if len(self.codes) == 0:\n"
    "            self.codes = D.instruments()\n"
    "\n"
    "        # Minute batch loading: only load first N days, reload per-day later\n"
    '        if self.freq in ("1min", "5min") and not _MINUTE_FULL_LOAD:\n'
    '            cal = D.calendar(start_time=self.start_time, end_time=self.end_time, freq="day")\n'
    "            if len(cal) > 0:\n"
    "                batch_end_day = cal[min(_MINUTE_BATCH_DAYS - 1, len(cal) - 1)]\n"
    "                batch_end_time = pd.Timestamp(batch_end_day).normalize() + pd.Timedelta(\n"
    "                    hours=23, minutes=59, seconds=59\n"
    "                )\n"
    "                self.quote_df = D.features(\n"
    "                    self.codes, self.all_fields,\n"
    "                    self.start_time, batch_end_time,\n"
    "                    freq=self.freq, disk_cache=False,\n"
    "                )\n"
    "                self.quote_df.columns = self.all_fields\n"
    "                self._loaded_start = pd.Timestamp(self.start_time).normalize()\n"
    "                self._loaded_end = pd.Timestamp(batch_end_day).normalize()\n"
    "                return\n"
    "            # empty calendar fallthrough to full load\n"
    "\n"
    "        self.quote_df = D.features(\n"
    "            self.codes,\n"
    "            self.all_fields,\n"
    "            self.start_time,\n"
    "            self.end_time,\n"
    "            freq=self.freq,\n"
    "            disk_cache=True,\n"
    "        )\n"
    "        self.quote_df.columns = self.all_fields"
)

if "Minute batch loading" not in src:
    if OLD_GQFQ in src:
        src = src.replace(OLD_GQFQ, NEW_GQFQ)
        print("Replaced get_quote_from_qlib")
    else:
        print("ERROR: Could not find get_quote_from_qlib pattern!")
        idx = src.find("def get_quote_from_qlib")
        print(repr(src[idx:idx+400]))
        raise SystemExit(1)
else:
    print("get_quote_from_qlib already patched")

# ── 4. Add ensure_data_for_day() method ──
ENSURE_METHOD = '''
    def ensure_data_for_day(self, trade_start_time, trade_end_time) -> None:
        """Reload minute-freq Exchange data for the batch containing trade_start_time.

        Called by qlib/backtest/backtest.py collect_data_loop before
        strategy.generate_trade_decision(). Day-freq Exchange is not affected.

        Caching: skips reload if trade_start_time falls within [_loaded_start, _loaded_end].
        """
        if self.freq not in ("1min", "5min") or _MINUTE_FULL_LOAD:
            return

        day_key = pd.Timestamp(trade_start_time).normalize()
        loaded_start = getattr(self, "_loaded_start", None)
        loaded_end = getattr(self, "_loaded_end", None)
        if loaded_start is not None and loaded_start <= day_key <= loaded_end:
            return  # current day is within loaded batch

        # Determine new batch boundary
        backtest_end = getattr(self, "end_time", trade_end_time)
        cal = D.calendar(start_time=trade_start_time, end_time=backtest_end, freq="day")
        if len(cal) == 0:
            return
        batch_end_day = cal[min(_MINUTE_BATCH_DAYS - 1, len(cal) - 1)]
        batch_end_time = pd.Timestamp(batch_end_day).normalize() + pd.Timedelta(
            hours=23, minutes=59, seconds=59
        )

        # Cleanup old data
        old_quote = getattr(self, "quote", None)
        old_df = getattr(self, "quote_df", None)
        if hasattr(self, "quote"):
            self.quote = None
        if old_quote is not None:
            if hasattr(old_quote, "data") and isinstance(old_quote.data, dict):
                old_quote.data.clear()
        del old_quote, old_df
        gc.collect()

        # Load new batch
        self.quote_df = D.features(
            self.codes, self.all_fields,
            trade_start_time, batch_end_time,
            freq=self.freq, disk_cache=False,
        )
        self.quote_df.columns = self.all_fields

        # Recompute limit flags
        self._update_limit(self.limit_threshold)

        # Re-merge extra_quote if present
        if self.extra_quote is not None:
            self.quote_df = pd.concat(
                [self.quote_df, self.extra_quote], sort=False, axis=0
            )

        # Rebuild high-performance quote
        self.quote = self.quote_cls(self.quote_df, self.freq)
        self._loaded_start = day_key
        self._loaded_end = pd.Timestamp(batch_end_day).normalize()

'''

if "def ensure_data_for_day" not in src:
    insert_marker = "    def check_stock_suspended("
    if insert_marker in src:
        src = src.replace(insert_marker, ENSURE_METHOD + "    def check_stock_suspended(")
        print("Added ensure_data_for_day method")
    else:
        print("ERROR: Could not find insertion point")
        raise SystemExit(1)
else:
    print("ensure_data_for_day already exists")

EXCHANGE.write_text(src, encoding="utf-8")
print(f"Done! exchange.py: {len(src)} chars")
