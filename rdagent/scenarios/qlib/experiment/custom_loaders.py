from typing import Optional

import pandas as pd
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.loader import StaticDataLoader


class IgnoreInstrumentsStaticDataLoader(StaticDataLoader):
    """A StaticDataLoader variant that ignores the `instruments` argument.

    Kept for compatibility; not used in the preferred plan-2 pipeline.
    """

    def load(  # type: ignore[override]
        self,
        instruments: Optional[object] = None,
        start_time: Optional[object] = None,
        end_time: Optional[object] = None,
    ) -> pd.DataFrame:
        # Ignore `instruments` completely and rely on the underlying
        # factor table's MultiIndex alignment. Only apply time window
        # if provided, consistent with the base loader's behavior.
        return super().load(instruments=None, start_time=start_time, end_time=end_time)


class CombinedAlpha158StaticLoader:
    """Combine Alpha158DL with a static factor table into a single DataFrame.

    This loader implements the "plan 2" pipeline:

    - Alpha158DL is used to generate labels and base price/volume factors;
    - A StaticDataLoader backed by a MultiIndex(datetime, instrument)
      factor table (e.g. combined_static_factors.parquet) provides
      additional static factors;
    - The two DataFrames are aligned on index and concatenated along
      columns, with an outer join and final sort_index().

    Qlib will see only this single loader in the dataset handler, which
    greatly reduces the risk of index-level mismatch inside
    NestedDataLoader.
    """

    def __init__(
        self,
        alpha158_config: dict,
        static_path: str,
        join: str = "outer",
    ) -> None:
        # Alpha158 configuration is passed exactly as in the original
        # YAML (label/feature lists etc.).
        self.alpha_loader = Alpha158DL(config=alpha158_config)
        # Static factors come from a single pre-merged table, typically
        # combined_static_factors.parquet.
        self.static_loader = StaticDataLoader(config=static_path)
        self.join = join

    def load(  # type: ignore[override]
        self,
        instruments: Optional[object] = None,
        start_time: Optional[object] = None,
        end_time: Optional[object] = None,
    ) -> pd.DataFrame:
        # 1) Alpha158 base loader：尊重 instruments/start/end
        df_alpha = self.alpha_loader.load(
            instruments=instruments, start_time=start_time, end_time=end_time
        )

        # 2) 静态因子表：忽略 instruments，只按时间窗口裁剪
        df_static = self.static_loader.load(
            instruments=None, start_time=start_time, end_time=end_time
        )

        # 3) 统一列索引结构，确保合并后仍然是 MultiIndex 列，
        #    顶层包含 'label' / 'feature' 等分组，便于 DataHandler 和 Processor 使用。
        if df_alpha is not None and not isinstance(df_alpha.columns, pd.MultiIndex):
            df_alpha = df_alpha.copy()
            df_alpha.columns = pd.MultiIndex.from_product([
                ["feature"], df_alpha.columns.astype(str)
            ])

        if df_static is not None and not isinstance(df_static.columns, pd.MultiIndex):
            df_static = df_static.copy()
            df_static.columns = pd.MultiIndex.from_product([
                ["feature"], df_static.columns.astype(str)
            ])

        # 4) 对齐并按列拼接
        if df_alpha is None:
            merged = df_static
        elif df_static is None:
            merged = df_alpha
        else:
            merged = pd.concat([df_alpha, df_static], axis=1, join=self.join)

        # 5) 保证索引为 MultiIndex(datetime, instrument) 且已排序，
        #    以便 pandas 在区间切片/merge 时不会抛出 UnsortedIndexError。
        if not isinstance(merged.index, pd.MultiIndex):
            merged = merged.copy()
            if "datetime" in merged.index.names and "instrument" in merged.index.names:
                merged = merged.sort_index()
            else:
                # 最保守退路：尝试按照列构造 MultiIndex
                if {"datetime", "instrument"}.issubset(merged.columns):
                    merged = merged.set_index(["datetime", "instrument"]).sort_index()

        else:
            merged = merged.sort_index()

        return merged


class CombinedAlpha158DynamicFactorsLoader:
    def __init__(
        self,
        alpha158_config: dict,
        dynamic_path: str,
        join: str = "left",
        min_dynamic_non_nan_ratio: float = 0.01,
        min_instrument_overlap_ratio: float = 0.8,
        enforce_instrument_format: bool = True,
    ) -> None:
        self.alpha_loader = Alpha158DL(config=alpha158_config)
        self.dynamic_loader = StaticDataLoader(config=dynamic_path)
        self.join = join
        self.min_dynamic_non_nan_ratio = float(min_dynamic_non_nan_ratio)
        self.min_instrument_overlap_ratio = float(min_instrument_overlap_ratio)
        self.enforce_instrument_format = bool(enforce_instrument_format)

    @staticmethod
    def _ensure_datetime_instrument_index(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return df

        if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {"datetime", "instrument"}:
            if {"datetime", "instrument"}.issubset(df.columns):
                df = df.copy()
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.set_index(["datetime", "instrument"])

        if isinstance(df.index, pd.MultiIndex):
            names = list(df.index.names)
            if set(names) == {"datetime", "instrument"} and names != ["datetime", "instrument"]:
                df = df.swaplevel("datetime", "instrument")
            df = df.sort_index()

        return df

    @staticmethod
    def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return df

        if not isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = pd.MultiIndex.from_product([["feature"], df.columns.astype(str)])
            return df

        if df.columns.nlevels == 1:
            df = df.copy()
            df.columns = pd.MultiIndex.from_product([["feature"], df.columns.astype(str)])
            return df

        level0 = df.columns.get_level_values(0).astype(str)
        if not level0.isin(["feature", "label"]).all():
            df = df.copy()
            df.columns = pd.MultiIndex.from_product([["feature"], df.columns.get_level_values(-1).astype(str)])
        return df

    @staticmethod
    def _validate_instrument_format(instruments: pd.Index) -> None:
        if instruments is None:
            return
        s = pd.Index(instruments.astype(str))
        ok = s.str.match(r"^\d{6}\.(SZ|SH)$")
        if not bool(ok.all()):
            bad = s[~ok][:20].tolist()
            raise ValueError(
                "Invalid instrument format detected in dynamic factors. "
                "Expected like '000001.SZ' or '600000.SH'. "
                f"Examples: {bad}"
            )

    def load(  # type: ignore[override]
        self,
        instruments: Optional[object] = None,
        start_time: Optional[object] = None,
        end_time: Optional[object] = None,
    ) -> pd.DataFrame:
        df_alpha = self.alpha_loader.load(
            instruments=instruments, start_time=start_time, end_time=end_time
        )
        df_alpha = self._ensure_datetime_instrument_index(df_alpha)
        df_alpha = self._ensure_feature_columns(df_alpha)

        df_dynamic = self.dynamic_loader.load(instruments=None, start_time=start_time, end_time=end_time)
        df_dynamic = self._ensure_datetime_instrument_index(df_dynamic)
        df_dynamic = self._ensure_feature_columns(df_dynamic)

        if df_dynamic is None or df_dynamic.empty:
            raise ValueError(
                "Dynamic factors parquet is empty after loading/time slicing. "
                "Refusing to run backtest with only Alpha158 factors."
            )

        if not isinstance(df_dynamic.index, pd.MultiIndex) or set(df_dynamic.index.names) != {"datetime", "instrument"}:
            raise ValueError(
                "Dynamic factors parquet must be indexed by MultiIndex(datetime, instrument). "
                "Either write parquet with the correct MultiIndex, or include 'datetime' and 'instrument' columns."
            )

        dyn_instruments = df_dynamic.index.get_level_values("instrument")
        if self.enforce_instrument_format:
            self._validate_instrument_format(dyn_instruments)

        if df_alpha is not None and isinstance(df_alpha.index, pd.MultiIndex) and set(df_alpha.index.names) == {"datetime", "instrument"}:
            alpha_instruments = pd.Index(df_alpha.index.get_level_values("instrument").astype(str).unique())
            dyn_unique = pd.Index(dyn_instruments.astype(str).unique())
            if len(dyn_unique) > 0:
                overlap_ratio = float(dyn_unique.isin(alpha_instruments).mean())
                if overlap_ratio < self.min_instrument_overlap_ratio:
                    raise ValueError(
                        "Dynamic factors instruments have low overlap with provider/Alpha158 instruments. "
                        f"overlap_ratio={overlap_ratio:.4f} < min_instrument_overlap_ratio={self.min_instrument_overlap_ratio:.4f}. "
                        f"dyn_examples={dyn_unique[:10].tolist()} alpha_examples={alpha_instruments[:10].tolist()}"
                    )

        if df_alpha is None:
            merged = df_dynamic
        elif df_dynamic is None:
            merged = df_alpha
        else:
            if self.join in {"left", "right"}:
                merged = df_alpha.join(df_dynamic, how=self.join)
            else:
                # pandas.concat only supports join={'inner','outer'}
                merged = pd.concat([df_alpha, df_dynamic], axis=1, join=self.join)
        merged = merged.sort_index()

        dyn_cols = df_dynamic.columns
        if dyn_cols is not None and len(dyn_cols) > 0:
            dyn_non_nan_ratio = float(merged.loc[:, dyn_cols].notna().any(axis=1).mean())
            if dyn_non_nan_ratio < self.min_dynamic_non_nan_ratio:
                raise ValueError(
                    "Dynamic factors become nearly all-NaN after alignment with Alpha158 index. "
                    f"non_nan_row_ratio={dyn_non_nan_ratio:.6f} < min_dynamic_non_nan_ratio={self.min_dynamic_non_nan_ratio:.6f}. "
                    "This usually indicates index/instrument mismatch between parquet and provider."
                )

        return merged
