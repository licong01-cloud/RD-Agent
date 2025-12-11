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
