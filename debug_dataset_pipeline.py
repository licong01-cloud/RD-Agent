import qlib
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158


PROVIDER_URI = "/mnt/c/Users/lc999/NewAIstock/AIstock/qlib_bin/qlib_bin_20251209"


def build_handler(learn_processors):
    """Build an Alpha158 handler with the same核心配置 as baseline YAML.

    learn_processors is a list of processor dicts, so we can test不同清洗组合.
    """
    handler_kwargs = {
        "start_time": "2010-11-01",
        "end_time": "2025-12-31",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2014-12-31",
        "instruments": "all",  # 与 market: all 对应
        "infer_processors": [
            {
                "class": "FilterCol",
                "kwargs": {
                    "fields_group": "feature",
                    # 与 conf_baseline_factors_model.yaml 中的因子子集保持一致
                    "col_list": [
                        "RESI5",
                        "WVMA5",
                        "RSQR5",
                        "KLEN",
                        "RSQR10",
                        "CORR5",
                        "CORD5",
                        "CORR10",
                        "ROC60",
                        "RESI10",
                        "VSTD5",
                        "RSQR60",
                        "CORR60",
                        "WVMA60",
                        "STD5",
                        "RSQR20",
                        "CORD60",
                        "CORD10",
                        "CORR20",
                        "KLOW",
                    ],
                },
            },
            {
                "class": "RobustZScoreNorm",
                "kwargs": {"fields_group": "feature", "clip_outlier": True},
            },
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": learn_processors,
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }
    return Alpha158(**handler_kwargs)


def inspect_dataset(tag: str, learn_processors):
    print("\n" + "=" * 80)
    print(f"[配置 {tag}] learn_processors = {learn_processors}")
    handler = build_handler(learn_processors=learn_processors)

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2016-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2020-12-31"),
            "test": ("2021-01-01", "2025-12-31"),
        },
    )

    # 1) 看原始 handler 在各段的样本量（不区分 feature/label，粗略参考）
    for seg_name, seg in dataset.segments.items():
        raw_df = handler.fetch(selector=seg)
        print(f"{tag} - raw {seg_name} shape:", raw_df.shape)

    # 2) 简单查看 label 的截面分布（选取若干日期做示例）
    try:
        label_df = handler.fetch(selector=dataset.segments["train"], col_set=["label"])
        label_col = label_df.columns[0] if hasattr(label_df.columns, "__len__") else None
        if label_col is not None:
            # 取若干代表性日期
            sample_dates = sorted(set(label_df.index.get_level_values("datetime")))[:5]
            for dt in sample_dates:
                cross = label_df.xs(dt, level="datetime")[label_col]
                print(
                    f"{tag} - label cross-section @ {dt.date()}: n={cross.shape[0]}, "
                    f"mean={cross.mean():.6f}, std={cross.std():.6f}, min={cross.min():.6f}, max={cross.max():.6f}"
                )
    except Exception as e:
        print(f"{tag} - label distribution inspection ERROR: {repr(e)}")

    # 3) 看经过 prepare("train"/"valid"/"test") 之后 feature+label 的样本量
    # DatasetH.prepare 在当前配置下直接返回 DataFrame，因此直接查看其 shape 和索引结构即可。
    for seg_name in ["train", "valid", "test"]:
        try:
            df_prepared = dataset.prepare(seg_name, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            n_levels = getattr(df_prepared.index, "nlevels", 1)
            print(
                f"{tag} - prepared {seg_name} size: {df_prepared.shape}, "
                f"index_levels={n_levels}"
            )
        except Exception as e:
            print(f"{tag} - prepared {seg_name} ERROR:", repr(e))


def main():
    print("初始化 Qlib...")
    qlib.init(provider_uri=PROVIDER_URI, region="cn")

    # A. 不带任何 learn_processors
    inspect_dataset(tag="A: no learn_processors", learn_processors=[])

    # B. 只带 DropnaLabel
    inspect_dataset(
        tag="B: DropnaLabel only",
        learn_processors=[{"class": "DropnaLabel"}],
    )

    # C. DropnaLabel + CSRankNorm（与 baseline 模型 YAML 一致）
    inspect_dataset(
        tag="C: DropnaLabel + CSRankNorm",
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
    )


if __name__ == "__main__":
    main()
