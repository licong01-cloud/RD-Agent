import importlib


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def main() -> None:
    libs = [
        "torch",      # PyTorch 深度学习
        "lightgbm",  # LightGBM GBDT
        "xgboost",   # XGBoost GBDT
        "catboost",  # CatBoost GBDT
        "sklearn",   # scikit-learn 经典 ML 模型
        "statsmodels",  # 统计建模/时间序列
        "tensorflow",   # TensorFlow（一般不会在本环境用，但可检测）
    ]

    print("=== 库安装状态 (Python packages) ===")
    for lib in libs:
        status = "INSTALLED" if check_module(lib) else "NOT INSTALLED"
        print(f"{lib:12s}: {status}")

    print("\n=== Qlib 常用模型类模块可导入情况 ===")
    qlib_models = {
        "LGBModel": "qlib.contrib.model.gbdt",
        "XGBModel": "qlib.contrib.model.xgboost",
        "CatBoostModel": "qlib.contrib.model.catboost",
        "LinearModel": "qlib.contrib.model.linear",
        "GBDTModel": "qlib.contrib.model.gbdt",
        "TabNetModel": "qlib.contrib.model.pytorch_tabnet",
        "GeneralPTNN": "qlib.contrib.model.pytorch_general_nn",
    }

    for name, module in qlib_models.items():
        try:
            importlib.import_module(module)
            status = "IMPORT OK"
        except ImportError as e:
            status = f"IMPORT FAIL ({e.__class__.__name__})"
        print(f"{name:15s} from {module:45s} -> {status}")


if __name__ == "__main__":
    main()
