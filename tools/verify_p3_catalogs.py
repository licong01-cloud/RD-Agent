import json
from pathlib import Path

def verify_catalogs():
    base_path = Path("RDagentDB/aistock")
    
    # 1. Verify Factor Catalog
    factor_file = base_path / "factor_catalog.json"
    if factor_file.exists():
        data = json.loads(factor_file.read_text(encoding="utf-8"))
        factors_with_interface = [f for f in data.get("factors", []) if "interface_info" in f]
        print(f"Factors total: {len(data.get('factors', []))}")
        print(f"Factors with interface_info: {len(factors_with_interface)}")
        if factors_with_interface:
            print("Example factor interface_info:", json.dumps(factors_with_interface[0]["interface_info"], indent=2))
    
    # 2. Verify Strategy Catalog
    strategy_file = base_path / "strategy_catalog.json"
    if strategy_file.exists():
        data = json.loads(strategy_file.read_text(encoding="utf-8"))
        strategies = data.get("strategies", [])
        strategies_with_p3 = [s for s in strategies if s.get("python_implementation")]
        print(f"Strategies total: {len(strategies)}")
        print(f"Strategies with python_implementation: {len(strategies_with_p3)}")
        if len(strategies) > len(strategies_with_p3):
            missing_paths = [s.get("workspace_example", {}).get("workspace_path") for s in strategies if not s.get("python_implementation")]
            print(f"Missing strategies (first 5 paths): {missing_paths[:5]}")
        if strategies_with_p3:
            print("Example strategy python_implementation:", json.dumps(strategies_with_p3[0]["python_implementation"], indent=2))
            
    # 3. Verify Model Catalog
    model_file = base_path / "model_catalog.json"
    if model_file.exists():
        data = json.loads(model_file.read_text(encoding="utf-8"))
        models_with_config = [m for m in data.get("models", []) if m.get("model_config")]
        print(f"Models total: {len(data.get('models', []))}")
        print(f"Models with model_config: {len(models_with_config)}")
        if models_with_config:
            print("Example model metadata keys:", list(models_with_config[0].keys()))

if __name__ == "__main__":
    verify_catalogs()
