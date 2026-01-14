import json
from pathlib import Path

def check_catalog_counts():
    root = Path('f:/Dev/RD-Agent-main/RDagentDB/aistock')
    if not root.exists():
        root = Path('/mnt/f/Dev/RD-Agent-main/RDagentDB/aistock')
        
    print(f"Checking catalogs in: {root}")
    print("Correct Catalog export commands:")
    print("python tools/export_aistock_factor_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/factor_catalog.json")
    print("python tools/export_aistock_model_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/model_catalog.json")
    print("python tools/export_aistock_loop_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/loop_catalog.json")
    print("python tools/export_aistock_strategy_catalog.py --registry-sqlite RDagentDB/registry.sqlite --output RDagentDB/aistock/strategy_catalog.json")
    for f in root.glob('*_catalog.json'):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            name = f.name.replace('_catalog.json', '')
            if name == 'strategy':
                key = 'strategies'
            else:
                key = name + 's'
            items = data.get(key, [])
            print(f"{f.name}: {len(items)} items")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

if __name__ == "__main__":
    check_catalog_counts()
