import json
import re
import os
from pathlib import Path
from typing import Any
import yaml

# Mocking parts of artifacts_writer to debug
def debug_sync_factor(ws_root: Path, factor_meta_payload: dict[str, Any]):
    factors = factor_meta_payload.get('factors') or []
    code_map = {}
    print(f"Scanning workspace: {ws_root}")
    for py_file in ws_root.rglob("*.py"):
        print(f"  Found py file: {py_file.name}")
        try:
            content = py_file.read_text(encoding="utf-8")
            for f in factors:
                name = f.get("name")
                if not name: continue
                if re.search(rf"(class|def)\s+{name}\b", content):
                    print(f"    Matched factor {name} in {py_file.name}")
                    code_map[name] = content
        except Exception as e:
            print(f"    Error reading {py_file.name}: {e}")
            continue
    
    for f in factors:
        name = f.get('name')
        if name in code_map:
            code = code_map[name]
            match = re.search(rf'((class|def)\s+{name}.*?)(?=\n(class|def|\s*#|\s*$)|\Z)', code, re.DOTALL)
            if match:
                print(f"    Extracted code block for {name}, type: {match.group(2)}")
                if match.group(2) == 'class':
                    f['interface_info'] = {'type': 'class', 'standard_wrapper': f'factor_{name}'}
                else:
                    f['interface_info'] = {'type': 'function'}
            else:
                print(f"    Could not extract code block for {name}")
        else:
            print(f"    Factor {name} not found in code_map")
    return factor_meta_payload

def debug_sync_strategy(ws_root: Path):
    yaml_files = list(ws_root.glob("*.yaml")) + list(ws_root.glob("*.yml"))
    print(f"Scanning for strategy YAMLs in {ws_root}")
    strategy_config = None
    for yf in yaml_files:
        print(f"  Found yaml file: {yf.name}")
        try:
            with yf.open("r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict) and ("strategy" in content or "task" in content):
                    print(f"    Matched strategy config in {yf.name}")
                    strategy_config = content
                    # break # Check all for debugging
        except Exception as e:
            print(f"    Error reading {yf.name}: {e}")
            continue
    return strategy_config

if __name__ == "__main__":
    # Test on the workspace from previous logs
    ws = Path("f:/Dev/RD-Agent-main/git_ignore_folder/RD-Agent_workspace/90c6c6441a83499bbfb2d22d87c4a98e")
    
    # 1. Factor Meta Test
    fm_path = ws / "factor_meta.json"
    if fm_path.exists():
        fm = json.loads(fm_path.read_text(encoding="utf-8"))
        # Only test first few
        fm['factors'] = fm['factors'][:5]
        res = debug_sync_factor(ws, fm)
        # print("Result factors:", json.dumps(res['factors'], indent=2))
    
    # 2. Strategy Test
    res_strat = debug_sync_strategy(ws)
    if res_strat:
        print("Found strategy config")
    else:
        print("No strategy config found")
