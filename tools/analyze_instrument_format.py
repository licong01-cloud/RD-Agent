"""
分析AIstock侧SOTA因子的instrument字段格式
"""
import re
from pathlib import Path
from collections import defaultdict


def analyze_instrument_format(code: str) -> dict:
    """
    分析因子代码中instrument字段的格式
    
    Returns:
        分析结果
    """
    results = {
        "instrument_usage": [],
        "index_format": None,
        "data_source": None,
        "instrument_operations": []
    }
    
    # 查找instrument的使用
    instrument_patterns = [
        r'\binstrument\b',
        r'\.groupby\(level=["\']instrument["\']\)',
        r'\.reset_index\(.*instrument.*\)',
        r'MultiIndex\(datetime, instrument\)',
    ]
    
    for pattern in instrument_patterns:
        matches = re.findall(pattern, code, re.IGNORECASE)
        if matches:
            results["instrument_usage"].extend(matches)
    
    # 查找数据源
    data_source_patterns = [
        r'pd\.read_hdf\(["\']([^"\']+)["\']',
        r'pd\.read_parquet\(["\']([^"\']+)["\']',
    ]
    
    for pattern in data_source_patterns:
        matches = re.findall(pattern, code)
        if matches:
            results["data_source"] = matches[0]
            break
    
    # 查找索引格式说明
    index_format_patterns = [
        r'MultiIndex\([^)]+\)',
        r'索引.*应为\s*([^。]+)',
    ]
    
    for pattern in index_format_patterns:
        matches = re.findall(pattern, code)
        if matches:
            results["index_format"] = matches[0]
            break
    
    # 查找instrument操作
    instrument_operations = [
        r'\.groupby\(level=["\']instrument["\']\)',
        r'\.reset_index\(level=["\']instrument["\']\)',
        r'\.reset_index\(level=0, drop=True\)',
    ]
    
    for pattern in instrument_operations:
        matches = re.findall(pattern, code)
        if matches:
            results["instrument_operations"].extend(matches)
    
    return results


def analyze_all_factors(code_dir: Path):
    """
    分析所有因子代码
    
    Args:
        code_dir: 因子代码目录
    """
    # 获取所有因子代码文件
    factor_files = sorted(code_dir.glob("factor_*.py"))
    
    print(f"找到 {len(factor_files)} 个因子代码文件")
    
    # 分析每个因子代码
    all_results = []
    for factor_file in factor_files:
        factor_name = factor_file.stem
        
        # 读取因子代码
        with open(factor_file, "r", encoding="utf-8") as f:
            code = f.read()
        
        # 分析因子代码
        result = analyze_instrument_format(code)
        result["factor_name"] = factor_name
        all_results.append(result)
    
    # 统计结果
    instrument_usage_count = sum(1 for r in all_results if r["instrument_usage"])
    data_source_count = sum(1 for r in all_results if r["data_source"])
    index_format_count = sum(1 for r in all_results if r["index_format"])
    
    print(f"\n统计结果:")
    print(f"  使用instrument的因子数量: {instrument_usage_count}/{len(all_results)}")
    print(f"  指定数据源的因子数量: {data_source_count}/{len(all_results)}")
    print(f"  指定索引格式的因子数量: {index_format_count}/{len(all_results)}")
    
    # 查找数据源
    data_sources = set(r["data_source"] for r in all_results if r["data_source"])
    print(f"\n数据源:")
    for ds in sorted(data_sources):
        print(f"  - {ds}")
    
    # 查找索引格式
    index_formats = set(r["index_format"] for r in all_results if r["index_format"])
    print(f"\n索引格式:")
    for fmt in sorted(index_formats):
        print(f"  - {fmt}")
    
    # 查找instrument操作
    instrument_operations = defaultdict(int)
    for r in all_results:
        for op in r["instrument_operations"]:
            instrument_operations[op] += 1
    
    print(f"\nInstrument操作统计:")
    for op, count in sorted(instrument_operations.items()):
        print(f"  - {op}: {count}次")
    
    # 保存结果
    import json
    output_file = code_dir / "instrument_format_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {output_file}")


if __name__ == "__main__":
    # 设置因子代码目录
    code_dir = Path("F:/Dev/AIstock/factors_ui")
    
    # 如果目录不存在，提示用户
    if not code_dir.exists():
        print(f"因子代码目录不存在: {code_dir}")
        exit(1)
    
    # 分析所有因子代码
    analyze_all_factors(code_dir)
