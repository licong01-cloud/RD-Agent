"""
分析所有因子代码中的股票代码格式
"""
import re
from pathlib import Path
from collections import defaultdict


def analyze_stock_code_format(code: str) -> dict:
    """
    分析股票代码的格式
    
    Returns:
        {
            "format_type": 格式类型,
            "examples": 示例,
            "patterns": 匹配到的模式
        }
    """
    patterns = {
        "6位数字": r'\b\d{6}\b',
        "字母+数字": r'\b[A-Za-z]+\d+\b',
        "字符串变量": r'\b[a-z_]+[a-z0-9_]*\b',
        "DataFrame列名": r'\b["\'][\w_]+["\']\b',
        "MultiIndex": r'\b\w+\.\w+\b',
    }
    
    results = {}
    for pattern_name, pattern in patterns.items():
        matches = re.findall(pattern, code)
        if matches:
            results[pattern_name] = matches[:10]  # 只保留前10个示例
    
    return results


def analyze_factor_code(code: str, factor_name: str) -> dict:
    """
    分析因子代码中的股票代码格式
    
    Args:
        code: 因子代码
        factor_name: 因子名称
    
    Returns:
        分析结果
    """
    # 查找所有可能的股票代码格式
    results = analyze_stock_code_format(code)
    
    # 查找特定的股票代码模式
    stock_code_patterns = {
        "instrument": r'\binstrument\b',
        "stock": r'\bstock\b',
        "code": r'\bcode\b',
        "symbol": r'\bsymbol\b',
    }
    
    variable_names = []
    for pattern_name, pattern in stock_code_patterns.items():
        if re.search(pattern, code, re.IGNORECASE):
            # 查找相关的变量名
            pattern = rf'\b\w*{pattern}\w*\b'
            matches = re.findall(pattern, code, re.IGNORECASE)
            variable_names.extend(matches[:5])
    
    return {
        "factor_name": factor_name,
        "stock_code_formats": results,
        "variable_names": list(set(variable_names)),
    }


def analyze_all_factors(code_dir: Path):
    """
    分析所有因子代码
    
    Args:
        code_dir: 因子代码目录
    """
    # 获取所有因子代码文件
    factor_files = list(code_dir.glob("*.py"))
    
    print(f"找到 {len(factor_files)} 个因子代码文件")
    
    # 统计所有格式
    all_formats = defaultdict(list)
    all_variable_names = defaultdict(list)
    
    # 分析每个因子代码
    for factor_file in factor_files:
        factor_name = factor_file.stem
        
        # 读取因子代码
        with open(factor_file, "r", encoding="utf-8") as f:
            code = f.read()
        
        # 分析因子代码
        result = analyze_factor_code(code, factor_name)
        
        # 统计格式
        for format_type, examples in result["stock_code_formats"].items():
            all_formats[format_type].extend(examples)
        
        # 统计变量名
        for var_name in result["variable_names"]:
            all_variable_names[var_name].append(factor_name)
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("股票代码格式统计")
    print("=" * 80)
    
    for format_type, examples in sorted(all_formats.items()):
        print(f"\n{format_type}:")
        print(f"  数量: {len(examples)}")
        print(f"  示例: {set(examples)}")
    
    print("\n" + "=" * 80)
    print("变量名统计")
    print("=" * 80)
    
    for var_name, factor_names in sorted(all_variable_names.items()):
        print(f"\n{var_name}:")
        print(f"  数量: {len(factor_names)}")
        print(f"  因子: {factor_names[:5]}")  # 只显示前5个
    
    # 保存结果
    import json
    output_file = code_dir / "stock_code_format_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "formats": dict(all_formats),
            "variable_names": dict(all_variable_names),
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {output_file}")


if __name__ == "__main__":
    # 设置因子代码目录
    code_dir = Path("F:/Dev/RD-Agent-main/log/2026-01-27_12-17-12-637715/sota_factors_code")
    
    # 如果目录不存在，提示用户
    if not code_dir.exists():
        print(f"因子代码目录不存在: {code_dir}")
        exit(1)
    
    # 分析所有因子代码
    analyze_all_factors(code_dir)
