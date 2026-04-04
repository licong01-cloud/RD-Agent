#!/usr/bin/env python3
"""测试 MiniMax API 连接和响应"""

import os
from rdagent.oai.backend.litellm import LiteLLMAPIBackend

def test_minimax_simple():
    """简单测试：短prompt"""
    print("=" * 60)
    print("测试 1: 简单短prompt")
    print("=" * 60)

    backend = LiteLLMAPIBackend()

    try:
        response = backend.build_messages_and_create_chat_completion(
            system_prompt="你是一个有用的助手。",
            user_prompt="请用一句话介绍量化交易。",
        )
        print(f"✓ 成功！响应: {response[:200]}...")
        return True
    except Exception as e:
        print(f"✗ 失败: {type(e).__name__}: {str(e)[:200]}")
        return False

def test_minimax_long():
    """长文本测试：模拟因子生成场景"""
    print("\n" + "=" * 60)
    print("测试 2: 长prompt（模拟因子生成）")
    print("=" * 60)

    backend = LiteLLMAPIBackend()

    # 模拟实际的因子生成prompt长度
    system_prompt = "你是量化研究员。" + "背景信息：" * 500  # ~5KB
    user_prompt = "请生成一个基于资金流的因子。" + "要求：" * 200  # ~2KB

    try:
        response = backend.build_messages_and_create_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        print(f"✓ 成功！响应长度: {len(response)} 字符")
        print(f"  前100字符: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ 失败: {type(e).__name__}: {str(e)[:300]}")
        return False

def test_minimax_json():
    """JSON模式测试"""
    print("\n" + "=" * 60)
    print("测试 3: JSON模式（CoSTEER使用的模式）")
    print("=" * 60)

    backend = LiteLLMAPIBackend()

    try:
        response = backend.build_messages_and_create_chat_completion(
            system_prompt="你是代码生成助手。",
            user_prompt="生成一个计算均值的Python函数。",
            json_mode=True,
            json_target_type=dict,
        )
        print(f"✓ 成功！响应: {response}")
        return True
    except Exception as e:
        print(f"✗ 失败: {type(e).__name__}: {str(e)[:300]}")
        return False

if __name__ == "__main__":
    print(f"当前模型: {os.getenv('CHAT_MODEL', 'default')}")
    print(f"API Backend: LiteLLM\n")

    results = []
    results.append(("简单测试", test_minimax_simple()))
    results.append(("长文本测试", test_minimax_long()))
    results.append(("JSON模式测试", test_minimax_json()))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n✓ MiniMax API 完全可用，可以继续运行 RDAgent")
    elif passed > 0:
        print("\n⚠ MiniMax API 部分可用，可能在长文本或JSON模式下不稳定")
    else:
        print("\n✗ MiniMax API 不可用，建议切换到 DeepSeek")
