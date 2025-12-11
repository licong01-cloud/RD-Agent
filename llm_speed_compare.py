import os
import time
from statistics import mean

from dotenv import load_dotenv
from openai import OpenAI


def timed_chat(client: OpenAI, model: str, messages, n_runs: int = 5, *, max_tokens: int = 64):
    latencies = []
    for i in range(n_runs):
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
        )
        end = time.perf_counter()
        latencies.append(end - start)
        text = resp.choices[0].message.content
        print(f"  Run {i+1}: {latencies[-1]:.3f}s, output_len={len(text)}")
    latencies.sort()
    avg = mean(latencies)
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.9) - 1]
    return avg, p50, p90


def main():
    # 优先从当前目录加载 .env（与 rdagent 一致）
    load_dotenv(".env")

    prompt = "简单介绍一下你自己，并用一句话回答。"
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt},
    ]

    n_runs = int(os.getenv("LLM_SPEED_TEST_RUNS", "5"))

    print(f"Using prompt: {prompt}")
    print(f"Runs per model: {n_runs}\n")

    # DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if deepseek_key:
        print(f"=== Testing DeepSeek: base={deepseek_base}, model={deepseek_model} ===")
        ds_client = OpenAI(api_key=deepseek_key, base_url=deepseek_base)
        ds_avg, ds_p50, ds_p90 = timed_chat(ds_client, deepseek_model, messages, n_runs, max_tokens=64)
        print(f"DeepSeek: avg={ds_avg:.3f}s, p50={ds_p50:.3f}s, p90={ds_p90:.3f}s\n")
    else:
        print("[Skip] DEEPSEEK_API_KEY not set, skipping DeepSeek test.\n")

    # SiliconFlow 固定模型测试：
    # 1) Pro/deepseek-ai/DeepSeek-V3.1-Terminus
    # 2) Pro/deepseek-ai/DeepSeek-V3.2-Exp
    # 只从环境变量中读取 OPENAI_API_KEY，其他参数写死，确保调用方式稳定可重复。
    sf_key = os.getenv("OPENAI_API_KEY")
    sf_base = "https://api.siliconflow.cn/v1"

    if sf_key:
        # 共享同一个客户端配置
        sf_client = OpenAI(api_key=sf_key, base_url=sf_base, timeout=15.0)

        # 模型 1：V3.1 Terminus
        sf_model_v31 = "Pro/deepseek-ai/DeepSeek-V3.1-Terminus"
        print(f"=== Testing SiliconFlow: base={sf_base}, model={sf_model_v31} ===")
        sf31_avg, sf31_p50, sf31_p90 = timed_chat(sf_client, sf_model_v31, messages, n_runs, max_tokens=64)
        print(f"SiliconFlow V3.1: avg={sf31_avg:.3f}s, p50={sf31_p50:.3f}s, p90={sf31_p90:.3f}s\n")

        # 模型 2：V3.2 Exp
        sf_model_v32 = "Pro/deepseek-ai/DeepSeek-V3.2-Exp"
        print(f"=== Testing SiliconFlow: base={sf_base}, model={sf_model_v32} ===")
        try:
            sf32_avg, sf32_p50, sf32_p90 = timed_chat(sf_client, sf_model_v32, messages, n_runs, max_tokens=64)
            print(f"SiliconFlow V3.2-Exp: avg={sf32_avg:.3f}s, p50={sf32_p50:.3f}s, p90={sf32_p90:.3f}s\n")
        except Exception as e:
            print(f"[Error] SiliconFlow V3.2-Exp test failed: {repr(e)}\n")
    else:
        print("[Skip] OPENAI_API_KEY not set, skipping SiliconFlow tests.\n")


if __name__ == "__main__":
    main()
