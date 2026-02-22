import requests
import os

api_key = os.environ.get('LITELLM_PROXY_API_KEY', 'sk-cshzmyeydshomofwovryfctdckkavcgnmsysyigttdgydwnp') # Using a fallback just in case
url = 'https://api.siliconflow.cn/v1/embeddings'
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# 常见 SiliconFlow Embedding 模型
models = [
    "BAAI/bge-m3",
    "BAAI/bge-large-zh-v1.5",
    "Pro/BAAI/bge-m3",
    "netease-youdao/bce-embedding-base_v1",
    "Qwen/Qwen2-math-BGE-M3",
    "Qwen/Qwen3-Embedding-0.6B",
]

# 构造一个超过 64 的大 batch (比如 100)
test_inputs = ["test"] * 100
print(f"Testing batch size of {len(test_inputs)}...")

for model in models:
    data = {
        "model": model,
        "input": test_inputs
    }
    try:
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code == 200:
            print(f"[SUCCESS] {model} supports batch size >= 100")
        else:
            print(f"[FAILED] {model} - {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[ERROR] {model}: {e}")

# 再测一次 1200 的大 batch
test_inputs_large = ["test"] * 1200
print(f"\nTesting batch size of {len(test_inputs_large)}...")

for model in models:
    data = {
        "model": model,
        "input": test_inputs_large
    }
    try:
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code == 200:
            print(f"[SUCCESS] {model} supports batch size >= 1200")
        else:
            print(f"[FAILED] {model} - {resp.status_code}: {resp.json().get('message', resp.text)}")
    except Exception as e:
        print(f"[ERROR] {model}: {e}")
