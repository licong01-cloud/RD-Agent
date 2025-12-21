"""Test embedding batch size using RD-Agent's LiteLLMAPIBackend.

This script intentionally复用 RD-Agent 内部的 LiteLLMAPIBackend，
走与 rdagent health_check 完全相同的调用路径，用于测量
当前 EMBEDDING_MODEL（例如 litellm_proxy/Qwen/Qwen3-Embedding-0.6B）
在实际环境中的最大可用 batch size。

使用方法（在 WSL 中）：

    cd /mnt/f/dev/RD-Agent-main
    conda activate rdagent
    set -a; source .env; set +a
    python test_embedding_batch.py

脚本会输出不同 batch_size 下是否成功，
一旦碰到错误（例如 413 或其他限制）就停止，
最后一个 "OK" 的 batch_size 即为当前环境下的安全上限。
"""

from __future__ import annotations

from typing import Iterable

from rdagent.oai.backend.litellm import LiteLLMAPIBackend


def test_batch_sizes(
    batch_sizes: Iterable[int] = (
        32,
        64,
        80,
        96,
        128,
        160,
        192,
        224,
        256,
        320,
        384,
        448,
        512,
    ),
) -> None:
    backend = LiteLLMAPIBackend()

    print("Using RD-Agent LiteLLMAPIBackend to test embedding batches.")
    print("Current EMBEDDING_MODEL is read from .env (e.g. litellm_proxy/Qwen/Qwen3-Embedding-0.6B).")

    for bs in batch_sizes:
        texts = [f"test sentence {i}" for i in range(bs)]
        try:
            embeddings = backend.create_embedding(texts)
            print(f"batch_size={bs}: OK, got {len(embeddings)} embeddings")
        except Exception as e:  # noqa: BLE001 - 这里只是测试工具
            print(f"batch_size={bs}: ERROR -> {repr(e)}")
            break


def main() -> None:
    test_batch_sizes()


if __name__ == "__main__":
    main()
