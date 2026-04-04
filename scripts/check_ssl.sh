#!/bin/bash
source /home/lc999/miniconda3/etc/profile.d/conda.sh
conda activate rdagent-gpu

echo "=== OpenSSL ==="
python -c "import ssl; print(ssl.OPENSSL_VERSION)"

echo "=== pyarrow ==="
python -c "import pyarrow; print(pyarrow.__version__)"

echo "=== libssl check ==="
ls -la /home/lc999/miniconda3/envs/rdagent-gpu/lib/libssl* 2>/dev/null
ls -la /home/lc999/miniconda3/envs/rdagent-gpu/lib/libcrypto* 2>/dev/null

echo "=== openssl package ==="
conda list openssl 2>/dev/null | grep openssl

echo "=== pyarrow package ==="
conda list pyarrow 2>/dev/null | grep pyarrow
