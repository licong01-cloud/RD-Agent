#!/bin/bash
source /home/lc999/miniconda3/etc/profile.d/conda.sh
conda activate rdagent-gpu
cd /tmp
python /mnt/f/Dev/RD-Agent-main/scripts/test_qlib_cache_inner.py
