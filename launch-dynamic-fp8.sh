#!/usr/bin/env bash
set -euo pipefail
cd /home/eureka/vllm
source dev-env.sh
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1
export VLLM_GGUF_PREPACK_MIXED=${VLLM_GGUF_PREPACK_MIXED:-1}
export VLLM_GGUF_DEQUANT_NOZERO=${VLLM_GGUF_DEQUANT_NOZERO:-1}
export VLLM_GGUF_MATVEC_MODE=${VLLM_GGUF_MATVEC_MODE:-3}
export VLLM_EXPERIMENTAL_PHASE_MEMORY=1
export VLLM_EXPERIMENTAL_PHASE_ADMISSION=1
export PYTHONPATH=/home/eureka/vllm:${PYTHONPATH:-}
exec .venv/bin/python -m vllm.entrypoints.openai.api_server --model /home/eureka/specdecode-lab/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf --tokenizer /home/eureka/qwen38-gguf-lab/tokenizer --hf-config-path /home/eureka/qwen38-gguf-lab/tokenizer --served-model-name qwen38-27b-iq3 --host 127.0.0.1 --port 8008 --max-model-len 16384 --max-num-seqs 1 --mm-processor-cache-gb 0 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.90 --enforce-eager --gdn-prefill-backend triton --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":589824}' --dtype bfloat16 --kv-cache-dtype fp8 --no-enable-prefix-caching --enable-extensible-kv-cache --worker-extension-cls phase_dev_worker.PhaseDevWorker "$@"
