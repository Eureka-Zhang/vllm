# Source from bash before running this checkout.
export VLLM_DEV_ROOT=/home/eureka/vllm
export CUDA_HOME=/home/eureka/miniconda3/envs/gpu-softmax
export PATH=$VLLM_DEV_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH
export CC=/home/eureka/miniconda3/envs/gpu-softmax/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/eureka/miniconda3/envs/gpu-softmax/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export NVCC_CCBIN=$CXX
export NVCC_PREPEND_FLAGS=-ccbin=$CXX
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}
export MAX_JOBS=2
# Experimental phase hooks are opt-in, not a serving default.
