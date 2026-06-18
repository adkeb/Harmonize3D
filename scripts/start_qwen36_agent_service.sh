#!/usr/bin/env bash
set -euo pipefail

export NVFP4_BACKEND="${NVFP4_BACKEND:-flashinfer-cutlass}"

# vLLM OpenAI-compatible service for the Harmonize3D Auto Agent planner.
# The model is downloaded from hf-mirror.com directly; proxy variables are
# removed for both the host-side docker command and the container runtime.

IMAGE_NAME="${IMAGE_NAME:-vllm/vllm-openai:cu130-nightly}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_qwen36_agent}"
HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/root/sakura/models}"
HF_MODEL_ID="${HF_MODEL_ID:-nvidia/Qwen3.6-35B-A3B-NVFP4}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-${MODEL_DIR}/Qwen3.6-35B-A3B-NVFP4}"
CONTAINER_MODEL_DIR="${CONTAINER_MODEL_DIR:-/root/.cache/huggingface/Qwen3.6-35B-A3B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b-nvfp4}"

if [[ -n "${MODEL_ID:-}" ]]; then
  EFFECTIVE_MODEL_ID="$MODEL_ID"
elif [[ -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
  EFFECTIVE_MODEL_ID="$CONTAINER_MODEL_DIR"
else
  EFFECTIVE_MODEL_ID="$HF_MODEL_ID"
fi

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
DTYPE="${DTYPE:-auto}"
QUANTIZATION="${QUANTIZATION:-modelopt}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
GPU_RUNTIME="${GPU_RUNTIME:-nvidia}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
CPU_OFFLOAD_PARAMS="${CPU_OFFLOAD_PARAMS:-}"
OFFLOAD_BACKEND="${OFFLOAD_BACKEND:-}"
OFFLOAD_GROUP_SIZE="${OFFLOAD_GROUP_SIZE:-}"
OFFLOAD_NUM_IN_GROUP="${OFFLOAD_NUM_IN_GROUP:-}"
OFFLOAD_PARAMS="${OFFLOAD_PARAMS:-}"
OFFLOAD_PREFETCH_STEP="${OFFLOAD_PREFETCH_STEP:-}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-}"
MAX_PARALLEL_LOADING_WORKERS="${MAX_PARALLEL_LOADING_WORKERS:-}"

DOCKER_GPU_ARGS=()
case "$GPU_RUNTIME" in
  nvidia)
    DOCKER_GPU_ARGS=(--runtime=nvidia)
    ;;
  gpus)
    DOCKER_GPU_ARGS=(--gpus all)
    ;;
  none)
    DOCKER_GPU_ARGS=()
    ;;
  *)
    echo "Unsupported GPU_RUNTIME=${GPU_RUNTIME}; use nvidia, gpus, or none." >&2
    exit 2
    ;;
esac

VLLM_ARGS=(
  "--model" "$EFFECTIVE_MODEL_ID"
  "--served-model-name" "$SERVED_MODEL_NAME"
  "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
  "--max-model-len" "$MAX_MODEL_LEN"
  "--quantization" "$QUANTIZATION"
  "--host" "0.0.0.0"
  "--port" "$CONTAINER_PORT"
  "--max-num-seqs" "$MAX_NUM_SEQS"
  "--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
  "--block-size" "$BLOCK_SIZE"
  "--dtype" "$DTYPE"
  "--kv-cache-dtype" "$KV_CACHE_DTYPE"
  "--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE"
  "--language-model-only"
  "--trust-remote-code"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "qwen3_coder"
  "--reasoning-parser" "qwen3"
)

case "$ENABLE_PREFIX_CACHING" in
  1|true|TRUE|yes|YES)
    VLLM_ARGS+=("--enable-prefix-caching")
    ;;
  0|false|FALSE|no|NO)
    VLLM_ARGS+=("--no-enable-prefix-caching")
    ;;
  *)
    echo "Unsupported ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}; use 1 or 0." >&2
    exit 2
    ;;
esac

if [[ "$CPU_OFFLOAD_GB" != "0" ]]; then
  VLLM_ARGS+=("--cpu-offload-gb" "$CPU_OFFLOAD_GB")
fi
if [[ -n "$CPU_OFFLOAD_PARAMS" ]]; then
  # shellcheck disable=SC2206
  CPU_OFFLOAD_PARAM_ARGS=($CPU_OFFLOAD_PARAMS)
  VLLM_ARGS+=("--cpu-offload-params" "${CPU_OFFLOAD_PARAM_ARGS[@]}")
fi
if [[ -n "$OFFLOAD_BACKEND" ]]; then
  VLLM_ARGS+=("--offload-backend" "$OFFLOAD_BACKEND")
fi
if [[ -n "$OFFLOAD_GROUP_SIZE" ]]; then
  VLLM_ARGS+=("--offload-group-size" "$OFFLOAD_GROUP_SIZE")
fi
if [[ -n "$OFFLOAD_NUM_IN_GROUP" ]]; then
  VLLM_ARGS+=("--offload-num-in-group" "$OFFLOAD_NUM_IN_GROUP")
fi
if [[ -n "$OFFLOAD_PARAMS" ]]; then
  # shellcheck disable=SC2206
  OFFLOAD_PARAM_ARGS=($OFFLOAD_PARAMS)
  VLLM_ARGS+=("--offload-params" "${OFFLOAD_PARAM_ARGS[@]}")
fi
if [[ -n "$OFFLOAD_PREFETCH_STEP" ]]; then
  VLLM_ARGS+=("--offload-prefetch-step" "$OFFLOAD_PREFETCH_STEP")
fi
if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  VLLM_ARGS+=("--kv-cache-memory-bytes" "$KV_CACHE_MEMORY_BYTES")
fi
if [[ -n "$KV_OFFLOADING_SIZE" ]]; then
  VLLM_ARGS+=("--kv-offloading-size" "$KV_OFFLOADING_SIZE")
fi
if [[ -n "$MAX_PARALLEL_LOADING_WORKERS" ]]; then
  VLLM_ARGS+=("--max-parallel-loading-workers" "$MAX_PARALLEL_LOADING_WORKERS")
fi

echo "Starting vLLM Auto Agent service: ${CONTAINER_NAME}"
echo "HF model: ${HF_MODEL_ID}"
echo "Effective model: ${EFFECTIVE_MODEL_ID}"
echo "Host local model dir: ${LOCAL_MODEL_DIR}"
echo "Served model: ${SERVED_MODEL_NAME}"
echo "HF endpoint: https://hf-mirror.com"
echo "NVFP4 backend: ${NVFP4_BACKEND}"
echo "Quantization: ${QUANTIZATION}"
echo "Docker GPU runtime mode: ${GPU_RUNTIME}"
echo "CPU offload GiB: ${CPU_OFFLOAD_GB}"
echo "Prefix caching: ${ENABLE_PREFIX_CACHING}"
echo "Port: ${HOST_PORT} -> ${CONTAINER_PORT}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'docker run args:'
  printf ' %q' "${VLLM_ARGS[@]}"
  printf '\n'
  exit 0
fi

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy NO_PROXY="*" no_proxy="*" \
  docker run --rm -it \
    --name "$CONTAINER_NAME" \
    "${DOCKER_GPU_ARGS[@]}" \
    --ipc=host \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    -v "${MODEL_DIR}:/root/.cache/huggingface" \
    -e HF_HOME="/root/.cache/huggingface" \
    -e HF_ENDPOINT="https://hf-mirror.com" \
    -e NO_PROXY="*" \
    -e no_proxy="*" \
    -e NVFP4_BACKEND="$NVFP4_BACKEND" \
    -e VLLM_USE_V1="$VLLM_USE_V1" \
    -e VLLM_LOGGING_LEVEL="$VLLM_LOGGING_LEVEL" \
    -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    "$IMAGE_NAME" \
    "${VLLM_ARGS[@]}"
