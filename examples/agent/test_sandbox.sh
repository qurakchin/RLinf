#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export DENO_NO_UPDATE_CHECK=1  # 禁用更新检查
export DENO_FUTURE=1 

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="test-sandbox"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/agent/test_sandbox.py --config-path ${CONFIG_PATH}/config/ --config-name $CONFIG_NAME
