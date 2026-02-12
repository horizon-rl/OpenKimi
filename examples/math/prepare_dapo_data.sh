#!/usr/bin/env bash
set -uxo pipefail

export DATA_DIR=${DATA_DIR:-"${HOME}/verl/data"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo-math-17k.parquet"}
export VAL_FILE=${VAL_FILE:-"${DATA_DIR}/math-aime-eval.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${DATA_DIR}"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/fengyao1909/dapo-math-17k-deduplicated/resolve/main/dapo-math-17k.parquet?download=true"
fi

if [ ! -f "${VAL_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${VAL_FILE}" "https://huggingface.co/datasets/zhenghaoxu/math-aime-eval/resolve/main/data/validation-00000-of-00001.parquet?download=true"
fi
