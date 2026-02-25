#!/usr/bin/env bash
# 批量按 interval 与 gen_frame 组合运行 evaluate_accompaniment_metrics.py
# 用法：编辑下面的 INTERVALS 与 GEN_FRAMES 数组，或传入环境变量 DRY_RUN=1 只打印命令。

set -eu

# --- 配置区（编辑这些变量） ---
# 要遍历的 interval 值列表
INTERVALS=(1 2 4 7)

# 要遍历的 generation frame 值列表
GEN_FRAMES=(3 5 9 15)

# 样例命令中固定的 prompt / gen 标记
PROMPT_DIR_NAME="prompt_128_gen_576"      # 用于 generated 和 groundtruth 路径的父目录名

# 根目录（按你的示例调整）
REALTIME_ROOT="/data/home/bowenzheng/mbzuai-projects/AE/StreamMUSE/experiments-AE0/realtime/baseline"

# 输出结果目录前缀
OUT_ROOT="results-experiments-AE0"
# 其它固定参数
MELODY="Guitar"
POLYDIS_ROOT="./icm-deep-music-generation"
EXTRA_FLAGS="--auto-phrase-analysis --frechet-music-distance"

# 干跑模式（只打印不执行），可通过环境变量 DRY_RUN=1 启用
DRY_RUN="${DRY_RUN:-0}"

# uv 命令（如需要可改为 python3 evaluate_accompaniment_metrics.py）
CMD_BASE=("uv" "run" "evaluate_accompaniment_metrics.py")

# --- 运行 ---
mkdir -p "${OUT_ROOT}"

for I in "${INTERVALS[@]}"; do
  for G in "${GEN_FRAMES[@]}"; do
    GENERATED_DIR="${REALTIME_ROOT}/interval_${I}_gen_frame_${G}/${PROMPT_DIR_NAME}/generated"
    GT_DIR="${REALTIME_ROOT}/interval_${I}_gen_frame_${G}/${PROMPT_DIR_NAME}/gt_generation"
    OUT_FILE="${OUT_ROOT}/interval${I}_gen${G}_prompt_128_gen_576.json"

    # 打印并检查路径存在性（仅警告，不终止）
    echo "-------------------------------"
    echo "interval=${I}, gen_frame=${G}"
    echo "generated-dir: ${GENERATED_DIR}"
    echo "groundtruth-dir: ${GT_DIR}"
    echo "output-json: ${OUT_FILE}"

    if [ ! -d "${GENERATED_DIR}" ]; then
      echo "WARNING: generated dir 不存在，跳过 -> ${GENERATED_DIR}" >&2
      continue
    fi
    if [ ! -d "${GT_DIR}" ]; then
      echo "WARNING: groundtruth dir 不存在，跳过 -> ${GT_DIR}" >&2
      continue
    fi

    CMD=( "${CMD_BASE[@]}"
      "--generated-dir" "${GENERATED_DIR}"
      "--groundtruth-dir" "${GT_DIR}"
      "--output-json" "${OUT_FILE}"
      "--melody-track-names" "${MELODY}"
      "--auto-phrase-analysis"
    )

    # CMD=( "${CMD_BASE[@]}"
    #   "--generated-dir" "${GENERATED_DIR}"
    #   "--groundtruth-dir" "${GT_DIR}"
    #   "--output-json" "${OUT_FILE}"
    #   "--melody-track-names" "${MELODY}"
    #   ${EXTRA_FLAGS}
    #   "--polydis-root" "${POLYDIS_ROOT}"
    # )

    # 打印命令
    printf "RUN: %s\n" "${CMD[*]}"

    if [ "${DRY_RUN}" = "1" ]; then
      echo "DRY RUN - not executing"
    else
      "${CMD[@]}"
      RC=$?
      if [ $RC -ne 0 ]; then
        echo "Command failed (rc=$RC) for interval=${I} gen_frame=${G}" >&2
      else
        echo "Finished OK -> ${OUT_FILE}"
      fi
    fi

  done
done

echo "All done."
