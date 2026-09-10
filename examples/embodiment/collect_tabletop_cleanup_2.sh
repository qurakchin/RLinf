#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
export LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/tabletop_cleanup/variant_2/$(date +'%Y%m%d-%H:%M:%S')}"

TASK_DESCRIPTION='Tidy up the table. Left and right refer to the top camera view. Sort all three bottles by brand. Place all Pepsi bottles in the bag on the right and all Coca-Cola bottles in the bag on the left. Move the bowls to uncover the spoons, place the white spoon in the white bowl and the pink spoon in the pink bowl, and return both bowls to their original marked positions: white bowl on the left and pink bowl on the right.'

echo "[采集组 B / 入口 2] 左右以 top 相机画面为准"
echo "[入袋规则] 左袋：可口可乐；右袋：百事可乐"
echo "[初始勺子] 左碗后：粉勺；右碗后：白勺"
echo "[最终状态] 同色勺入同色碗；白碗归左标记，粉碗归右标记"
echo "[中间踏板] 开启跟随后可复位；录制中回位完成，再按白键结束"
echo "[录制结束后] 左脚删除，右脚保存；中间不用"
echo '[保留目标] 默认 90 条；如传入条数参数，以最终配置和进度条为准'
echo "[摆放计划] 按累计保留编号：001–030 百事在左；031–060 在中；061–090 在右（人工摆放）"
echo "[输出目录] ${LOG_DIR}"

bash "${SCRIPT_DIR}/collect_data.sh" realworld_dual_yam_collect_data \
    runner.num_data_episodes=90 \
    "env.eval.override_cfg.task_description=\"${TASK_DESCRIPTION}\"" \
    "$@"
