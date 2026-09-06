#!/usr/bin/env bash
# ============================================================
# TSQ + IPN-Hand (RGB, 8 段) 测试/评估脚本
# 用法（脚本会自动 cd 到仓库根目录）：
#   bash scripts/test_tsq_ipnhand_rgb_8f.sh
# 依赖与注意点同训练脚本；权重路径相对仓库根目录。
# ============================================================
set -e

# 进入仓库根目录（scripts/ 的上一级）
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---- 激活 ipnhand conda 环境（激活后使用环境内的 python，与系统 python3 无关）----
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    echo "[ERROR] 未找到 conda。请安装 miniconda3，或执行 export CONDA_DIR=<conda安装目录> 后重试。" >&2
    exit 1
fi
if conda env list | grep -qE '(^|[[:space:]])ipnhand([[:space:]]|$)'; then
    conda activate ipnhand
else
    echo "[ERROR] 不存在 conda 环境 'ipnhand'，请先执行: conda env create -f environment.yml" >&2
    exit 1
fi
echo "==> 使用解释器: $(command -v python) ($(python --version 2>&1))"
echo "==> 数据集根目录: $(python -c 'from ops import dataset_config; print(dataset_config.ROOT_DATASET)')"

# 按服务器实际可用 GPU 调整；若只有 1 张卡可改成 0
export CUDA_VISIBLE_DEVICES=0,1

python test_models.py ipnhand \
    --weights=checkpoint/TSQ_ipnhand_RGB_resnet50_shift0.5_blockres_avg_segment8_e50/ckpt.best.pth.tar \
    --test_segments=8 --test_crops=5 \
    --batch_size=4 -j 32  \
    --multi_clip_test > ipnhand_dete_test.out 2>&1

echo "==> 测试结束。查看结果: tail -n 30 ipnhand_dete_test.out"
