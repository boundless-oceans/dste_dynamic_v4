#!/usr/bin/env bash
# ============================================================
# TSQ + IPN-Hand (RGB, 8 段) 训练脚本
#
# 用法（脚本会自动 cd 到仓库根目录，从哪调用都行）：
#   bash scripts/train_tsq_ipnhand_rgb_8f.sh
#
# 前置条件：
#   1) 已创建 conda 环境:  conda env create -f environment.yml
#   2) conda 不在默认位置时，先执行: export CONDA_DIR=/path/to/miniconda3
#      （默认依次尝试 PATH 中的 conda 与 ~/miniconda3）
#   3) 数据路径与 ops/dataset_config.py 中的 ROOT_DATASET 一致
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

python main.py ipnhand RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 16 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=0.50 --shift_place=blockres --npb > ipnhand_dste_train.out 2>&1

echo "==> 训练结束。查看进度: tail -f ipnhand_dste_train.out"
