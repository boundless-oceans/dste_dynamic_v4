# TSQ / IPN-Hand 手势识别（RGB 8 帧）训练代码

基于 TSM/TSQ 手势识别训练代码改造，数据集为 IPN-Hand，训练/推理在 **Ubuntu 22.04 + NVIDIA GPU 服务器**上运行。

## 目录

```
main.py                        # 训练 / 验证入口
test_models.py                 # 测试 / 评估入口（多 clip 平均）
opts.py                        # 命令行参数
ops/
  dataset.py                   # IPN-Hand JSON 标注解析 + 采样 + 训练水平翻转
  models.py                    # TSN 模型（resnet50 + temporal module）
  temporal_module.py           # CoordAtt / SpatialAttention / 时域模块
  transforms.py                # 数据增强（含 GroupColorJitter 等）
  utils.py                     # load_checkpoint（兼容 torch>=2.6）等
scripts/
  train_tsq_ipnhand_rgb_8f.sh  # 训练脚本（自动激活 conda 环境）
  test_tsq_ipnhand_rgb_8f.sh   # 测试脚本（自动激活 conda 环境）
docs/                          # 数据流图（Mermaid）+ API 参考，见 docs/README.md
environment.yml                # conda 环境定义（环境名 ipnhand）
checkpoint/.../ckpt.best.pth.tar  # 已训练好的权重（不上 git，scp 上传）
```

> 想先搞清楚代码怎么跑通的，建议从 **[docs/ 文档](docs/README.md)** 看起，共两份：
> - **数据流图**（36 张 Mermaid）：端到端总览 → 数据加载 → 模型前向 → 训练回路 → 评测流程
> - **[API 参考](docs/api/README.md)**：逐文件讲解每个类与函数的作用、参数与坑

## 服务器部署步骤

### 1. 上传

把整个仓库（含 `ops/`、`scripts/`、`checkpoint/`）按原结构传到服务器（`checkpoint/` 体积大，用 scp/rsync，别走 git）。之后所有命令都在仓库根目录执行。

### 2. 创建 conda 环境

```bash
# 若还没装 miniconda（任选其一安装）：
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh

conda env create -f environment.yml
conda activate ipnhand
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.device_count())"
```

> 说明：环境名 `ipnhand`、python 3.10、torch 2.1.2 / torchvision 0.16.2（CUDA 11.8）。
> - 若服务器驱动支持 CUDA 12（`nvidia-smi` 显示 driver >= 525），把 `environment.yml` 里改成
>   `pytorch=2.4.1` / `torchvision=0.19.1` / `pytorch-cuda=12.1`（或 12.4）后再创建。
> - Ubuntu 系统只自带 `python3`，但这与训练无关：脚本会先 `conda activate ipnhand`，
>   激活后使用的就是环境内的 `python`。

### 3. 检查数据路径

`ops/dataset_config.py` 里写死了 `ROOT_DATASET = '/home/vis/datasets/'`，要求服务器上有：

```
/home/vis/datasets/IPNHand/
  annotation_ipnhand/training.json
  annotation_ipnhand/validation.json
  每个视频一个目录，帧命名 000001.jpg, 000002.jpg ...
```

路径不一致就改这一个文件（或设软链接），其余代码无需动。

### 4. 先跑测试做端到端验证（用现成权重）

```bash
bash scripts/test_tsq_ipnhand_rgb_8f.sh
tail -n 30 ipnhand_dete_test.out
```

能正常结束并打印 `Overall Prec@1 ...`（约 89% 量级）即说明环境、数据、模型都通。

### 5. 训练

```bash
# 直接前台跑（ssh 断开会被杀）：
bash scripts/train_tsq_ipnhand_rgb_8f.sh
# 或推荐后台跑（日志仍写在 ipnhand_dste_train.out）：
nohup bash scripts/train_tsq_ipnhand_rgb_8f.sh > ipnhand_dste_train_console.log 2>&1 &
tail -f ipnhand_dste_train.out
```

训练会按 epoch 自动保存到 `checkpoint/<store_name>/`，best 模型为 `ckpt.best.pth.tar`。
训练脚本默认 `CUDA_VISIBLE_DEVICES=0,1`（2 卡 DataParallel）；只有 1 张卡就改成 `0`。

训练开始日志应出现 16 行 `=> in_channels: ..., in_h: ...`，随后注意一行：
- `=> 训练水平翻转已启用：…` → 已识别左右方向类别配对，翻转 + 标签对调生效；
- `WARNING: 无法自动识别…` → 该数据集标注名不在识别规则内，翻转关闭（安全，只是少了左右均衡）。

## 版权与致谢

本仓库是 **TSM: Temporal Shift Module for Efficient Video Understanding**
官方实现（[mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module)）的衍生作品，
在其基础上针对 **IPN-Hand** 手势识别数据集做了改造与优化。许可为 **MIT**，详见 [LICENSE](LICENSE)。

> 说明：`main.py`、`test_models.py`、`opts.py`、`ops/dataset.py`、`ops/models.py`、
> `ops/dataset_config.py` 文件头保留的 `Code for "TSM: ..."` 与 `{jilin, songhan}@mit.edu`
> 是上游原始版权声明，按 MIT 许可要求**原样保留，请勿删除**。
> `ops/non_local.py` 来自 [AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-Local_pytorch)，经 TSM 官方实现引入。

### 贡献划分

| 阶段 | 贡献者 | 内容 |
|---|---|---|
| 地基（提交 1，2026-03-12） | **TangShuQiang** | 初始版本：IPN-Hand 数据读取与标注解析（`ops/dataset.py`、`ops/dataset_config.py`）、TSN/TSQ 模型搭建（`ops/models.py`、`ops/basic_ops.py`）、训练与测试入口（`main.py`、`test_models.py`）、运行脚本 |
| 优化（提交 2，2026-09-06） | **boundless_oceans** | 见下方清单 |
| 修正（提交 3，2026-09-06） | **boundless_oceans** | 数据集根目录路径修正 |

> 提交 1 由师兄 **TangShuQiang** 完成；提交 2、3 是在其基础上由 **boundless_oceans** 优化并提交。
> 原开发环境中提交 2、3 曾借用师兄的 Git 身份提交，移植到本仓库时已按实际贡献改回各自作者。
> 提交 1 保留师兄的原始提交邮箱以忠实反映其贡献；本人提交的邮箱为
> `boundless-oceans@users.noreply.github.com`（GitHub noreply 地址，避免公开真实邮箱）。

### 优化清单（提交 2）

- **`ops/temporal_module.py`**：`CoordAtt` 坐标注意力改造 —— 由固定深度可分离卷积改为按时序段数
  `n_segment` 与特征图高度 `in_h` 动态生成 3×3 空间卷积核（`fc1`/`fc2` 分别沿 h、w 预测核并做 softmax，
  外积后以 `groups=n*c` 的深度卷积施加），并新增 `BN+ReLU`；时序模块相应改为调用新版 `CoordAtt`。
- **类别均衡采样**：`WeightedRandomSampler` 按类别频率倒数加权，提升稀有类（如双指双击）被抽中的概率。
- **训练水平翻转**：整段帧一起镜像，并自动识别「向左/向右」类别对、同步对调标签；识别不到配对时安全关闭翻转。
- **`GroupColorJitter`**：同组帧共享随机参数做亮度/对比度抖动，避免给时序通道引入额外噪声。
- **可复现性**：DataLoader 各 worker 独立随机种子，并统一设置 `torch`/`numpy`/`random` 种子。
- **`ops/utils.py`**：新增 `load_checkpoint`，兼容 torch >= 2.6 默认 `weights_only=True` 导致的加载失败。
- **工程化**：新增 `README.md`、`environment.yml`、`.gitignore`；训练/测试脚本自动激活 conda 环境并校验数据路径。

### 本仓库相对源开发仓库的差异

- 清除了源仓库 `init` 误提交的 `ops/__pycache__/*.pyc`（9 个编译产物），全部提交中均已不含。

## 常见问题

| 现象 | 处理 |
|---|---|
| `[ERROR] 未找到 conda` | 装 miniconda，或运行前 `export CONDA_DIR=/path/to/miniconda3`（脚本也接受 `~/anaconda3` 以外的路径） |
| 提示没有 `ipnhand` 环境 | 先 `conda env create -f environment.yml` |
| `torch` import 报 CUDA 驱动错误 | 驱动太老/太新 → 按上文调整 `pytorch-cuda` 版本重建环境；或 `nvidia-smi` 确认驱动 |
| 一堆 `video path not exist!!!` | 数据路径/JSON 与 `ops/dataset_config.py` 的 `ROOT_DATASET` 不一致 |
| 首次构建卡住/报下载失败 | 需联网下载 ImageNet 预训练权重（torchvision 缓存到 `~/.cache/torch/hub/checkpoints`），可提前手动缓存 |
| 加载 ckpt 报错 | torch >= 2.6 时默认 `weights_only=True`；本仓库已统一走 `ops/utils.load_checkpoint`，无需处理 |

## 性能表现

以下数值为配套权重
`checkpoint/TSQ_ipnhand_RGB_resnet50_shift0.5_blockres_avg_segment8_e50/ckpt.best.pth.tar`
在 IPN-Hand 验证集（1097 段手势，13 类）上的实测结果：

| 指标 | 数值 |
|---|---|
| **Overall Prec@1** | **89.06%** |
| **Overall Prec@5** | **99.27%** |
| Class Accuracy（各类准确率的平均） | 83.44% |
| 理论上界（每类取最优，`test_models.py` 的 `upper`） | 83.58% |

评测口径：8 段 × 5 crop × 10 clip，共 400 次前向/样本，对 50 个预测取平均
（`--test_segments=8 --test_crops=5 --multi_clip_test`）。

各类准确率分布很宽（**48.08% ~ 100%**），说明类别不均衡确实存在 ——
这正是训练时引入 `WeightedRandomSampler` 类别均衡采样的原因。

> 复现方式：`bash scripts/test_tsq_ipnhand_rgb_8f.sh`，随后 `tail -n 30 ipnhand_dete_test.out`。
> 权重体积较大未纳入版本库，需用 scp/rsync 单独上传。
>
> ⚠️ 上表是**测试口径**（多 clip 多 crop 集成）。训练日志里的 `Best Prec@1: 88.97`
> 是**验证口径**（单 clip 单 crop），两者不可直接比较；这里数值接近是正常的，
> 因为 `ckpt.best.pth.tar` 正是由验证口径选出的。
