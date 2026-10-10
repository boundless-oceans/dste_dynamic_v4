# 数据流图（Data Flow）

本目录用 Mermaid 图描述 TSQ / IPN-Hand 手势识别代码的数据流。
所有图形均可在 GitHub 上直接渲染（GitHub 原生支持 Mermaid）。

## 文档结构

| 文件 | 内容 |
|---|---|
| [01-pipeline-overview.md](01-pipeline-overview.md) | 端到端总览：从 JSON 标注到预测结果 |
| [02-data-loading.md](02-data-loading.md) | 数据加载与增强：8 段采样、变换链、水平翻转 |
| [03-model-forward.md](03-model-forward.md) | 模型前向：张量形状、时序模块、CoordAtt 动态卷积核 |
| [04-train-loop.md](04-train-loop.md) | 训练回路：类别均衡采样、损失、日志、存档 |
| [05-eval-pipeline.md](05-eval-pipeline.md) | 评测流程：多 clip × 多 crop 集成 |
| [api/README.md](api/README.md) | **API 参考**：逐文件讲解每个类与函数 |

> 两组文档互补：上面 5 篇讲**数据怎么流动**，[api/](api/README.md) 讲**代码里有什么**。

## 关键配置（本仓库脚本实际使用值）

| 项 | 值 | 出处 |
|---|---|---|
| 数据集 | IPN-Hand，13 类 | `ops/dataset_config.py` `return_ipnhand` |
| 输入模态 | RGB | 训练脚本 `main.py ipnhand RGB` |
| 帧模板 | `{:06d}.jpg` | `return_ipnhand` 的 `prefix` |
| 时序段数 | `num_segments = 8` | 训练脚本 `--num_segments 8` |
| 每段帧数 | `new_length = 1`（RGB） | `main.py` `data_length` |
| 骨干网络 | ResNet-50（ImageNet 预训练） | `--arch resnet50` |
| 时序模块 | `--shift --shift_div=0.50 --shift_place=blockres` | 训练脚本 |
| 共识方式 | `avg`（8 段特征取平均） | `--consensus_type=avg` |
| 批大小 | 训练 16 / 测试 4 | `--batch-size` |
| 训练轮数 | 50，`lr_steps = 20 40` | `--epochs 50 --lr_steps 20 40` |
| partial BN | 关闭（`--npb`，全网络 BN 参与训练） | 训练脚本 |
| 测试 | `--test_segments=8 --test_crops=5 --multi_clip_test` | 测试脚本 |

## 张量形状速查

单位：`B` = 批大小，`S` = 时序段数（8），`C` = 通道数，`H×W` = 空间尺寸。

| 阶段 | 形状 | 说明 |
|---|---|---|
| 单条样本（Dataset 输出） | `[24, 224, 224]` | 8 段 × 3 通道，通道维由 `Stack` 拼接 |
| 训练批 | `[16, 24, 224, 224]` | `B=16` |
| 送入骨干前 | `[128, 3, 224, 224]` | `B×S = 16×8`，`view((-1, 3) + 尺寸)` |
| 骨干特征输出 | `[128, 13]` | 每帧一个 13 类预测（已过 `new_fc`） |
| 共识前 | `[16, 8, 13]` | 按段还原 |
| 共识后（模型输出） | `[16, 13]` | 8 段取平均 |
| 评测批（含 crop） | `[B×num_crop×8, 3, 224, 224]` | `num_crop` 见 [05](05-eval-pipeline.md) |

> 注意：RGB 模态下 `new_length = 1`，所以「一段」就是「一帧」。8 段即 8 帧，
> 通道维 `24 = 8 × 3` 是把时间维折叠进通道维的结果 —— 这是 TSM 系列的标志性做法，
> 好处是能直接复用 2D 卷积骨干。
