# 05 · 评测流程

对应 `test_models.py`。本仓库的评测脚本使用
`--test_segments=8 --test_crops=5 --multi_clip_test`，
即 **8 段 × 5 crop × 10 clip = 每个样本 400 次前向**，再对 50 个预测取平均。

## 从权重文件名反解网络结构

`test_models.py` 不接收架构参数，而是**从权重路径的字符串里解析**：

```mermaid
flowchart TB
    A["--weights=checkpoint/TSQ_ipnhand_RGB_<br/>resnet50_shift0.5_blockres_avg_segment8_e50/<br/>ckpt.best.pth.tar"] --> B["parse_shift_option_from_log_name"]
    B --> C["文件名含 'shift' → is_shift = True"]
    B --> D["解析 shift_div = 0.5"]
    B --> E["解析 shift_place = 'blockres'"]
    A --> F["'RGB' in 权重串 → modality = 'RGB'"]
    A --> G["split('TSQ_')[1].split('_')[2]<br/>→ this_arch = 'resnet50'"]
    A --> H["'_nl' in 权重串 → non_local 开关"]
    C --> I["构建 TSN(num_class, test_segments, ...)"]
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J["load_checkpoint(weights)<br/>取 checkpoint['state_dict']"]
    J --> K["剥离 DataParallel 的 'module.' 前缀<br/>'.'.join(k.split('.')[1:])"]
    K --> L["net.load_state_dict(base_dict)"]

    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class B,C,D,E,G note
```

> ⚠️ **这是评测链路最脆弱的一环**：文件名必须严格符合
> `TSQ_<数据集>_<模态>_<骨干>_shift<div>_<place>_<共识>_segment<N>_e<轮数>` 的格式。
> 重命名权重文件、或改成别的目录结构，都会让解析失败或解析出错误的 `shift_div`
> —— 而 `shift_div` 错了，权重加载会因形状不匹配而报错（这还算好的；
> 若只是 `shift_place` 解析错，可能加载成功但语义不一致）。
> 训练时 `args.store_name` 就是按这个格式拼的，所以只要权重是训练脚本产出的就安全。

## 两种集成维度

```mermaid
flowchart TB
    V["一个验证样本<br/>n_frames 帧的某段手势"] --> CLIP["时间维：10 个 clip<br/>--multi_clip_test"]
    V --> CROP["空间维：5 个 crop<br/>--test_crops=5"]
    V --> SEG["段维：8 段<br/>--test_segments=8"]

    CLIP --> C1["__getitem__ 里生成 80 个帧索引<br/>（10 组 × 8 段随机偏移）"]
    CROP --> C2["GroupOverSample<br/>左上/右上/左下/右下/中心<br/>flip=False"]
    SEG --> C3["ConsensusModule(avg)<br/>模型内部对 8 段取平均"]

    C1 --> TOT["每样本 10 × 5 × 8 = 400 次前向"]
    C2 --> TOT
    C3 --> TOT
    TOT --> AVG["50 个预测取平均<br/>（10 clip × 5 crop）"]

    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class C1,C2,C3 note
```

三者的分工：

| 维度 | 参数 | 处理位置 | 作用 |
|---|---|---|---|
| 段（时间） | `--test_segments=8` | **模型内部** `ConsensusModule` | 8 段特征平均，模型输出即 1 个预测 |
| clip（时间） | `--multi_clip_test` | **Dataset** `__getitem__` | 10 组不同的时间偏移，抵消采样随机性 |
| crop（空间） | `--test_crops=5` | **变换链** `GroupOverSample` | 5 个位置裁剪，抵消空间裁剪偏差 |

## 集成计算的位置

```mermaid
flowchart TB
    A["data: [B, num_crop × 8 × 3 × 224 × 224]"] --> B["data.view(-1, 3, 224, 224)"]
    B --> C["[B × num_crop × 8, 3, 224, 224]"]
    C --> D{"is_shift?"}
    D -->|是| E["view(B × num_crop, 8, 3, 224, 224)"]
    D -->|否| F["保持原状"]
    E --> G["net(data_in)"]
    F --> G
    G --> H["rst: [B × num_crop, 13]"]
    H --> I["reshape(B, num_crop, -1).mean(1)<br/>crop 维平均"]
    I --> J["[B, 13]"]
    J --> K{"--softmax?"}
    K -->|是| L["F.softmax(rst, dim=1)"]
    K -->|否| M["保持 logits"]
    L --> N["rst.reshape(B, 13)"]
    M --> N
    N --> O["返回 (i, rst, label)"]

    classDef shape fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class B,C,E,H,J,N shape
```

要点：`num_crop` 是**累乘**出来的，不是直接等于 `--test_crops`：

```python
num_crop = args.test_crops          # 5
if args.dense_sample: num_crop *= 10
if args.twice_sample: num_crop *= 2
if args.multi_clip_test: num_crop *= 10   # → 50
```

这解释了为什么 8 段是被折进 `num_crop` 之外的维度：
`data` 里每个 clip 就已经含 8 段 × 3 通道 = 24 个通道，
`view(B × num_crop, 8, 3, 224, 224)` 只是把 24 通道重新解释成 `8 × 3`。

## 多权重集成

`test_models.py` 支持一次评测多个权重并加权融合
（`--weights=a.pth,b.pth --test_segments=8,16 --coeff=1,0.5`）：

```mermaid
flowchart LR
    subgraph W1["权重 A"]
        A1["独立 net_list"] --> A2["独立 data_loader"]
    end
    subgraph W2["权重 B"]
        B1["独立 net_list"] --> B2["独立 data_loader"]
    end
    A2 --> M["zip(*data_iter_list)<br/>同步迭代"]
    B2 --> M
    M --> MUL["每个预测乘系数<br/>coeff_list"]
    MUL --> SUM["sum(rst_list) / len(rst_list)"]
    SUM --> ACC["accuracy(topk=(1,5))"]

    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class MUL,SUM note
```

本仓库脚本只用单个权重，所以 `zip` 里只循环一次，等价于直接取 `rst[1]`。

## 输出

```mermaid
flowchart TB
    A["评测结束"] --> B["逐批打印 Test: [i/N]<br/>Time / Loss / Prec@1 / Prec@5"]
    B --> C["汇总打印并写入 out 文件：<br/>Testing Results: Prec@1 ... Prec@5 ... Loss ..."]
    C --> D["混淆矩阵<br/>cm.npy"]

    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class D note
```

`scripts/test_tsq_ipnhand_rgb_8f.sh` 把 stdout 重定向到 `ipnhand_dete_test.out`，
所以最终数字通过 `tail -n 30 ipnhand_dete_test.out` 查看。

`cm.npy`（混淆矩阵）已在 `.gitignore` 中排除。README 的「性能表现」表格里应填写的
`Overall Prec@1` / `Overall Prec@5` 就取自这个输出文件。

## 与训练内验证的差异

训练时的 `validate()`（`main.py`）与 `test_models.py` 不是同一套流程：

| | `main.py` 的 `validate()` | `test_models.py` |
|---|---|---|
| 采样 | `_get_val_indices` 居中采样（1 组） | 10 组随机 clip |
| 空间 | 单次 `GroupCenterCrop` | 5 crop 过采样 |
| 增强 | `GroupNormalize` | 同 |
| 融合 | 仅 8 段 avg | 8 段 avg → 50 预测再平均 |
| 用途 | 每轮选 best 模型 | 最终报告数字 |

因此**测试脚本报出的数字通常略高于训练日志里的 Best Prec@1**，
两者不可直接比较。README 中的性能表建议统一以测试脚本为准。
