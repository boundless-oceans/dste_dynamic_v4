# 01 · 端到端总览

从 IPN-Hand 数据集的 JSON 标注到最终预测结果的完整链路。

## 训练路径

```mermaid
flowchart TB
    subgraph DS["数据源（磁盘）"]
        JSON["标注文件<br/>IPNHand/annotation_ipnhand/training.json<br/>validation.json"]
        FRAMES["已抽帧的图片目录<br/>每段手势一个目录<br/>000001.jpg, 000002.jpg ..."]
    end

    subgraph CFG["配置解析"]
        DC["ops/dataset_config.py<br/>return_ipnhand<br/>13 类 / 帧模板 {:06d}.jpg"]
        OPTS["opts.py<br/>argparse 命令行参数"]
        SN["拼接 store_name<br/>TSQ_ipnhand_RGB_resnet50_<br/>shift0.5_blockres_avg_segment8_e50"]
    end

    subgraph DATA["数据加载（CPU，多进程）"]
        DSET["ops/dataset.py<br/>TSNDataSet"]
        SAMP["采样 8 段<br/>_sample_indices"]
        LOAD["_load_image<br/>按索引读帧"]
        TF["变换链<br/>ops/transforms.py"]
        WS["WeightedRandomSampler<br/>类别均衡"]
        DL["DataLoader<br/>batch=16, workers=32"]
    end

    subgraph MODEL["模型（GPU）"]
        TSN["ops/models.py<br/>TSN 包装 ResNet-50"]
        TM["ops/temporal_module.py<br/>TemporalModule + CoordAtt"]
        CONS["ConsensusModule<br/>8 段特征 avg"]
    end

    subgraph LOSS["损失与优化"]
        CE["CrossEntropyLoss"]
        SGD["SGD + 分组学习率"]
        CLIP["clip_grad_norm_"]
    end

    subgraph OUT["产物"]
        CKPT["checkpoint/store_name/<br/>ckpt.pth.tar<br/>ckpt.best.pth.tar"]
        LOG["log/store_name/<br/>log.csv + args.txt"]
        TB["TensorBoard 事件文件<br/>loss / acc / lr 曲线"]
    end

    JSON --> DSET
    FRAMES --> LOAD
    DC --> DSET
    OPTS --> SN
    DSET --> SAMP --> LOAD --> TF --> DL
    DSET --> WS --> DL
    DL -->|"input [16, 24, 224, 224]"| TSN
    TSN --> TM --> CONS
    CONS -->|"output [16, 13]"| CE
    CE --> SGD --> CLIP
    CLIP -.->|"更新权重"| TSN
    CE --> TB
    CONS -->|"验证集评估"| LOG
    LOG --> CKPT
    TB --> CKPT

    classDef disk fill:#e8f4ea,stroke:#2e7d32,color:#1b5e20
    classDef cfg fill:#fff4e5,stroke:#ef6c00,color:#e65100
    classDef cpu fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef gpu fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef out fill:#fce4ec,stroke:#ad1457,color:#880e4f

    class JSON,FRAMES disk
    class DC,OPTS,SN cfg
    class DSET,SAMP,LOAD,TF,WS,DL cpu
    class TSN,TM,CONS gpu
    class CKPT,LOG,TB out
```

## 评测路径

```mermaid
flowchart LR
    subgraph IN["输入"]
        W["权重<br/>ckpt.best.pth.tar"]
        V["validation.json"]
        F["抽帧图片"]
    end

    subgraph EVAL["test_models.py"]
        P["parse_shift_option_from_log_name<br/>从权重文件名反解<br/>is_shift / shift_div / shift_place"]
        NET["重建 TSN 网络<br/>并 load_state_dict"]
        CROP["多 crop<br/>GroupOverSample ×5"]
        CLIP["多 clip<br/>每样本 10 组采样"]
        ENS["集成：50 个预测取平均"]
    end

    R["Prec@1 / Prec@5<br/>+ 混淆矩阵 cm.npy"]

    W --> P --> NET
    V --> CROP
    F --> CROP
    CROP --> CLIP --> ENS
    NET --> ENS
    ENS --> R

    classDef io fill:#e8f4ea,stroke:#2e7d32,color:#1b5e20
    class W,V,F,R io
```

## 两条路径的关键差异

| | 训练路径 | 评测路径 |
|---|---|---|
| 入口 | `main.py` | `test_models.py` |
| 采样 | 训练集随机采样；验证集居中采样 | 10 组随机采样（`--multi_clip_test`） |
| 空间裁剪 | 随机裁剪 + 颜色抖动 + RandomErasing | 5 crop 过采样（四角 + 中心） |
| 水平翻转 | 按概率翻转，并同步对调左右类标签 | 不翻转 |
| BN 行为 | `model.train()`，partial BN 关闭即全部更新 | `model.eval()`，用滑动统计量 |
| 集成 | 8 段特征 avg（`ConsensusModule`） | 50 个预测再取平均 |
| 输出 | 权重 + 日志 + TensorBoard | Prec@1 / Prec@5 + 混淆矩阵 |

## 模块与文件对应

```mermaid
flowchart LR
    M["main.py<br/>训练/验证入口"] --> D["ops/dataset.py<br/>TSNDataSet"]
    M --> MO["ops/models.py<br/>TSN"]
    M --> T["ops/transforms.py<br/>变换与增强"]
    M --> U["ops/utils.py<br/>AverageMeter / accuracy<br/>load_checkpoint"]
    M --> DC["ops/dataset_config.py<br/>路径与类别数"]
    MO --> B["ops/basic_ops.py<br/>ConsensusModule"]
    MO --> TM["ops/temporal_module.py<br/>TemporalModule / CoordAtt<br/>SpatialAttention"]
    MO --> NL["ops/non_local.py<br/>non-local block（默认未启用）"]
    E["test_models.py<br/>评测入口"] --> D
    E --> MO
    E --> DC
    E --> U
```

> 详细拆解见后续各篇：数据加载 [02](02-data-loading.md)、模型前向 [03](03-model-forward.md)、
> 训练回路 [04](04-train-loop.md)、评测流程 [05](05-eval-pipeline.md)。
