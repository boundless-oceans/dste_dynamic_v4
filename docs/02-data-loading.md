# 02 · 数据加载与增强

对应 `ops/dataset.py` 与 `ops/transforms.py`。IPN-Hand 的数据是**已抽帧的图片**，
不是视频文件，因此每次取样本都是「按索引开图」。

## 标注解析：从 JSON 到样本列表

```mermaid
flowchart TB
    A["training.json / validation.json"] --> B["load_annotation_data<br/>json.load"]
    B --> C["get_class_labels<br/>labels → class_labels_map<br/>{类名: 索引}"]
    B --> D["get_video_names_and_annotations<br/>遍历 database<br/>key.split('^')[0] 取视频名"]
    C --> E["index_to_class<br/>{索引: 类名}"]
    D --> F{"逐条标注"}
    E --> F
    F --> G["video_path = os.path.join(root, 视频名)"]
    G --> H{"os.path.exists<br/>目录存在?"}
    H -->|否| I["打印 video path not exist!!!<br/>跳过该条"]
    H -->|是| J["start_frame / end_frame<br/>n_frames = end − start + 1"]
    J --> K["样本字典"]
    K --> L["dataset 列表<br/>+ index_to_class 作为 class_names"]

    classDef disk fill:#e8f4ea,stroke:#2e7d32,color:#1b5e20
    classDef warn fill:#fff4e5,stroke:#ef6c00,color:#e65100
    class A disk
    class I warn
```

产出的样本字典 `sample`：

| 字段 | 含义 |
|---|---|
| `video_path` | 该段手势的帧目录绝对路径 |
| `segment` | `[start_frame, end_frame]` |
| `start_frame` / `end_frame` | 起止帧号（JSON 里是浮点字符串，转 int） |
| `n_frames` | 有效帧数 |
| `video_id` | 在 dataset 中的序号 |
| `label` | 类别索引，由 `class_to_index` 映射得到 |
| `frame_indices` | `range(start, end+1)` 列表 |

> ⚠️ 关键点：标注描述的是**一段手势在长视频中的时间区间**，而不是一整段视频。
> 所有采样都在 `[start_frame, end_frame]` 内进行，最后要加回 `start_frame` 偏移。
> `ops/dataset.py` 里的关键行：`return offsets + record['start_frame']`。

## 采样：如何从 n_frames 里挑出 8 帧

```mermaid
flowchart TB
    A["__getitem__(index)<br/>record = self.data[index]"] --> B{"test_mode?"}

    B -->|"否（训练/验证）"| C{"random_shift?"}
    C -->|"是 → 训练集"| D["_sample_indices<br/>随机采样"]
    C -->|"否 → 验证集"| E["_get_val_indices<br/>居中采样"]

    B -->|"是 → 测试集"| F{"multi_clip_test?"}
    F -->|"是"| G["10 组随机偏移<br/>拼接成 80 个索引"]
    F -->|"否"| H["_get_test_indices<br/>居中采样"]

    D --> Z["segment_indices"]
    E --> Z
    G --> Z
    H --> Z

    Z --> Y["get(record, segment_indices)"]

    subgraph SD["_sample_indices 内部"]
        D1["average_duration =<br/>(n_frames − new_length + 1) // num_segments"]
        D2{"average_duration > 0 ?"}
        D3["每段随机偏移：<br/>段号 × average_duration<br/>+ randint(average_duration)"]
        D4{"n_frames > num_segments ?"}
        D5["randint 后排序"]
        D6["全 0（帧数不足，退化为首帧复制）"]
        D1 --> D2
        D2 -->|是| D3
        D2 -->|否| D4
        D4 -->|是| D5
        D4 -->|否| D6
    end

    classDef train fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef val fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef warn fill:#fff4e5,stroke:#ef6c00,color:#e65100
    class D,D3,D5 train
    class E,H val
    class D6 warn
```

三种采样策略的对比：

| 策略 | 触发条件 | 偏移计算 | 用途 |
|---|---|---|---|
| 随机采样 | 训练集（`random_shift=True`） | `段号 × 平均时长 + randint(平均时长)` | 时间抖动，等价于时序增强 |
| 居中采样 | 验证集（`random_shift=False`） | `tick/2 + tick × 段号`，`tick = n_frames/8` | 确定性，便于比较 |
| 多 clip | 测试集 `--multi_clip_test` | 上述随机偏移重复 10 次，拼接 80 个索引 | 评测时降低采样方差 |

## 取帧：索引到张量

```mermaid
flowchart TB
    A["segment_indices<br/>如 [12, 25, 31, ...]（8 个）"] --> B["for seg_ind in indices<br/>p = int(seg_ind)"]
    B --> C["for i in range(new_length)<br/>RGB 时 new_length = 1"]
    C --> D["_load_image(video_path, p)"]
    D --> E["Image.open(<br/>root/视频名/{:06d}.jpg<br/>).convert('RGB')"]
    E --> F{"读取成功?"}
    F -->|否| G["打印 error loading image<br/>回退到第 1 帧"]
    F -->|是| H["PIL.Image"]
    G --> H
    H --> I["images.extend(...)<br/>累积成 8 张 PIL 图"]
    I --> J["transform(images)<br/>见下方变换链"]
    J --> K["process_data<br/>torch.Tensor [24, 224, 224]"]
    K --> L{"_do_hflip<br/>且 rand < 0.5?"}
    L -->|是| M["torch.flip(dims=[-1])<br/>最后一维即宽度 → 水平镜像"]
    L -->|否| N["保持"]
    M --> O{"label 在左右配对中?"}
    O -->|是| P["对调标签<br/>left ↔ right"]
    O -->|否| Q["标签不变"]
    N --> R["return (张量, label)"]
    P --> R
    Q --> R

    classDef warn fill:#fff4e5,stroke:#ef6c00,color:#e65100
    class G warn
```

`torch.flip(dims=[-1])` 之所以是水平翻转：张量布局为 `[C, H, W]`，最后一维是宽度。
整段 8 帧在 `Stack` 之后已经拼成同一个张量，所以一次 flip 就是**8 帧一起镜像**，
不会出现帧间翻转不一致的问题。

## 变换链

```mermaid
flowchart TB
    subgraph TR["训练变换链（main.py）"]
        T1["train_augmentation<br/>GroupRandomSizedCrop / GroupRandomCrop<br/>+ GroupScale"]
        T2["GroupColorJitter<br/>brightness=0.2, contrast=0.2<br/>同组帧共享同一随机参数"]
        T3["Stack<br/>8 张 HWC 图 → 拼成 [H, W, 24]"]
        T4["ToTorchFormatTensor<br/>HWC → CHW，除以 255"]
        T5["GroupNormalize<br/>ImageNet mean/std"]
        T6["RandomErasing p=0.2<br/>（torchvision 原生，逐帧独立）"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6
    end

    subgraph VA["验证/测试变换链"]
        V1["GroupScale(scale_size)"]
        V2["GroupCenterCrop(crop_size)"]
        V3["Stack"]
        V4["ToTorchFormatTensor"]
        V5["GroupNormalize"]
        V1 --> V2 --> V3 --> V4 --> V5
    end

    subgraph TE["测试专用裁剪（test_models.py）"]
        E1["test_crops=5 →<br/>GroupOverSample(input_size, scale_size, flip=False)<br/>四角 + 中心，不翻转"]
        E2["test_crops=10 →<br/>GroupOverSample(...) 含翻转"]
    end

    A["PIL 图列表<br/>8 张 [640×480]"] --> TR
    A --> VA
    A --> TE

    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class T2,T6 note
```

### 为什么 Group* 前缀很重要

`Group` 系列变换（`GroupRandomCrop`、`GroupScale`、`GroupColorJitter`、`GroupNormalize`…）
对**同一段内的所有帧施加完全相同的随机参数**。如果用 torchvision 原生的逐帧变换，
8 帧会各自独立随机裁剪/调色，等于给时序通道注入噪声，破坏运动信息。

这也是 `GroupColorJitter` 被单独实现的原因 —— 原生 `ColorJitter` 不支持 group 语义：

```python
class GroupColorJitter(object):
    def __call__(self, img_group):
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        # 参数在循环外抽一次，组内 8 帧共用
        for img in img_group:
            img = ImageEnhance.Brightness(img).enhance(b)
            img = ImageEnhance.Contrast(img).enhance(c)
```

> 例外：`RandomErasing` 是逐帧独立的。它作用在已归一化的张量上，模拟局部遮挡，
> 逐帧独立反而是一种正则化，不破坏整体运动方向。

## 水平翻转与标签对调

这是本仓库相对上游 TSM 的一个重要改动，逻辑在 `_find_lr_pair`：

```mermaid
flowchart TB
    A["class_names<br/>{索引: 类名}"] --> B["遍历所有类名<br/>小写化"]
    B --> C{"类名含 'left' / 'right'<br/>或等于 'g05' / 'g06'?"}
    C -->|是| D["加入候选集 idxs"]
    C -->|否| E["跳过"]
    D --> F{"len(idxs) == 2 ?"}
    F -->|是| G["返回 (左类索引, 右类索引)"]
    F -->|否| H["返回 None"]

    G --> I["_do_hflip = True<br/>打印：水平翻转已启用"]
    H --> J["_do_hflip = False<br/>打印 WARNING：不做水平翻转"]

    classDef ok fill:#e8f4ea,stroke:#2e7d32,color:#1b5e20
    classDef warn fill:#fff4e5,stroke:#ef6c00,color:#e65100
    class G,I ok
    class H,J warn
```

设计要点：**宁可不翻转，也不翻转错标签**。

镜像是左右互换的：一个「向右」的样本翻转后视觉上就是「向左」。
如果只翻转图像而不改标签，模型就会学到「向右 = 左手向右划」这种错误监督，
而且这种错误是**静默的**（loss 正常下降，只有方向类准确率上不去）。

所以只有**恰好**识别出两个方向类时才启用翻转；类名规则对不上就整体关闭，
并打印 WARNING 提示补充规则。

## DataLoader 组装

```mermaid
flowchart LR
    subgraph TRL["train_loader"]
        A["train_dataset"] --> B["WeightedRandomSampler<br/>权重 = 1 / 该类样本数<br/>replacement=True"]
        B --> C["batch_size=16<br/>shuffle 由 sampler 接管<br/>drop_last=True"]
        C --> D["worker_init_fn=_init_worker<br/>每 worker 种子 = 1 + worker_id"]
    end

    subgraph VLL["val_loader"]
        E["val_dataset<br/>random_shift=False"] --> F["shuffle=False<br/>顺序确定"]
        F --> G["batch_size=16"]
    end

    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class B,D note
```

- `drop_last=True`：丢弃不足一个批的尾部，避免多卡 DataParallel 下批大小不整除。
- `worker_init_fn`：给每个 worker 设 `random` / `numpy` / `torch` 种子，保证多进程
  数据增强可复现。注意 `main.py` 里还统一设了全局种子（`torch.manual_seed(1)` 等）。
- `WeightedRandomSampler` 与 `shuffle=True` 互斥 —— 用了 sampler 就不能再开 shuffle。

## 小结：一条训练样本的完整旅程

```mermaid
sequenceDiagram
    autonumber
    participant DL as DataLoader worker
    participant DS as TSNDataSet
    participant FS as 磁盘（抽帧图片）
    participant TF as 变换链
    participant GPU as GPU 批

    DL->>DS: __getitem__(index)
    DS->>DS: record = self.data[index]
    DS->>DS: _sample_indices → 8 个帧索引
    loop 8 个段
        DS->>FS: Image.open(.../{:06d}.jpg)
        FS-->>DS: PIL.Image (RGB)
    end
    DS->>TF: transform(8 张 PIL 图)
    TF->>TF: 随机裁剪 / 颜色抖动（组内同参）
    TF->>TF: Stack → [H, W, 24]
    TF->>TF: ToTorchFormatTensor → [24, H, W]
    TF->>TF: GroupNormalize + RandomErasing
    TF-->>DS: [24, 224, 224]
    DS->>DS: 按 0.5 概率水平翻转 + 标签对调
    DS-->>DL: (张量, label)
    DL->>GPU: 拼批 → [16, 24, 224, 224]
```
