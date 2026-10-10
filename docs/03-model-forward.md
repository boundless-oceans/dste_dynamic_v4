# 03 · 模型前向与形状流转

对应 `ops/models.py`、`ops/temporal_module.py`、`ops/basic_ops.py`。

## 整体前向

```mermaid
flowchart TB
    IN["input [16, 24, 224, 224]<br/>B=16, 8 段 × 3 通道"] --> V["input.view((-1, sample_len) + 尺寸)<br/>sample_len = 3 × new_length = 3"]
    V --> B1["[128, 3, 224, 224]<br/>128 = 16 × 8（帧维被展平成批维）"]
    B1 --> RES["ResNet-50 骨干<br/>（layer1~4 的 conv1 被 TemporalModule 包裹）"]
    RES --> FEAT["[128, 2048]"]
    FEAT --> FC["Dropout(0.5) → new_fc<br/>Linear(2048, 13)"]
    FC --> LOGITS["[128, 13]<br/>每帧一个预测"]
    LOGITS --> RS["view((-1, num_segments, 13))"]
    RS --> C3["[16, 8, 13]"]
    C3 --> CONS["ConsensusModule(avg)<br/>对 dim=1 求均值"]
    CONS --> OUT["[16, 1, 13] → squeeze(1)"]
    OUT --> FINAL["output [16, 13]"]

    classDef shape fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class IN,V,B1,FEAT,LOGITS,C3,OUT,FINAL shape
```

关键理解：**TSM 把「时间」折叠进了批维**。
8 帧不是作为 5D 张量 `[B, 8, 3, H, W]` 送进 3D 卷积，而是展平成 `[B×8, 3, H, W]`
交给标准 2D ResNet。帧与帧之间的信息交换由 `TemporalModule` 在两个空间位置之间
「搬运」通道来实现 —— 这样就能直接复用 ImageNet 预训练的 2D 权重。

## TemporalModule 的插入位置

`make_temporal_module` 把每个残差块的 `conv1` 替换成 `TemporalModule(b.conv1, ...)`：

```mermaid
flowchart TB
    subgraph L1["layer1：4 个 block，特征图 56×56"]
        A1["block0 → i=0 → 插入<br/>in_h = 56"]
        A2["block1 → i=1"]
        A3["block2 → i=2 → 插入<br/>in_h = 56"]
        A4["block3 → i=3"]
    end
    subgraph L2["layer2：4 个 block，28×28"]
        B1["block0 → 插入<br/>in_h = 28"]
        B2["block1"]
        B3["block2 → 插入<br/>in_h = 28"]
        B4["block3"]
    end
    subgraph L3["layer3：6 个 block，14×14"]
        C1["block0 → 插入<br/>in_h = 14"]
        C2["block1"]
        C3["block2 → 插入"]
        C4["block3"]
        C5["block4 → 插入"]
        C6["block5"]
    end
    subgraph L4["layer4：3 个 block，7×7"]
        D1["block0 → 插入<br/>in_h = 7"]
        D2["block1"]
        D3["block2 → 插入<br/>in_h = 7"]
    end

    classDef ins fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef skip fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class A1,A3,B1,B3,C1,C3,C5,D1,D3 ins
    class A2,A4,B2,B4,C2,C4,C6,D2 skip
```

`make_temporal_module` 的规则（`place='blockres'`）：

- `n_round = 1`，但如果 `len(layer3) >= 23` 则设为 2。ResNet-50 的 `layer3` 只有 6 个 block，
  所以 `n_round` 保持 1，即**每个 block** 都插入。
- 实际插入数：`layer1(4) + layer2(4) + layer3(6) + layer4(3，按 n_round 折算 2) = 16`

`h_list` 正好 16 项，与插入顺序一一对应：

```python
h_list = [56,56,56,56,       # layer1 前 4 个
          28,28,28,28,       # layer2 4 个
          14,14,14,14,14,14, # layer3 6 个
          7,7]               # layer4 的 block0 与 block2
```

`TemporalModule` 用类变量 `count` 递增取用 `h_list[count]`，因此
`make_temporal_module` 开头必须重置它：

```python
def make_temporal_module(net, n_segment, ...):
    TemporalModule.count = 0   # 避免同一进程多次 build 时越界
```

> ⚠️ 改动 `arch` / 输入分辨率 / `num_segments` 会让 `h_list` 失配，
> 需要同步维护这张表，否则会 `IndexError` 或取到错的 `in_h`。

## TemporalModule.forward 详细数据流

```mermaid
flowchart TB
    X["x: [nt, c, h, w]<br/>nt = n_batch × n_segment"] --> NB["n_batch = nt // n_segment"]
    NB --> V1["view(n_batch, n_segment, c, h, w)<br/>.transpose(1, 2)<br/>→ [n, c, t, h, w]"]
    V1 --> SP["按 num_shift_channel 切成两半<br/>x = [x_shift, x_keep]"]
    SP --> X0["x[0]: [n, c_s, t, h, w]<br/>c_s = int(c × 0.5) = 通道的 50%"]

    X0 --> P1["permute(0,4,1,2,3)<br/>view(n×w, c_s, t, h)<br/>固定一个 w 列，观察 t×h"]
    X0 --> P2["permute(0,3,1,2,4)<br/>view(n×h, c_s, t, w)<br/>固定一个 h 行，观察 t×w"]

    P1 --> AT1["att_th: CoordAtt(n_segment, in_h, c_s, c_s)"]
    P2 --> AT2["att_tw: CoordAtt(n_segment, in_h, c_s, c_s)"]

    AT1 --> R1["view + permute<br/>还原成 [n, c_s, t, h, w]"]
    AT2 --> R2["view + permute<br/>还原成 [n, c_s, t, h, w]"]
    R1 --> ADD["x[0] = x1 + x2<br/>两个方向的时序注意力相加"]
    R2 --> ADD

    ADD --> CAT["torch.cat([x_att, x_keep], dim=1)<br/>→ [n, c, t, h, w]"]
    SP --> CAT
    CAT --> SA["spatial_attention<br/>SpatialAttention(c)"]
    SA --> BACK["transpose(1,2).view(nt, c, h, w)<br/>还原成 4D"]
    BACK --> NET["self.net(x)<br/>即原本的 conv1"]

    classDef shape fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class X,V1,X0,P1,P2,R1,R2,CAT,BACK shape
    class ADD,SA note
```

### 为什么是「固定 w 列 / 固定 h 行」

这是本仓库相对原始 TSM 的核心改动。原始 TSM 做的是 channel shift：
把一部分通道沿时间轴平移，实现零参数的帧间信息交换。

本仓库改为**注意力引导的移位**：把第 50% 的通道拿出来（`shift_div=0.50`），
分别从两个正交视角建模时序关系：

| 分支 | 张量组织 | 语义 |
|---|---|---|
| `att_th` | `[n×w, c_s, t, h]` | 固定某个宽度列，看「时间 × 高度」平面 |
| `att_tw` | `[n×h, c_s, t, w]` | 固定某个高度行，看「时间 × 宽度」平面 |

两个分支各自算出一份移位结果，相加后与未动的 50% 通道拼接。
这样既保留了 TSM「零额外分支、直接复用 2D 卷积」的轻量特性，
又让「往哪个时间方向移、移多少」由数据学出来，而不是固定的 ±1 帧。

## CoordAtt：用注意力算卷积核

`ops/temporal_module.py` 里改动最大的部分。它同时做两件事：
**算坐标注意力权重** 和 **生成动态 3×3 空间卷积核**。

```mermaid
flowchart TB
    X["x: [n, c, h, w]<br/>（实际是 n×w 或 n×h 折进批维）"] --> ID["identity = x"]
    X --> PH["pool_h: AdaptiveAvgPool2d((None,1))<br/>沿宽度池化 → [n, c, h, 1]"]
    X --> PW["pool_w: AdaptiveAvgPool2d((1,None))<br/>沿高度池化后再 permute → [n, c, w, 1]"]

    PH --> KH["squeeze(3).reshape(n×c, -1)<br/>k_h: [nc, h]"]
    PW --> KW["squeeze(3).reshape(n×c, -1)<br/>k_w: [nc, w]"]

    PH --> CAT["torch.cat([x_h, x_w], dim=2)<br/>y: [n, c, h+w, 1]"]
    PW --> CAT
    CAT --> C1["conv1: Conv2d(c, mip, 1×1)<br/>mip = max(8, c // 4)"]
    C1 --> BN1["bn1 + h_swish"]
    BN1 --> SPL["split([h, w], dim=2)<br/>拆回两个方向"]

    SPL --> AH["conv_h → sigmoid<br/>a_h: 高度方向注意力"]
    SPL --> AW["conv_w → sigmoid → permute<br/>a_w: 宽度方向注意力"]

    KH --> FC1["fc1: Linear(h, 2h) → ReLU<br/>→ Linear(2h, 3) → Softmax<br/>k_h: [nc, 3]"]
    KW --> FC2["fc2: Linear(w, 2w) → ReLU<br/>→ Linear(2w, 3) → Softmax<br/>k_w: [nc, 3]"]

    FC1 --> RK["view(nc, 1, 3, 1)"]
    FC2 --> RW["view(nc, 1, 1, 3)"]
    RK --> KER["外积 conv_kernel = k_h × k_w<br/>[nc, 1, 3, 3]"]
    RW --> KER

    ID --> MUL["out = identity × a_h × a_w"]
    AH --> MUL
    AW --> MUL

    MUL --> CONV["F.conv2d(<br/>out.reshape(1, n×c, h, w),<br/>conv_kernel,<br/>padding=(1,1),<br/>groups=n×c<br/>)<br/>逐通道深度卷积"]
    KER --> CONV
    CONV --> RS["reshape(n, c, h, w)"]
    RS --> BN2["bn2: BatchNorm2d + ReLU"]
    BN2 --> OUT["输出 [n, c, h, w]"]

    classDef att fill:#e8f4ea,stroke:#2e7d32,color:#1b5e20
    classDef ker fill:#fce4ec,stroke:#ad1457,color:#880e4f
    classDef shape fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class AH,AW,MUL att
    class FC1,FC2,RK,RW,KER,CONV ker
    class X,KH,KW,CAT,OUT shape
```

### 三个技术要点

**1. 坐标注意力（原版 CoordAtt 思路）**
不直接把特征图压成 `[n, c, 1, 1]` 的全局向量，而是分别沿高度、宽度池化，
得到 `[n, c, h, 1]` 和 `[n, c, w, 1]`。这样注意力保留了**位置信息** ——
对细长的手势（如「向左划」）来说，手的位置和运动方向都重要。
两个方向共享 `conv1` 压缩，再各自 `sigmoid` 得到 `a_h`、`a_w`。

**2. 动态卷积核（本仓库改动）**
`fc1` / `fc2` 把池化后的 `[nc, h]`、`[nc, w]` 映射到 3 个值，`Softmax` 归一化，
再外积成 `[nc, 1, 3, 3]`。也就是说**每个样本、每个通道都有自己的一组 3×3 卷积核**，
由输入内容实时生成。

配合 `groups=n×c` 的深度卷积，等价于「逐通道做一次可变形的小邻域聚合」：

```python
out = F.conv2d(
    out.reshape(1, n * c, h, w),   # 把 n×c 折进批维，当成一个 batch
    conv_kernel,                   # [nc, 1, 3, 3]，每组核对应一个通道
    padding=(1, 1),
    groups=n * c                   # 分组数 = 通道数 → 深度卷积
)
```

**3. 与原始实现的差异**
原始版本用固定的深度可分离卷积 `conv_th` / `conv_tw`（3×3，参数静态），
本仓库把它们注释掉，换成上面的动态核方案。带来的变化：

- 参数量：新增两个 `Linear`（`2h²+6h` 量级），去掉了两个固定的 3×3 深度卷积
- 表达能力：卷积核随输入变化，对「不同手势、不同运动幅度」更自适应
- 约束：`in_h` 必须与特征图实际高度一致，因此需要 `h_list` 这张表

### SpatialAttention（补充分支）

`TemporalModule` 末尾还有一个 `SpatialAttention`：

```mermaid
flowchart LR
    A["x: [n, c, t, h, w]"] --> P1["max(dim=2)"]
    A --> P2["mean(dim=2)"]
    P1 --> CAT["concat → [n, 2, h, w]"]
    P2 --> CAT
    CAT --> C["Conv2d(2, 1, 7×7, padding=3)<br/>+ sigmoid"]
    C --> ATTN["attn: [n, 1, h, w]"]
    A --> RES["identity + identity × attn<br/>残差形式"]
    ATTN --> RES
```

它在时间维上池化，得到一个纯空间的 7×7 大核注意力，
以残差形式叠加（`identity + identity × attn`），让模型关注手部所在的区域。

## ConsensusModule：8 段如何合成一个预测

```mermaid
flowchart LR
    A["[16, 8, 13]<br/>8 段各自的 13 类 logits"] --> B["SegmentConsensus(avg)<br/>input.mean(dim=1, keepdim=True)"]
    B --> C["[16, 1, 13]"]
    C --> D["squeeze(1)"]
    D --> E["[16, 13]<br/>最终输出"]

    classDef shape fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    class A,C,E shape
```

`basic_ops.py` 里还支持 `identity` 模式（不聚合），本仓库用 `avg`。
注意 `SegmentConsensus` 的 `dim` 默认为 1，与上面的 `[B, S, num_class]` 布局一致。

## 参数分组：为什么学习率不是一刀切

`get_optim_policies` 把所有参数按类型分成 9 组，每组有独立的 `lr_mult` / `decay_mult`：

```mermaid
flowchart TB
    P["model.parameters()"] --> A["第一个 Conv2d 的 weight<br/>lr_mult=1（Flow 时 5）"]
    P --> B["第一个 Conv2d 的 bias<br/>lr_mult=2, decay_mult=0"]
    P --> C["其余 Conv/Linear weight<br/>lr_mult=1"]
    P --> D["其余 bias<br/>lr_mult=2, decay_mult=0"]
    P --> E["BN scale/shift<br/>lr_mult=1, decay_mult=0"]
    P --> F["GN 参数<br/>lr_mult=1, decay_mult=0"]
    P --> G["new_fc（fc_lr5 时）<br/>lr_mult=5, decay_mult=1<br/>bias lr_mult=10"]

    classDef note fill:#f5f5f5,stroke:#9e9e9e,color:#424242
    class B,D,E,F,G note
```

主要考虑：

- **bias 不做 weight decay**（`decay_mult=0`）：偏置衰减到 0 会削弱表达能力。
- **BN 的 scale/shift 不做 weight decay**：同理。
- **分类头用更大学习率**（`lr_mult=5`）：`new_fc` 是随机初始化的，而骨干是预训练权重，
  让新层学得快一些。本仓库脚本传了 `--npb`（关闭 partial BN），
  同时 `fc_lr5` 在未指定 `--tune_from` 时为真。
