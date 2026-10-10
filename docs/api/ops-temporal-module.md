# `ops/temporal_module.py` — 时序模块与坐标注意力

408 行，**本仓库改动最大的文件**。核心思想：把「时间」折叠进通道维，
在两个正交空间视角上做注意力引导的时序建模，从而复用 2D 卷积骨干。

```python
from ops.temporal_module import TemporalModule, CoordAtt, make_temporal_module, make_temporal_pool
```

## 模块内清单

| 名称 | 类型 | 本仓库是否改动 |
|---|---|---|
| `h_sigmoid` / `h_swish` | 激活函数 | 否（上游） |
| `CoordAtt` | 坐标注意力 + 动态卷积核 | ⭐ **改造** |
| `SpatialAttention` | 空间自注意力 | ⭐ **新增** |
| `h_list` | 各插入点特征图高度表 | ⭐ **新增** |
| `TemporalModule` | 时序模块（替换 ResNet 的 conv1） | ⭐ **改造** |
| `make_temporal_module` | 把 TemporalModule 注入 ResNet | ⭐ **新增计数器重置** |
| `make_temporal_pool` / `TemporalPool` | 时序池化（默认未启用） | 否 |

---

## `class h_sigmoid(nn.Module)`

```python
def forward(self, x): return self.relu(x + 3) / 6      # relu = nn.ReLU6(inplace)
```

Hard-Sigmoid 的近似实现，取值范围 `[0, 1]`。比标准 `sigmoid` 便宜，是 MobileNetV3/CoordAtt 系的标配。

### `class h_swish(nn.Module)`

```python
self.sigmoid = h_sigmoid(inplace=inplace)
def forward(self, x): return x * self.sigmoid(x)
```

Hard-Swish：`x × h_sigmoid(x)`。用作 `CoordAtt` 里 `conv1` 之后的激活。

---

## `class CoordAtt(nn.Module)` ⭐ 本仓库核心改造

### `__init__(in_h, in_w, inp, oup, reduction=4)`

| 参数 | 说明 |
|---|---|
| `in_h` | 输入特征图的**高度**，用于生成动态卷积核 |
| `in_w` | 实际传入的是 `self.n_segment`，即**时间长度** |
| `inp` / `oup` | 输入 / 输出通道数（本仓库相等） |
| `reduction` | 通道压缩比，`mip = max(8, inp // 4)` |

**子模块**：

| 名称 | 结构 | 用途 |
|---|---|---|
| `pool_h` | `AdaptiveAvgPool2d((None, 1))` | 沿宽度池化 → `[n, c, h, 1]` |
| `pool_w` | `AdaptiveAvgPool2d((1, None))` | 沿高度池化 → 再 permute 成 `[n, c, w, 1]` |
| `conv1` + `bn1` + `act` | `Conv2d(inp, mip, 1×1)` + BN + h_swish | 两方向共享的压缩分支 |
| `conv_h` / `conv_w` | `Conv2d(mip, oup, 1×1)` | 生成两个方向的注意力 |
| `fc1` | `Linear(in_h, 2·in_h) → ReLU → Linear(2·in_h, 3) → Softmax` | 生成高度方向的核 |
| `fc2` | `Linear(in_w, 2·in_w) → ReLU → Linear(2·in_w, 3) → Softmax` | 生成时间方向的核 |
| `bn2` | `BatchNorm2d(oup) + ReLU` | 输出前归一化 |

> ⚠️ **`in_w` 的命名与实际含义不一致**：调用处传的是 `self.n_segment`（时间长度 8），
> 不是特征图宽度。见 [TemporalModule](#class-temporalmodulennmodule-本仓库改造) 的说明。

### `forward(x)`

输入 `[n, c, h, w]`，输出同形状。两条并行支路：

**支路 A — 坐标注意力**

```python
x_h = self.pool_h(x)                      # [n, c, h, 1]
x_w = self.pool_w(x).permute(0,1,3,2)     # [n, c, w, 1]
y = torch.cat([x_h, x_w], dim=2)          # [n, c, h+w, 1]
y = self.act(self.bn1(self.conv1(y)))
x_h, x_w = torch.split(y, [h, w], dim=2)
a_h = self.conv_h(x_h).sigmoid()          # 高度方向注意力
a_w = self.conv_w(x_w).sigmoid()          # 宽度方向注意力
```

分别沿两个方向池化（保留位置信息，而非压成全局向量），共享 `conv1` 压缩后各自出注意力。

**支路 B — 动态卷积核**（本仓库新增）

```python
k_h = x_h.squeeze(3).reshape(n * c, -1)   # [nc, h]
k_w = x_w.squeeze(3).reshape(n * c, -1)   # [nc, w]
k_h = self.fc1(k_h).view(n*c, 1, 3, 1)    # [nc, 1, 3, 1]
k_w = self.fc2(k_w).view(n*c, 1, 1, 3)    # [nc, 1, 1, 3]
conv_kernel = k_h * k_w                   # [nc, 1, 3, 3]  ← 外积
```

`fc1`/`fc2` 把池化结果映射到 3 个值并 `Softmax` 归一化，再外积成 3×3 核。
**每个样本、每个通道都有自己的一组核，由输入内容实时生成。**

**合成**：

```python
out = identity * a_w * a_h                # 先施加坐标注意力
out = F.conv2d(out.reshape(1, n*c, h, w), conv_kernel,
               padding=(1,1), groups=n*c)  # 再用动态核做深度卷积
out = self.bn2(out.reshape(n, c, h, w))
```

`groups = n*c` 即**逐通道深度卷积**：把 `n×c` 折进批维，每组核对应一个通道。

**与上游实现的差异**：上游用两个固定的深度可分离卷积 `conv_th` / `conv_tw`（参数静态），
本仓库将其注释掉（保留在 `__init__` 与 `forward` 中作为参照），换成上述动态核方案。
代码里还保留了两版被完全注释的旧 `CoordAtt` 实现（第 106–226 行），供对比查阅。

---

## `class SpatialAttention(nn.Module)` ⭐ 本仓库新增

**时间差分 + 空间自注意力**。注意：它**不是**卷积注意力，而是标准的多头自注意力结构
（单头，Q/K/V 均为逐通道深度 1×1 卷积）。

### `__init__(in_channels)`

| 子模块 | 说明 |
|---|---|
| `norm` | `GroupNorm(1, in_channels)`（等价 LayerNorm 式的整样本归一化） |
| `q` / `k` / `v` | `Conv2d(c, c, 1×1, groups=c)` 深度投影，各自生成查询/键/值 |
| `scaler` | `c ** -0.5`，点积缩放因子 |
| `softmax` | `Softmax(-1)` |
| `dropout` | `Dropout(0.1)` |

### `forward(x)` — 输入 `[n, c, t, h, w]`

```python
identity = x
x = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]   # ① 时间差分 [n, c, t-1, h, w]
x = torch.mean(x, dim=2)                     # ② 时间平均 → [n, c, h, w]
x = self.norm(x)                             # ③ 归一化
q = self.q(x).view(n, c, -1).permute(0, 2, 1)   # [n, hw, c]
k = self.k(x).view(n, c, -1)                    # [n, c, hw]
v = self.v(x).view(n, c, -1).permute(0, 2, 1)   # [n, hw, c]
attn = torch.matmul(q, k) * self.scaler         # ④ [n, hw, hw] 空间自注意力
attn = self.dropout(self.softmax(attn))
attn = torch.matmul(attn, v)                    # [n, hw, c]
attn = torch.mean(attn, dim=2).view(n, 1, 1, h, w)   # ⑤ → 单个空间权重图
return identity + identity * attn               # ⑥ 残差形式施加
```

**逐段含义**：

| 步骤 | 作用 |
|---|---|
| ① 时间差分 | 提取**相邻帧的变化**，等价于廉价的光流近似 |
| ② 时间平均 | 把整段的运动信息聚合成一张空间图 |
| ③④ 空间自注意力 | 让每个空间位置与其他所有位置交互，`hw × hw` 的注意力矩阵 |
| ⑤ 通道维平均 | 压成单个空间显著性图 `[n, 1, 1, h, w]` |
| ⑥ 残差 | `identity + identity × attn`，初始时近恒等映射，训练稳定 |

**设计意图**：用「帧间差」而非原始帧来算注意力，让模型关注**运动剧烈的区域**（即手部），
这与手势识别的需求直接对应。

---

## `h_list` ⭐ 本仓库新增

```python
h_list = [56,56,56,56, 28,28,28,28, 14,14,14,14,14,14, 7,7]
```

**各 `TemporalModule` 插入点处特征图的高度**，16 项，按注入顺序排列。

| 分组 | 特征图 | 项数 | 对应 |
|---|---|---|---|
| 前 4 项 | 56×56 | 4 | `layer1` 的 4 个 bottleneck |
| 中 4 项 | 28×28 | 4 | `layer2` 的 4 个 |
| 6 项 | 14×14 | 6 | `layer3` 的 6 个 |
| 末 2 项 | 7×7 | 2 | `layer4` 的第 1、3 个 |

**为什么需要它**：`CoordAtt.fc1` 需要知道特征图高度来构造 `Linear(in_h, 2·in_h)`，
而 `TemporalModule` 在 `__init__` 时还看不到实际输入，只能靠这张查表。

**维护要求**：

> ⚠️ 改 `--arch`、输入分辨率或 `--num_segments` 都会让这张表失配。
> 表现是 `IndexError`（越界）或取到错误的 `in_h` 导致 `Linear` 维度不匹配。

---

## `class TemporalModule(nn.Module)` ⭐ 本仓库改造

包装一层原始 `conv1`，在其前面插入时序建模。

### 类变量 `count`

```python
count = 0
```

**全局插入计数器**，每次 `__init__` 时取 `h_list[count]` 并自增。
因为是类变量，多次构建模型（如 `--resume` 后重建）会让它累积越界，
所以 `make_temporal_module` 开头必须重置（见下）。

### `__init__(net, n_segment, n_div)`

| 参数 | 说明 |
|---|---|
| `net` | 被包裹的原始模块（实际是 `bottleneck.conv1`） |
| `n_segment` | 时序段数（8） |
| `n_div` | 参与时序建模的通道比例（0.5） |

```python
self.in_h = h_list[TemporalModule.count]
TemporalModule.count += 1
print('=> in_channels: {}, in_h: {}'.format(self.in_channels, self.in_h))

self.num_shift_channel = int(self.in_channels * self.n_div)     # 通道数的 50%
if self.num_shift_channel != 0:
    self.split_sizes = [self.num_shift_channel, self.in_channels - self.num_shift_channel]
    self.att_th = CoordAtt(self.n_segment, self.in_h, self.num_shift_channel, self.num_shift_channel)
    self.att_tw = CoordAtt(self.n_segment, self.in_h, self.num_shift_channel, self.num_shift_channel)
self.spatial_attention = SpatialAttention(self.in_channels)
```

**注意 `CoordAtt` 的实参顺序**：签名是 `(in_h, in_w, inp, oup)`，
但调用传的是 `(self.n_segment, self.in_h, c, c)` ——
所以 `in_h` 收到 8（时间长度），`in_w` 收到特征图高度。

结合 `forward` 里张量的实际布局（`[n, c, t, h]` 或 `[n, c, t, w]`），
`AdaptiveAvgPool2d((None,1))` 池化掉的是**最后一维（空间维 h 或 w）**，
结果 `[n, c, t, 1]` 恰好对应 `fc1` 期望的 `in_h = t = 8`。
**参数命名与实际语义是错位的，但数值是自洽的** —— 阅读时以形状为准。

`num_shift_channel == 0` 时（如 `n_div` 极小）跳过全部时序建模，只做空间注意力。

### `_initialize_weights()`

对 Conv/BN 做 Kaiming 风格初始化。**注意：本仓库没有调用它** ——
`__init__` 末尾没有调用，且被包裹的 `conv1` 自带 ImageNet 预训练权重，
新增的 `CoordAtt` / `SpatialAttention` 由 PyTorch 默认初始化。
保留此方法属于上游遗留。

### `forward(x)` — 输入 `[nt, c, h, w]`

```python
nt, c, h, w = x.size()
n_batch = nt // self.n_segment
if self.num_shift_channel != 0:
    x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2).contiguous()  # [n, c, t, h, w]
    x = list(x.split(self.split_sizes, dim=1))        # 切成 [时序通道, 保留通道]

    x1 = x[0].permute(0,4,1,2,3).view(n_batch*w, c_s, t, h)   # 固定 w，看 t×h
    x1 = self.att_th(x1)
    x1 = x1.view(n_batch, w, c_s, t, h).permute(0,2,3,4,1)

    x2 = x[0].permute(0,3,1,2,4).view(n_batch*h, c_s, t, w)   # 固定 h，看 t×w
    x2 = self.att_tw(x2)
    x2 = x2.view(n_batch, h, c_s, t, w).permute(0,2,3,1,4)

    x[0] = x1 + x2
    x = torch.cat(x, dim=1)

x = self.spatial_attention(x)
x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
return self.net(x)
```

**两个正交分支**：

| 分支 | 张量组织 | 语义 |
|---|---|---|
| `att_th` | `[n×w, c_s, t, h]` | 固定某个宽度列，观察「时间 × 高度」平面 |
| `att_tw` | `[n×h, c_s, t, w]` | 固定某个高度行，观察「时间 × 宽度」平面 |

两分支结果相加后，与未参与时序建模的另外 50% 通道 `cat` 回去，
再过 `SpatialAttention`，最后交给被包裹的原始 `conv1`。

**为什么这样设计**：TSM 原始做法是把通道沿时间轴平移 ±1 帧（零参数）。
本仓库改为**由数据学出该往哪个方向、移多少**，同时保留「不引入 3D 卷积、
直接复用 2D 预训练权重」的轻量特性。

---

## `make_temporal_module(net, n_segment, n_div=8, place='blockres', temporal_pool=False)`

**把 `TemporalModule` 注入 ResNet 的每个残差块。** 在 `TSN._prepare_base_model` 里被调用。

```python
TemporalModule.count = 0        # ⭐ 本仓库新增：重置全局计数器
if temporal_pool:
    n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
else:
    n_segment_list = [n_segment] * 4

if isinstance(net, torchvision.models.ResNet):
    if 'blockres' in place:
        n_round = 1
        if len(list(net.layer3.children())) >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))

        def make_block_temporal(stage, this_segment):
            blocks = list(stage.children())
            for i, b in enumerate(blocks):
                if i % n_round == 0:
                    blocks[i].conv1 = TemporalModule(b.conv1, n_segment=this_segment, n_div=n_div)
            return nn.Sequential(*blocks)

        net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
        net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
        net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
        net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
```

**`TemporalModule.count = 0` 是必须的**：`count` 是类变量，同一进程内若构建两次模型
（如 `--resume` 路径下重建、或 `test_models.py` 里多权重循环），
不清零就会累加并越过 `h_list` 的 16 项而 `IndexError`。
`main.py` 的启动日志里会打印 16 行 `=> in_channels: ..., in_h: ...`，可用于确认注入数量。

**`n_round` 逻辑**：阈值 23 是为 ResNet-101/152 的 `layer3` 设计的（它们的 `layer3` 有 23 个 block）。
**ResNet-50 的 `layer3` 只有 6 个 block，6 < 23，所以 `n_round` 保持 1 —— 每个 block 都插入。**

**实际插入数**：`layer1(4) + layer2(4) + layer3(6) + layer4(3 中按 n_round 取 2) = 16`，
与 `h_list` 长度一致。

**`place` 参数**：只实现了 `'blockres'`（替换 `conv1`）；其它值会抛 `NotImplementedError`。
非 ResNet 网络同样抛 `NotImplementedError`。

---

## `make_temporal_pool(net, n_segment)`

在 `layer2` 前插入时序池化，把时间维减半：

```python
net.layer2 = TemporalPool(net.layer2, n_segment)
```

仅在 `--temporal_pool` 时启用（`main.py` 里 `args.temporal_pool`）。
注意 `main.py` 的调用时机有两处：`--resume` 时在加载权重**之前**调用，
否则在之后 —— 这是为了让 `state_dict` 的键能对上。

### `class TemporalPool(nn.Module)`

- `forward(x)`：调用静态方法 `temporal_pool` 后交给 `self.net`
- `temporal_pool(x, n_segment)`（`@staticmethod`）：

```python
x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)      # [n, c, t, h, w]
x = F.max_pool3d(x, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
```

沿时间维做核 3、步长 2 的最大池化，**时间长度减半**（8 → 4）。
这就是 `TSN.forward` 里 `num_segments // 2` 还原形状的原因。

---

## 上游被注释掉的代码

| 位置 | 内容 |
|---|---|
| 第 106–144 行 | 第一版 `CoordAtt`（纯坐标注意力，无动态核） |
| 第 147–226 行 | 第二版 `CoordAtt`（用全局池化 `pool_hw` 生成核，思路相近但实现不同） |
| `__init__` 与 `forward` 内 | `conv_th` / `conv_tw` 固定深度卷积的定义与调用 |

这些注释块对理解演进过程有帮助，但**不参与执行**。

## 改动提示

| 想做的事 | 必须同步修改 |
|---|---|
| 换骨干 / 改分辨率 / 改段数 | `h_list` 要重算（按新的插入顺序与特征图尺寸） |
| 改 `shift_div` | 一般无需改代码，改命令行参数即可 |
| 新增时序模块变体 | 注意 `TemporalModule.count` 的取表逻辑 |
| 调试时确认注入数量 | 数启动日志里 `=> in_channels: ..., in_h: ...` 的行数，应为 16 |
