# `ops/non_local.py` — non-local 块

173 行。**本仓库默认不启用**（`--non_local 0`）。保留自上游 TSM，
用于在时空特征上做非局部（自注意力式）建模。

> 📌 **出处**：文件头注明来自
> [AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch)，
> 经 TSM 官方实现引入本仓库。属于第三方代码，修改时注意保留出处。
>
> ⚠️ **当前状态下启用会报错**：`make_non_local` 里有一行 `import archs`，
> 而本仓库没有这个模块。详见文末「已知问题」。

---

## `class _NonLocalBlockND(nn.Module)`

**通用实现**，支持 1D/2D/3D，其余三个类都是它的薄封装。

### `__init__(in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True)`

| 参数 | 说明 |
|---|---|
| `in_channels` | 输入通道数 |
| `inter_channels` | 中间通道数，默认 `in_channels // 2`（为 0 时取 1） |
| `dimension` | 1 / 2 / 3，决定用 `Conv1d/2d/3d` 与 `BatchNorm1d/2d/3d` |
| `sub_sample` | 是否对 `g` 与 `phi` 做空间下采样以省算力 |
| `bn_layer` | `W` 是否带 BatchNorm |

**维度映射**：

| `dimension` | 卷积 | 池化层 | 归一化 |
|---|---|---|---|
| 3 | `Conv3d` | `MaxPool3d((1,2,2))` | `BatchNorm3d` |
| 2 | `Conv2d` | `MaxPool2d((2,2))` | `BatchNorm2d` |
| 1 | `Conv1d` | `MaxPool1d(2)` | `BatchNorm1d` |

**三个 1×1 卷积的语义**（注意力机制的 Q/K/V 类比）：

| 名称 | 作用 |
|---|---|
| `theta` | 生成 query |
| `phi` | 生成 key（`sub_sample` 时带池化） |
| `g` | 生成 value（`sub_sample` 时带池化） |
| `W` | 输出投影，把 `inter_channels` 映回 `in_channels` |

**关键初始化**：`W` 的权重与偏置被**显式置零**：

```python
nn.init.constant_(self.W[1].weight, 0)
nn.init.constant_(self.W[1].bias, 0)
```

因此训练初期 non-local 分支输出为 0，模块等价于恒等映射，**后接残差相加时不会破坏预训练权重**。
这是 non-local 论文的标准做法。

`sub_sample` 时用 `nn.Sequential(self.g, max_pool_layer)` 把池化拼在卷积后面
（注意 `self.g` 被**重新赋值**为 Sequential，原卷积被包在里面）。

### `forward(x)`

```python
g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)      # [b, N, C]
theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)                    # [b, C, N]
f = torch.matmul(theta_x, phi_x)              # [b, N, N] 相似度矩阵
f_div_C = F.softmax(f, dim=-1)
y = torch.matmul(f_div_C, g_x)                # [b, N, C]
y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *x.size()[2:])
W_y = self.W(y)
z = W_y + x                                   # 残差
return z
```

标准的**嵌入式高斯** non-local：`theta` 与 `phi` 的点积经 softmax 得到
所有位置对之间的注意力权重，再对 `g` 加权求和。`N` 是所有时空位置数，
所以复杂度是 `O(N²)` —— 这也是它只插在分辨率较低的 `layer2`/`layer3` 的原因。

**残差**：`z = W_y + x`，配合 `W` 初始化为 0，训练起点是恒等映射。

---

## 三个维度封装

```python
class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__(in_channels, inter_channels=inter_channels,
                         dimension=1, sub_sample=sub_sample, bn_layer=bn_layer)
```

`NONLocalBlock2D` / `NONLocalBlock3D` 结构完全相同，只是 `dimension` 分别为 2 / 3。
它们**只重写了 `__init__`**，`forward` 全部继承自 `_NonLocalBlockND`。

---

## `class NL3DWrapper(nn.Module)`

**把 3D non-local 块包在 ResNet 的残差块外**。

### `__init__(block, n_segment)`

```python
self.block = block
self.nl = NONLocalBlock3D(block.bn3.num_features)
self.n_segment = n_segment
```

`block.bn3.num_features` 即该残差块的输出通道数，用它作为 non-local 的通道数。

### `forward(x)`

```python
x = self.block(x)                                          # 先跑原始残差块
nt, c, h, w = x.size()
x = x.view(nt // self.n_segment, self.n_segment, c, h, w).transpose(1, 2)   # [n, c, t, h, w]
x = self.nl(x)                                             # 在 5D 张量上做 non-local
x = x.transpose(1, 2).contiguous().view(nt, c, h, w)       # 还原成 4D
return x
```

**这是「时间折进批维」与「3D 模块」之间的桥**：
因为主网络全程用 4D 张量 `[nt, c, h, w]`（时间在批维里），
要用 3D non-local 就必须先把 `nt` 拆回 `[n, t]`，算完再折回去。
`transpose(1, 2)` 就是在批维与时间维之间搬数据。

---

## `make_non_local(net, n_segment)`

**把 non-local 块插入 ResNet 的 `layer2` 与 `layer3`。**

```python
def make_non_local(net, n_segment):
    import torchvision
    import archs                                    # ⚠️ 本仓库不存在此模块
    if isinstance(net, torchvision.models.ResNet):
        net.layer2 = nn.Sequential(
            NL3DWrapper(net.layer2[0], n_segment), net.layer2[1],
            NL3DWrapper(net.layer2[2], n_segment), net.layer2[3],
        )
        net.layer3 = nn.Sequential(
            NL3DWrapper(net.layer3[0], n_segment), net.layer3[1],
            NL3DWrapper(net.layer3[2], n_segment), net.layer3[3],
            NL3DWrapper(net.layer3[4], n_segment), net.layer3[5],
        )
    else:
        raise NotImplementedError
```

**插入位置**：`layer2` 的第 0、2 个块，`layer3` 的第 0、2、4 个块 ——
**都是偶数下标**，且恰好是 ResNet-50 的 4/4/6/3 结构能容纳的数量。

> ⚠️ 这里的索引是**硬编码**的。若换成 `layer3` 只有 4 个块的结构（如 ResNet-18/34），
> `net.layer3[4]` 与 `net.layer3[5]` 会 `IndexError`。

---

## 已知问题

### 1. `import archs` 不存在

`make_non_local` 第 134 行 `import archs`，但本仓库（以及上游 TSM 仓库）都没有 `archs.py`。
**只要设置 `--non_local 1`，就会立刻 `ModuleNotFoundError`。**

该 import 在整个函数里**没有被使用**，属于无用导入。
若要启用 non-local，删掉这一行即可：

```python
def make_non_local(net, n_segment):
    import torchvision
    # import archs          # ← 删掉
```

### 2. 与 `make_temporal_module` 的 `n_round` 假设冲突

`NL3DWrapper` 替换的是整个块的输出位置，而 `make_temporal_module` 认为
`layer3` 有 6 个块（`n_round=1` 时每个块都插入 TemporalModule）。
两者都启用时都按「6 个块」假设工作，因此可以共存 —— 但
`h_list` 是按 `layer3` 6 块硬编码的，若 non-local 改变了层的结构，需一并核对。

### 3. `__main__` 自测块

文件末尾（第 154–174 行）有一段自测代码，分别构造 1D/2D/3D non-local 块并打印输出尺寸。
直接运行本文件即可验证实现是否正常，**但不要与训练流程混用**。

---

## 本仓库为何不启用

`main.py` 默认 `--non_local 0`，`TSN._prepare_base_model` 里 `if self.non_local:` 为假，
因此本模块**从未被导入**（`make_non_local` 是函数内 import）。

时序建模已由 `ops/temporal_module.py` 的 `TemporalModule` + `CoordAtt` 承担，
non-local 是另一条可选路线，启用需先修掉上面的 `import archs`。
