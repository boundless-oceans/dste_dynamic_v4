# `opts.py` — 命令行参数

76 行。定义 `main.py` 的全部参数，导出单个对象 `parser`。

```python
from opts import parser        # main.py 的用法
args = parser.parse_args()
```

> ⚠️ **本文件被 import 时就会创建 `parser`**（模块级代码），
> 且 `main.py` 在 `main()` 内部才调用 `parse_args()`。

## 位置参数（必填）

| 参数 | 说明 | 本仓库取值 |
|---|---|---|
| `dataset` | 数据集名，需在 `dataset_config.return_dataset` 中注册 | `ipnhand` |
| `modality` | 模态，**有 `choices=['RGB', 'Flow']` 限制** | `RGB` |

> 注意 `choices` 只限定了 `RGB` / `Flow`，没有 `RGBDiff`。
> 虽然 `models.py` 与 `dataset.py` 里有 `RGBDiff` 分支，但**命令行传不进去**。

---

## 模型配置

| 参数 | 默认 | 本仓库脚本 | 说明 |
|---|---|---|---|
| `--arch` | `BNInception` ⚠️ | `resnet50` | 骨干名。**只有 ResNet 系可用**，其它值会触发 `NotImplementedError` / `ValueError` |
| `--num_segments` | `3` | **`8`** | 时序段数 |
| `--consensus_type` | `avg` | `avg` | 段间融合方式，传给 `ConsensusModule` |
| `--k` | `3` | – | **未被使用**（上游 TSM 的 RNN 共识遗留） |
| `--dropout` / `--do` | `0.5` | `0.5` | 分类头前的 dropout |
| `--loss_type` | `nll` | – | **有 `choices=['nll']`**，只能是它，对应 `CrossEntropyLoss` |
| `--img_feature_dim` | `256` | – | 每帧特征维度，仅 Flow 分支使用 |
| `--suffix` | `None` | – | 追加到 `store_name` 末尾，用于区分实验 |
| `--pretrain` | `imagenet` | – | 非 `imagenet` 时会写进 `store_name` |
| `--tune_from` | `None` | – | 微调起点权重路径（只加载权重，不恢复 epoch） |

> ⚠️ **`--arch` 默认值是 `BNInception`，但代码里已不支持它**（相关分支被注释）。
> 不显式传 `--arch` 会直接报错。本仓库脚本总是显式传 `resnet50`。

---

## 学习配置

| 参数 | 默认 | 本仓库脚本 | 说明 |
|---|---|---|---|
| `--epochs` | `120` | `50` | 总轮数 |
| `-b` / `--batch-size` | `128` | `16` | 批大小（**训练**） |
| `--lr` / `--learning-rate` | `0.001` | `0.01` | 初始学习率 |
| `--lr_type` | `step` | – | `step` 或 `cos`，非 `step` 时会写进 `store_name` |
| `--lr_steps` | `[50, 100]` | `20 40` | 衰减节点，step 策略下每次 ×0.1 |
| `--momentum` | `0.9` | – | SGD 动量 |
| `--weight-decay` / `--wd` | `5e-4` | `1e-4` | 权重衰减 |
| `--clip-gradient` / `--gd` | `None` | `20` | 梯度范数裁剪，`None` 表示不裁剪 |
| `--no_partialbn` / `--npb` | `False` | **`--npb`** | 见下方说明 |

**`--npb` 的语义容易搞反**，注意代码里的用法：

```python
# main.py
model = TSN(..., partial_bn=not args.no_partialbn, ...)
# train() 每个 epoch：
model.module.partialBN(False if args.no_partialbn else True)
```

| 传参 | `partial_bn` | 效果 |
|---|---|---|
| 不传（默认） | `True` | **冻结**除首个之外的 BN，保留 ImageNet 统计量 |
| 传 `--npb` | `False` | **所有 BN 都参与训练** |

名字叫 "no partial bn"，即「不要做 partial BN」→ 全部 BN 正常训练。本仓库脚本传了 `--npb`。

---

## 监控配置

| 参数 | 默认 | 本仓库脚本 | 说明 |
|---|---|---|---|
| `--print-freq` / `-p` | `20` | – | 每多少 batch 打印一行训练日志（help 里写 10，**与实际不符**） |
| `--eval-freq` / `-ef` | `5` | `1` | 每多少轮做一次验证 |

---

## 运行时配置

| 参数 | 默认 | 本仓库脚本 | 说明 |
|---|---|---|---|
| `-j` / `--workers` | `8` | `32` | DataLoader 进程数 |
| `--resume` | `''` | – | 续训权重路径（恢复 epoch / optimizer） |
| `-e` / `--evaluate` | `False` | – | 只评估不训练 |
| `--snapshot_pref` | `''` | – | **未被使用**（上游遗留） |
| `--start-epoch` | `0` | – | 手动指定起始轮次（`--resume` 会覆盖它） |
| `--gpus` | `None` | – | GPU 编号列表。`None` 时 `DataParallel` 用全部可见卡 |
| `--flow_prefix` | `''` | – | **未被使用**（光流帧前缀遗留） |
| `--root_log` | `log` | – | 日志根目录 |
| `--root_model` | `checkpoint` | – | 权重根目录 |

> **`--gpus` 与 `CUDA_VISIBLE_DEVICES` 的分工**：脚本里设的是
> `export CUDA_VISIBLE_DEVICES=0,1`（对进程隐藏其它卡），
> `--gpus` 则是 `DataParallel(device_ids=...)` 的实参。两者都不传时，
> PyTorch 默认使用当前可见的所有卡。

---

## 时序模块配置

| 参数 | 默认 | 本仓库脚本 | 说明 |
|---|---|---|---|
| `--shift` | `False` | **`--shift`** | 是否插入 `TemporalModule`。**不开就退化成纯 TSN（逐帧独立）** |
| `--shift_div` | `8` | `0.50` | 参与时序建模的通道比例。默认 8 是「1/8 通道」的旧语义，本仓库改为比例 0.5 |
| `--shift_place` | `blockres` | `blockres` | 插入位置，**只实现了 `blockres`** |
| `--temporal_pool` | `False` | – | 是否在 `layer2` 前做时序池化（时间减半） |
| `--non_local` | `False` | – | `store_true` 布尔值。⚠️ 启用会因 `import archs` 报错，见 [ops-non-local.md](ops-non-local.md#已知问题) |
| `--dense_sample` | `False` | – | i3D 式密集采样（每样本 10 clip） |

**`--shift_div` 的语义变化值得注意**：

```python
# ops/temporal_module.py
self.num_shift_channel = int(self.in_channels * self.n_div)
```

它被当作**比例**使用（`in_channels × 0.5`）。而默认值 `8` 会让
`int(c × 8)` 变成输入通道的 8 倍 —— 超出通道数，`split_sizes` 的第二个元素变负，
`torch.split` 会报错。**因此 `--shift` 必须配一个 (0, 1) 之间的小数**，
本仓库脚本传 `0.50`。

> 这也是 `store_name` 里出现 `shift0.5` 而不是 `shift8` 的原因。

---

## `store_name` 如何受参数影响

`main.py` 按以下顺序拼接实验名，**这些参数都会改变目录名与权重文件名**：

```python
['TSQ', dataset, modality, full_arch_name, consensus_type, 'segment%d' % num_segments, 'e%d' % epochs]
```

其中 `full_arch_name` = `arch` + （`_shift{div}_{place}` 若 `--shift`）+ （`_tpool` 若 `--temporal_pool`）。

另外条件追加：`_{pretrain}`（非 imagenet）、`_{lr_type}`（非 step）、`_dense`、`_nl`、`_{suffix}`。

**本仓库实际得到**：

```
TSQ_ipnhand_RGB_resnet50_shift0.5_blockres_avg_segment8_e50
```

> ⚠️ `test_models.py` 会从这个名字**反解** `is_shift` / `shift_div` / `shift_place` / `arch`
> （见 [test-models.md](test-models.md#parse_shift_option_from_log_namelog_name)）。
> 所以改这些参数后，权重文件名会变，评测时的自动解析结果也会跟着变 ——
> 这是设计使然，但**手动重命名权重文件会破坏它**。

---

## 参数与脚本的对应关系

训练脚本传给 `main.py` 的实参：

```bash
python main.py ipnhand RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
     --batch-size 16 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=0.50 --shift_place=blockres --npb
```

| 与默认值对比 | 参数 |
|---|---|
| 显式覆盖（因默认值不适合） | `--arch`、`--num_segments`、`--epochs`、`--batch-size`、`--lr`、`--wd`、`--lr_steps`、`--shift_div` |
| 值恰好等于默认（显式写出以示明确） | `--dropout 0.5`、`--consensus_type=avg`、`--shift_place=blockres` |
| 打开开关 | `--shift`、`--npb` |
| 性能相关 | `--gd 20`（梯度裁剪）、`-j 32`（数据加载进程）、`--eval-freq=1`（每轮验证） |
