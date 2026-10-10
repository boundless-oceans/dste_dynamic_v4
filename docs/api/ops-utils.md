# `ops/utils.py` — 通用工具

54 行。四个工具函数/类，被 `main.py` 与 `test_models.py` 使用。

```python
from ops.utils import AverageMeter, accuracy, load_checkpoint, softmax
```

---

## ⭐ `load_checkpoint(path, map_location=None)` 本仓库新增

**兼容 torch >= 2.6 的权重加载。**

```python
def load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
```

| 参数 | 说明 |
|---|---|
| `path` | 权重文件路径 |
| `map_location` | 透传给 `torch.load`。默认 `None`（按保存时的设备还原）；纯 CPU 调试可传 `torch.device('cpu')` |

**返回**：反序列化后的对象（本仓库里是含 `state_dict` / `optimizer` 等键的 dict）。

**为什么需要它**：torch 2.6 起 `torch.load` 的 `weights_only` 默认值由 `False` 改为 `True`。
`weights_only=True` 只允许加载张量与基本类型，而本项目的 checkpoint 里含
**numpy 标量**（`best_prec1` 等），会加载失败。

解决方式是显式传 `weights_only=False`。但旧版 torch 没有这个参数，会抛 `TypeError`，
所以用 `try/except TypeError` 回退 —— 这样一份代码在 torch 2.1（`environment.yml` 钉的版本）
到 2.6+ 上都能跑。

**调用点**（全项目统一走它，不再直接 `torch.load`）：

| 位置 | 用途 |
|---|---|
| `main.py` `--resume` | 恢复训练 |
| `main.py` `--tune_from` | 加载预训练权重微调 |
| `test_models.py` | 加载评测权重 |

> ⚠️ **安全提示**：`weights_only=False` 会执行 pickle 反序列化，理论上可执行任意代码。
> 因此**只加载自己训练产出的权重**，不要加载来源不明的 `.pth.tar`。

---

## `softmax(scores)`

```python
def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]
```

数值稳定的 numpy softmax（先减去最大值防止 `exp` 溢出）。

**本仓库未被调用** —— 属于上游遗留工具。`test_models.py` 用的是
`torch.nn.functional.softmax`（由 `--softmax` 控制）。

---

## `class AverageMeter`

**滑动平均累加器**，用于统计 loss 与准确率。

```python
def __init__(self):  self.reset()
def reset(self):
    self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
```

| 属性 | 含义 |
|---|---|
| `val` | 最近一次的值（瞬时值） |
| `sum` | 加权累计和 |
| `count` | 加权样本总数 |
| `avg` | `sum / count`，全程平均 |

`n` 通常是 batch size，所以 `avg` 是**按样本加权**的平均，而非按 batch 简单平均 ——
这正是打印格式里 `Loss {loss.val:.4f} ({loss.avg:.4f})` 两个数的区别：
括号内是全程平均。

**被用于**：`main.py` 的 `batch_time` / `data_time` / `losses` / `top1` / `top5`。

> ⚠️ `test_models.py` **自带一份完全相同的实现**（第 59–74 行），并未从这里导入。
> 改这个类时要注意另一处。

---

## `accuracy(output, target, topk=(1,))`

**计算 Prec@k**。

```python
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
```

| 参数 | 说明 |
|---|---|
| `output` | 模型输出 `[B, num_class]`（logits） |
| `target` | 真值标签 `[B]` |
| `topk` | 要计算的 k 值元组，默认 `(1,)` |

**返回**：`list[torch.Tensor]`，与 `topk` 一一对应的**百分数**（已 ×100）。
例如 `accuracy(out, tgt, topk=(1, 5))` 返回 `[prec1, prec5]`。

**实现要点**：

- `output.topk(maxk, 1, True, True)` — 沿 dim=1 取前 `maxk` 个，`largest=True`、`sorted=True`
- `pred.t()` 转置成 `[maxk, B]`，再与扩展后的 target 逐元素比较得到布尔矩阵 `[maxk, B]`
- `correct[:k]` 取前 k 行，展平后求和 → 命中数；除以 `batch_size` 并 ×100 得百分比
- 只做一次 topk 就同时得到所有 k 的结果（这是 `maxk` 的用意）

**注意**：`mul_` 是**原地**操作，作用在 `correct_k`（由 `sum` 新建的张量）上，
不会污染 `correct`。返回的张量需用 `.item()` 取标量
（`main.py` 里写作 `prec1.item()`）。

> ⚠️ 与 `AverageMeter` 一样，`test_models.py` 也**自带一份**（第 77–88 行）。
