# `main.py` — 训练 / 验证入口

454 行。直接运行即开始训练：

```bash
python main.py ipnhand RGB --arch resnet50 --num_segments 8 --shift ...
```

## 模块级变量

| 名称 | 说明 |
|---|---|
| `best_prec1 = 0` | 全局最优 Prec@1，用于判断是否保存 best 权重 |
| `args` | 在 `main()` 里由 `parser.parse_args()` 赋值，随后被所有函数以全局变量方式读取 |

> ⚠️ `args` 是**模块级全局变量**，`train()` / `validate()` / `save_checkpoint()` 都直接引用它。
> 这意味着这些函数不能作为库被外部调用（除非先设好 `main.args`）。这是上游 TSM 的遗留设计。

---

## `_init_worker(worker_id)`

```python
def _init_worker(worker_id):
    seed = 1 + worker_id
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
```

**作用**：作为 `DataLoader(worker_init_fn=...)` 的回调，在每个子进程启动时设置独立随机种子。

**为什么需要**：`DataLoader` 的多个 worker 是 `fork` 出来的，若不在子进程内重新播种，
所有 worker 会用**同一个**随机状态，导致同一 batch 内不同 worker 的随机裁剪/翻转参数高度相关，
削弱增强的多样性。

**参数**：`worker_id` — 由 PyTorch 传入的 worker 序号。种子取 `1 + worker_id`，
避开 0（与主进程种子 `1` 区分开）。

**本仓库改动点**：新增函数，上游 TSM 没有。

---

## `main()`

训练全过程的主控函数。流程分 6 段：

### ① 参数与随机种子

```python
args = parser.parse_args()
torch.manual_seed(1); torch.cuda.manual_seed_all(1)
np.random.seed(1); random.seed(1)
```

四个种子统一设为 1，保证可复现（受 `cudnn.benchmark` 影响，见文末）。

### ② 数据配置与 store_name

```python
num_class, args.train_list, args.val_list, args.root_path, prefix = \
    dataset_config.return_dataset(args.dataset, args.modality)
```

从 `ops/dataset_config.py` 取回 5 个值，并据此拼接实验目录名：

```
TSQ_ipnhand_RGB_resnet50_shift0.5_blockres_avg_segment8_e50
```

**这个字符串很关键** —— 它是 `checkpoint/` 与 `log/` 的子目录名，
也是 `test_models.py` 反解网络结构的依据（见 [test-models.md](test-models.md)）。
命名规则为：

| 片段 | 来源 |
|---|---|
| `TSQ_<数据集>_<模态>` | 固定前缀 |
| `<骨干>_shift<div>_<place>` | 仅当 `--shift` |
| `_tpool` | 仅当 `--temporal_pool` |
| `<共识方式>_segment<N>_e<轮数>` | 固定 |
| `_<pretrain>` / `_<lr_type>` / `_dense` / `_nl` / `_<suffix>` | 条件追加 |

### ③ 建模型

```python
model = TSN(num_class, args.num_segments, args.modality, base_model=args.arch, ...)
crop_size = model.crop_size;  scale_size = model.scale_size
input_mean = model.input_mean; input_std = model.input_std
policies = model.get_optim_policies()
train_augmentation = model.get_augmentation(flip=False if 'ipnhand' in args.dataset ... else True)
model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
summary(model, input_size=(24, 224, 224))
```

注意两点：

- `flip=False`：**手势数据集默认关闭 TSM 自带的 `GroupRandomHorizontalFlip`**
  （因为它只翻图像不改标签），改由 `TSNDataSet` 内部带标签对调的翻转负责。
- `summary(model, (24, 224, 224))`：打印结构。`24 = 8 段 × 3 通道`，
  这个数写死与 `--num_segments 8` 绑定；换段数需同步改，否则报错。

### ④ 恢复与微调

| 分支 | 参数 | 行为 |
|---|---|---|
| 续训 | `--resume <路径>` | 恢复 `epoch` / `best_prec1` / `state_dict` / `optimizer` |
| 微调 | `--tune_from <路径>` | 只加载权重，处理 `.net` 前后缀差异；换数据集时**丢弃 fc 层** |

`--tune_from` 的处理逻辑：先收集所有 `.net` 增删导致的键名差异并重命名，
再剔除 `fc`（新数据集）或 `conv1.weight`（Flow 微调 RGB）。

### ⑤ 构建数据加载器

训练集与验证集各一个 `TSNDataSet` + `DataLoader`。训练侧的三个要点：

```python
# 1. 自定义变换链
transform = Compose([train_augmentation, GroupColorJitter(brightness=0.2, contrast=0.2),
                     Stack(...), ToTorchFormatTensor(...), normalize,
                     torchvision.transforms.RandomErasing(p=0.2)])

# 2. 类别均衡采样
label_counts = {}                        # 统计每类样本数
for s in train_dataset.data: label_counts[s['label']] = ... + 1
sample_weights = [1.0 / label_counts[s['label']] for s in train_dataset.data]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset.data), replacement=True)

# 3. DataLoader
DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=32,
           pin_memory=True, worker_init_fn=_init_worker, drop_last=True)
```

- `drop_last=True`：丢弃不足一批的尾部，防止 DataParallel 下批大小不整除。
- **`sampler` 与 `shuffle` 互斥**，这里用 sampler 接管了打乱。
- 验证集 `random_shift=False, shuffle=False`，保证评估确定性。

### ⑥ 训练循环与评估

```python
for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
    train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
    if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
        prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({...}, is_best)
```

若传 `--evaluate`，则只跑一次 `validate()` 后直接返回，不训练。

---

## `train(train_loader, model, criterion, optimizer, epoch, log, tf_writer)`

跑完一个 epoch 的训练。

**参数**：

| 参数 | 说明 |
|---|---|
| `train_loader` | 训练 DataLoader |
| `model` | 已被 `DataParallel` 包装的 TSN |
| `criterion` | `CrossEntropyLoss` |
| `optimizer` | SGD，参数分 9 组 |
| `epoch` | 当前轮次（从 0 开始，打印时 +1） |
| `log` | 已打开的 `log.csv` 文件对象 |
| `tf_writer` | `tensorboardX.SummaryWriter` |

**单个 batch 的顺序**：

```python
target = target.cuda(); input_var = Variable(input); target_var = Variable(target)
output = model(input_var)                    # [16, 13]
loss = criterion(output, target_var)
prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
loss.backward()
if args.clip_gradient is not None:           # --clip-gradient / --gd
    clip_grad_norm_(model.parameters(), args.clip_gradient)
optimizer.step(); optimizer.zero_grad()
```

**监控指标**用 5 个 `AverageMeter` 累积：`batch_time` / `data_time` / `losses` / `top1` / `top5`，
每 `--print-freq` 次打印一行并写 `log.csv`。

**epoch 末尾**写 TensorBoard：`loss/train`、`acc/train_top1`、`acc/train_top5`、`lr`。

**本仓库改动点**：打印学习率时由上游的 `lr * 0.1` 改为 `lr`
（0.1 是历史遗留的硬编码折算，与实际分组学习率不符）。

---

## `validate(val_loader, model, criterion, epoch, log=None, tf_writer=None)`

在验证集上评估，**返回 `top1.avg`**（供 `main()` 判断是否 best）。

带 `torch.no_grad()`，`model.eval()` 切到推理模式（BN 用滑动统计量、关闭 dropout）。

一并计算并打印 `Prec@1` / `Prec@5` / `Loss`，写 TensorBoard 的
`loss/test`、`acc/test_top1`、`acc/test_top5`。

`log` 与 `tf_writer` 可省略 —— 这样 `--evaluate` 单独跑时也能复用本函数。

> 注意：这里的结果与 `test_models.py` 的口径不同（单 clip 单 crop vs 多 clip 多 crop），
> 详见 [test-models.md](test-models.md#与-mainpy-内验证的差异)。

---

## `save_checkpoint(state, is_best)`

```python
filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
torch.save(state, filename)
if is_best:
    shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
```

每轮都覆写 `ckpt.pth.tar`；仅当 `is_best` 时复制一份 `ckpt.best.pth.tar`。
**没有版本号或轮次后缀**，所以训练中途的历史权重不会被保留。

`state` 字典含 5 个键：`epoch`、`arch`、`state_dict`、`optimizer`、`best_prec1`。

---

## `adjust_learning_rate(optimizer, epoch, lr_type, lr_steps)`

按策略衰减学习率，并**逐参数组**乘上该组的 `lr_mult` / `decay_mult`。

```python
if lr_type == 'step':
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
elif lr_type == 'cos':
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
for param_group in optimizer.param_groups:
    param_group['lr'] = lr * param_group['lr_mult']
    param_group['weight_decay'] = decay * param_group['decay_mult']
```

`--lr_steps 20 40` 时：epoch 0–19 用 `lr`，20–39 用 `lr×0.1`，40+ 用 `lr×0.01`。

`decay` 变量在 `step` 分支里被**复用**为 weight decay（先当学习率衰减因子，再被赋成 `args.weight_decay`），
可读性不佳但逻辑正确。

---

## `check_rootfolders()`

创建 4 个目录（已存在则跳过）：`args.root_log`、`args.root_model`、
以及二者下以 `store_name` 命名的子目录。默认 `root_log='log'`、`root_model='checkpoint'`。

---

## 附：可复现性说明

| 因素 | 影响 |
|---|---|
| 三处种子设置（`main()`、`_init_worker`、Dataset 内） | 数据顺序与增强可复现 |
| `cudnn.benchmark = True` | cuDNN 自选卷积算法，浮点累加顺序可能变 |
| 多卡 DataParallel | 梯度归约顺序不固定 |
| `WeightedRandomSampler` | 抽样依赖全局随机状态 |

追求严格复现需额外设 `cudnn.benchmark = False` 与 `cudnn.deterministic = True`。
