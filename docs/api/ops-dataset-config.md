# `ops/dataset_config.py` — 数据集注册表

162 行。**决定「数据在哪、有几类、帧文件名长什么样」**。
换数据集或数据路径不匹配时，第一个要改的就是这个文件。

> ⚠️ 第 2 个提交（`fix: 修复数据集路径错误问题`）改的就是本文件的 `ROOT_DATASET` 一行。
> 这是全项目最容易配错的地方。

---

## `ROOT_DATASET`

```python
ROOT_DATASET = '/home/vis/data/datasets/'  # 实际数据在 6TB 卷: /home/vis/data/datasets/
```

**所有数据路径的根**。注释里的原值是 `/home/vis/datasets/`，第 3 个提交把它修正为
`/home/vis/data/datasets/`。

**部署到新服务器时必须改这一行**（或建同名软链接）。
路径不对的表现见 [ops-dataset.md](ops-dataset.md)：满屏 `video path not exist!!!`，
样本被**静默跳过**，不报错。

---

## `return_dataset(dataset, modality)`

**唯一的对外入口**，其余 `return_*` 都由它按名字分发。

```python
def return_dataset(dataset, modality):
    dict_single = {'jester': return_jester, 'something': return_something, ...,
                   'ipnhand': return_ipnhand}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = \
            dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):          # 类别清单是一个文件
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            categories = [item.rstrip() for item in f.readlines()]
    else:                                          # 类别数是一个整数
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
```

**参数**：

| 参数 | 本仓库取值 | 说明 |
|---|---|---|
| `dataset` | **`'ipnhand'`** | 数据集名（命令行第一个位置参数） |
| `modality` | **`'RGB'`** | 模态（第二个位置参数） |

**返回 5 元组**，被 `main.py` 与 `test_models.py` 直接解包：

```python
num_class, args.train_list, args.val_list, args.root_path, prefix = \
    dataset_config.return_dataset(args.dataset, args.modality)
```

| 返回值 | 说明 |
|---|---|
| `n_class` | 类别数（IPN-Hand 为 **13**） |
| `file_imglist_train` | 训练标注的**绝对路径** |
| `file_imglist_val` | 验证标注的**绝对路径** |
| `root_data` | 帧数据根目录 |
| `prefix` | 帧文件名模板 |

**两种 `file_categories` 形式**：字符串表示「类别清单文件路径」（如 UCF101 的 `classInd.txt`），
整数表示「类别数量」（如 IPN-Hand 的 13）。这在设计上支持两种标注风格，
但 IPN-Hand 走整数分支 —— **类名实际来自 JSON 的 `labels` 字段**（见 [ops-dataset.md](ops-dataset.md)），
`file_categories = 13` 只用来确定类别数。

> 注意 `n_class` 的计算方式：整数分支下 `categories = [None] * 13`，`len()` 得 13。
> 所以这个数字必须与 JSON 里真实的类别数一致，否则分类头维度会错。

---

## `return_ipnhand(modality)` ⭐ 本仓库使用的配置

```python
def return_ipnhand(modality):
    filename_categories = 13
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'IPNHand'
        filename_imglist_train = 'IPNHand/annotation_ipnhand/training.json'
        filename_imglist_val = 'IPNHand/annotation_ipnhand/validation.json'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
```

| 项 | 值 |
|---|---|
| 类别数 | **13** |
| 训练标注 | `IPNHand/annotation_ipnhand/training.json` |
| 验证标注 | `IPNHand/annotation_ipnhand/validation.json` |
| 数据根 | `<ROOT_DATASET>/IPNHand` |
| 帧模板 | `'{:06d}.jpg'` → `000001.jpg`、`000002.jpg` |

**只支持 RGB** —— 传 `'Flow'` 会 `raise NotImplementedError`。

**最终的实际路径**（`ROOT_DATASET = '/home/vis/data/datasets/'` 时）：

```
/home/vis/data/datasets/IPNHand/annotation_ipnhand/training.json
/home/vis/data/datasets/IPNHand/annotation_ipnhand/validation.json
/home/vis/data/datasets/IPNHand/<视频名>/000001.jpg
```

> `root_data` 用字符串拼接 `ROOT_DATASET + 'IPNHand'`（而非 `os.path.join`），
> 所以 `ROOT_DATASET` **必须以 `/` 结尾**。
> 而两个 json 路径走 `os.path.join(ROOT_DATASET, 'IPNHand/annotation_ipnhand/training.json')`，
> 拼接后是 `/home/vis/data/datasets/IPNHand/annotation_ipnhand/training.json`，与上面一致。

**帧模板 `{:06d}.jpg` 的含义**：帧号补零到 6 位，从 **1** 开始（不是 0）。
`TSNDataSet._load_image` 用它拼文件名。

---

## 其余数据集的配置（本仓库未使用）

这些函数保留自上游 TSM，供参考或迁移用。签名统一为 `return_xxx(modality)`。
下表数值均按本文件代码逐一核对：

| 函数 | 类别来源 | RGB 帧模板 | 支持模态 |
|---|---|---|---|
| `return_ucf101` | 清单文件 `UCF101/labels/classInd.txt` | `img_{:05d}.jpg` | RGB / Flow |
| `return_hmdb51` | 整数 **51** | `img_{:05d}.jpg` | RGB / Flow |
| `return_something` | 清单文件 `something/v1/category.txt` | `{:05d}.jpg` | RGB / Flow |
| `return_somethingv2` | 清单文件 `something/v2/category.txt` | `{:06d}.jpg` | RGB / Flow |
| `return_kinetics` | 整数 **400** | `img_{:05d}.jpg` | 仅 RGB |
| `return_jester` | 整数 **27** | `{:05d}.jpg` | 仅 RGB |
| `return_hcigesture` | 整数 **10** | `{:05d}.png` ⚠️ PNG | 仅 RGB |
| `return_egogesture` | 整数 **83** | `{:06d}.jpg` | 仅 RGB |
| `return_ipnhand` | 整数 **13** | `{:06d}.jpg` | 仅 RGB |

两处容易踩的坑：

- **帧模板位数不统一**：`{:05d}` 与 `{:06d}` 混用，`hcigesture` 甚至是 **`.png`**。
  照抄别的数据集的 `prefix` 会导致图片全部读不到（然后静默回退到第 1 帧）。
- **类别来源有两种**：UCF-101 与 Something 系用清单文件，其余用整数。
  用整数时必须与标注里的真实类别数一致，否则分类头维度错。

**共同结构**：

```python
def return_xxx(modality):
    filename_categories = <整数 或 清单文件相对路径>
    if modality == 'RGB':      # 或 'Flow' / 'RGBDiff'
        root_data = ROOT_DATASET + '<数据集目录>'
        filename_imglist_train = '<相对路径>'
        filename_imglist_val = '<相对路径>'
        prefix = '<帧模板>'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
```

**若要新增一个数据集**，照此格式加一个 `return_xxx`，并在 `return_dataset` 的
`dict_single` 里注册即可。别忘了 `filename_categories` 必须与标注里的真实类别数一致。

---

## 改动清单（部署到新环境）

| 场景 | 改什么 |
|---|---|
| 数据换了根目录 | `ROOT_DATASET`（最常见） |
| 新增数据集 | 加 `return_xxx` + 在 `dict_single` 注册 |
| 类别数变了 | `filename_categories` 与标注的 `labels` 长度必须同时对上 |
| 帧号不是 6 位补零 | `prefix`（如 `return_kinetics` 用 `img_{:05d}.jpg`） |
| 想用光流 | 需为 `return_ipnhand` 添加 `'Flow'` 分支（目前直接报错） |

验证路径是否配对，最直接的方法是启动时看脚本打印的那一行
（训练/测试脚本都会执行）：

```bash
python -c 'from ops import dataset_config; print(dataset_config.ROOT_DATASET)'
```
