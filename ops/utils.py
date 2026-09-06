import numpy as np

import torch


def load_checkpoint(path, map_location=None):
    """兼容 torch>=2.6 的安全加载。
    checkpoint 内含 numpy 标量，torch>=2.6 默认 weights_only=True 会加载失败，
    需显式 weights_only=False；老版本 torch 无此参数，捕获 TypeError 回退。
    map_location 透传给 torch.load：GPU 服务器默认 None（按保存时的设备还原）；
    若需在纯 CPU 机器上加载/调试，可传 torch.device('cpu')。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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