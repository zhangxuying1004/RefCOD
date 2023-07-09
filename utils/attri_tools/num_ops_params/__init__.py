import torch


def to_cuda(data):
    if isinstance(data, torch.Tensor):
        data = data.cuda()
    elif isinstance(data, (list, tuple)):
        data = [to_cuda(x) for x in data]
    elif isinstance(data, dict):
        data = {k: to_cuda(v) for k, v in data.items()}
    return data
