import pynvml
import torch

from ..misc import to_cuda

pynvml.nvmlInit()


@torch.no_grad()
def cal_gpu_mem(model, data, device=0):
    assert torch.cuda.is_available()

    print(f"Counting GPU memory for {model.__class__.__name__}")
    model.cpu()
    torch.cuda.empty_cache()

    data = to_cuda(data)

    handle = pynvml.nvmlDeviceGetHandleByIndex(device)

    initial_mem = count_stable_mem(handle)

    model.cuda()

    runtime_mem = count_stable_mem(
        handle, func=model, args=data, initial_mem=initial_mem
    )

    average_mem = [
        f"Total: {(runtime_mem + initial_mem) / 1024 ** 2:.3f}MB",
        f"Model: {runtime_mem / 1024 ** 2:.3f}MB",
        f"Other: {initial_mem / 1024 ** 2:.3f}MB",
    ]
    average_mem = " | ".join(average_mem)
    return average_mem


def count_stable_mem(handle, func=None, args=None, initial_mem=None):
    mem = 0
    stable_process = 0
    while True:
        if func is not None:
            if args is not None:
                func(args)
            else:
                func()

        new_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        if initial_mem is not None:
            new_mem -= initial_mem

        if mem != new_mem:
            mem = new_mem
            stable_process = 0
        else:
            stable_process += 1

        if stable_process >= 10:
            break
    return 
