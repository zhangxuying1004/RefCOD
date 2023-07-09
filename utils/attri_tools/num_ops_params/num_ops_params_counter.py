import torch

from . import count_info, tool_funcs


@torch.no_grad()
def cal_macs_params(
    model,
    data,
    custom_ops=None,
    verbose_for_hook=False,
    verbose_for_count=False,
    exclude_self_modules=None,
):
    print(
        f"Calculating the MACs and the number of parameters for model {model.__class__.__name__} with data "
    )

    macs, params = count_info.profile(
        model,
        inputs=(data,),
        custom_ops=custom_ops,
        verbose_for_hook=verbose_for_hook,
        verbose_for_count=verbose_for_count,
        exclude_self_modules=exclude_self_modules,
    )
    macs, params = tool_funcs.clever_format([macs, params], "%.3f")
    return macs, params


@torch.no_grad()
def cal_macs_params_v2(
    model,
    data,
    custom_ops=None,
    verbose_for_hook=False,
    verbose_for_count=False,
    exclude_self_modules=None,
    return_flops=False,
):
    print(f"Counting Number of Ops. & Params. for {model.__class__.__name__}")

    macs, params = count_info.profile(
        model,
        inputs=(data,),
        custom_ops=custom_ops,
        verbose_for_hook=verbose_for_hook,
        verbose_for_count=verbose_for_count,
        exclude_self_modules=exclude_self_modules,
    )
    if return_flops:
        macs *= 2
    macs, params = tool_funcs.clever_format([macs, params], "%.3f")
    return macs, params
