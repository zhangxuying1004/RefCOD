from typing import Tuple

import torch
import torch.nn as nn

from . import tool_funcs as tools
from .flops_of_ops import count_parameters, count_total_parameters, register_hooks

default_dtype = torch.float64


def profile_with_inter_params(
    model: nn.Module,
    inputs: tuple,
    custom_ops: dict = None,
    verbose_for_hook: bool = False,
    verbose_for_count: bool = False,
    exclude_self_modules: Tuple[nn.Module] = None,
):
    _custom_ops = {}
    _skip_self_modules = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
    if custom_ops is not None:
        _custom_ops.update(custom_ops)
    if exclude_self_modules is not None:
        assert all([issubclass(x, nn.Module) for x in exclude_self_modules])
        _skip_self_modules = _skip_self_modules + exclude_self_modules

    # 用于保存各个模块的forward hook，便于最后remove，同时亦可记录被加了hook的模块对象（作为key）
    handler_collection = {}
    types_collection = set()  # 用于不重复地记录各个模块的类型

    def add_hooks(m: nn.Module):
        """
        Add the forward hook for each ops in register_hooks and custom_ops.
        It is noted that, the ops in register_hooks is usually some basic ``nn.`` ops provided by pytorch.
        If your module itself has other ops, you need to define the info ``{module_class: count_flops_func}``
        in custom_ops, and the ops defined in register_hooks would not need to be redefined, unless you want
        to replace the default calculation procedure.
        :param m:
        :return:
        """
        m.register_buffer("total_ops", torch.zeros(1, dtype=default_dtype))
        m.register_buffer("total_params", torch.zeros(1, dtype=default_dtype))

        m_type = type(m)

        fn = None
        if m_type in _custom_ops:
            # if defined both op maps, use _custom_ops to overwrite.
            fn = _custom_ops[m_type]
            if m_type not in types_collection and verbose_for_hook:
                tools.prYellow(f"[INFO] Customize rule {fn.__qualname__} for {m_type}.")
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose_for_hook:
                tools.prGreen(f"[INFO] Register {fn.__qualname__} for {m_type}.")
        else:
            # 未定义的Module（注意，其子模块可以是定义的，并不影响，这里只是没法计算该模块除了这些定义了的子模块的flops之外的信息）
            if m_type not in types_collection and verbose_for_hook:
                tools.prRed(
                    f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs and zero Params."
                )

        if fn is not None:
            handler_collection[m] = (
                m.register_forward_hook(fn),
                m.register_forward_hook(count_parameters),
            )
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    layer_infos = []

    def dfs_count(module: nn.Module, prefix="->"):
        _total_ops, _total_params = 0, 0
        # 已经加了hook的模块对象，即custom_ops and register_hooks中存在的那些模块
        if module in handler_collection and not isinstance(module, _skip_self_modules):
            # 获取除子模块外的本层操作的统计
            _total_ops, _total_params = (
                module.total_ops.item(),
                module.total_params.item(),
            )
        layer_infos.append(
            (prefix, f"LayerSelf: ops: {_total_ops}, params: {_total_params}")
        )

        for sub_name, sub_module in module.named_children():
            # 存在子模块，收集子模块信息
            # 对于nn.Module类型的模块，不论是基础的nn.Conv2d，还是复杂的用户自定义，都是可以使用named_children遍历的
            # 用于处理那些包含子模块的模块，通常是nn.Sequential, nn.ModuleList, nn.ModuleDict，或者自定义的Module
            _m_ops, _m_params = dfs_count(
                module=sub_module,
                prefix=f"{prefix}->{sub_name}({sub_module._get_name()})",
            )
            _total_ops += _m_ops
            _total_params += _m_params
        layer_infos.append(
            (prefix, f"LayerTotal: ops: {_total_ops}, params: {_total_params}")
        )
        return _total_ops, _total_params

    total_ops, total_params = dfs_count(model, prefix="main")

    if verbose_for_count:
        for name, info in layer_infos:
            print(name, info, sep="\n")

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return total_ops, total_params


def profile(
    model: nn.Module,
    inputs: tuple,
    custom_ops: dict = None,
    verbose_for_hook: bool = False,
    verbose_for_count: bool = False,
    exclude_self_modules: Tuple[nn.Module] = None,
):
    _custom_ops = {}
    _skip_self_modules = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
    if custom_ops is not None:
        _custom_ops.update(custom_ops)
    if exclude_self_modules is not None:
        assert all([issubclass(x, nn.Module) for x in exclude_self_modules])
        _skip_self_modules = _skip_self_modules + exclude_self_modules

    # 用于保存各个模块的forward hook，便于最后remove，同时亦可记录被加了hook的模块对象（作为key）
    handler_collection = {}
    types_collection = set()  # 用于不重复地记录各个模块的类型

    def add_hooks(m: nn.Module):
        """
        Add the forward hook for each ops in register_hooks and custom_ops.
        It is noted that, the ops in register_hooks is usually some basic ``nn.`` ops provided by pytorch.
        If your module itself has other ops, you need to define the info ``{module_class: count_flops_func}``
        in custom_ops, and the ops defined in register_hooks would not need to be redefined, unless you want
        to replace the default calculation procedure.
        :param m:
        :return:
        """
        m.register_buffer("total_ops", torch.zeros(1, dtype=default_dtype))

        m_type = type(m)

        fn = None
        if m_type in _custom_ops:
            # if defined both op maps, use _custom_ops to overwrite.
            fn = _custom_ops[m_type]
            if m_type not in types_collection and verbose_for_hook:
                tools.prYellow(f"[INFO] Customize rule {fn.__qualname__} for {m_type}.")
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose_for_hook:
                tools.prGreen(f"[INFO] Register {fn.__qualname__} for {m_type}.")
        else:
            # 未定义的Module（注意，其子模块可以是定义的，并不影响，这里只是没法计算该模块除了这些定义了的子模块的flops之外的信息）
            if m_type not in types_collection and verbose_for_hook:
                tools.prRed(
                    f"[WARN] Cannot find rule for {m_type}. Treat it as zero Macs and zero Params."
                )

        if fn is not None:
            handler_collection[m] = m.register_forward_hook(fn)
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    layer_infos = []

    def dfs_count(module: nn.Module, prefix="->"):
        _total_ops = 0
        # 已经加了hook的模块对象，即custom_ops and register_hooks中存在的那些模块
        if module in handler_collection and not isinstance(module, _skip_self_modules):
            # 获取除子模块外的本层操作的统计
            _total_ops = module.total_ops.item()
        layer_infos.append((prefix, f"LayerSelf: ops: {_total_ops}"))

        for sub_name, sub_module in module.named_children():
            # 存在子模块，收集子模块信息
            # 对于nn.Module类型的模块，不论是基础的nn.Conv2d，还是复杂的用户自定义，都是可以使用named_children遍历的
            # 用于处理那些包含子模块的模块，通常是nn.Sequential, nn.ModuleList, nn.ModuleDict，或者自定义的Module
            _total_ops += dfs_count(
                module=sub_module,
                prefix=f"{prefix}->{sub_name}({sub_module._get_name()})",
            )
        layer_infos.append((prefix, f"LayerTotal: ops: {_total_ops}"))
        return _total_ops

    total_ops = dfs_count(model, prefix="main")
    total_params = count_total_parameters(model)

    if verbose_for_count:
        for name, info in layer_infos:
            print(name, info, sep="\n")

    # reset model to original status
    model.train(prev_training_status)
    for m, op_handler in handler_collection.items():
        op_handler.remove()
        m._buffers.pop("total_ops")

    return total_ops, total_params
