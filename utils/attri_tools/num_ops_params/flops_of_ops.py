import logging
import operator
from distutils.version import LooseVersion
from functools import reduce

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from .tool_funcs import print_ops_name

multiply_adds = 1


def count_parameters(ops: nn.Module, in_tensor, out_tensor):
    # 仅计算当前层的参数，不包含子模块
    # 这里使用`=`，保证多次调用会自动覆盖，仅计算一次
    ops.total_params[0] = sum(
        [torch.DoubleTensor([p.numel()]) for p in ops.parameters(recurse=False)]
    )


def count_total_parameters(model: nn.Module):
    return sum([p.numel() for p in model.parameters()])


def zero_ops(ops: nn.Module, in_tensor, out_tensor):
    ops.total_ops += torch.DoubleTensor([int(0)])


@print_ops_name(verbose=False)
def count_convNd(ops, in_tensor, out_tensor):
    in_tensor = in_tensor[0]

    kernel_ops = torch.zeros(ops.weight.size()[2:]).numel()  # Kw in_tensor Kh
    bias_ops = 1 if ops.bias is not None else 0

    # N in_tensor Cout in_tensor H in_tensor W in_tensor  (Cin in_tensor Kw in_tensor Kh + bias)
    total_ops = out_tensor.nelement() * (
        ops.in_channels // ops.groups * kernel_ops + bias_ops
    )

    ops.total_ops += torch.DoubleTensor([int(total_ops)])  # 多次调用会自动累加，便于应用在siamese结构中


def count_convNd_ver2(
    ops, in_tensor, out_tensor
):
    in_tensor = in_tensor[0]

    # N in_tensor H in_tensor W (exclude Cout)
    output_size = torch.zeros((out_tensor.size()[:1] + out_tensor.size()[2:])).numel()
    # Cout in_tensor Cin in_tensor Kw in_tensor Kh
    kernel_ops = ops.weight.nelement()
    if ops.bias is not None:
        # Cout in_tensor 1
        kernel_ops += ops.bias.nelement()
    # in_tensor N in_tensor H in_tensor W in_tensor Cout in_tensor (Cin in_tensor Kw in_tensor Kh + bias)
    ops.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])


def count_convNd_mul(
    ops, in_tensor, out_tensor: torch.Tensor
):
    in_tensor = in_tensor[0]

    # N in_tensor H in_tensor W (exclude Cout)
    output_size = torch.zeros((out_tensor.size()[:1] + out_tensor.size()[2:])).numel()
    # Cout in_tensor Cin in_tensor Kw in_tensor Kh
    kernel_ops = ops.weight.nelement()
    # in_tensor N in_tensor H in_tensor W in_tensor Cout in_tensor (Cin in_tensor Kw in_tensor Kh + bias)
    ops.total_ops += torch.DoubleTensor([int(output_size * kernel_ops)])


def count_relu_w_params(ops: nn.Module, in_tensor, out_tensor):
    in_tensor = in_tensor[0]

    nelements = in_tensor.numel()

    ops.total_ops += torch.DoubleTensor([int(nelements)])


@print_ops_name(verbose=False)
def count_softmax(ops: nn.Module, in_tensor, out_tensor):
    in_tensor = in_tensor[0]

    N, C, *HW = in_tensor.size()
    nelements = C * reduce(operator.mul, HW)  # CHW

    total_exp = nelements
    total_add = nelements - 1
    total_div = nelements
    total_ops = N * (total_exp + total_add + total_div)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_avgpool(ops: nn.Module, in_tensor, out_tensor):
    # total_add = torch.prod(torch.Tensor([ops.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = out_tensor.numel()
    total_ops = kernel_ops * num_elements

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_adap_avgpool(ops: nn.Module, in_tensor, out_tensor):
    kernel = torch.DoubleTensor([*(in_tensor[0].shape[2:])]) // torch.DoubleTensor(
        [*(out_tensor.shape[2:])]
    )
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = out_tensor.numel()
    total_ops = kernel_ops * num_elements

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


# TODO: verify the accuracy
def count_upsample(ops: nn.Module, in_tensor, out_tensor):
    if ops.mode not in (
        "nearest",
        "linear",
        "bilinear",
        "bicubic",
    ):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % ops.mode)
        return zero_ops(ops, in_tensor, out_tensor)

    if ops.mode == "nearest":
        return zero_ops(ops, in_tensor, out_tensor)

    in_tensor = in_tensor[0]
    if ops.mode == "linear":
        total_ops = out_tensor.nelement() * 5  # 2 muls + 3 add
    elif ops.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = out_tensor.nelement() * 11  # 6 muls + 5 adds
    elif ops.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] in_tensor [4x4] in_tensor [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = out_tensor.nelement() * (ops_solve_A + ops_solve_p)
    elif ops.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = out_tensor.nelement() * (13 * 2 + 5)

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_linear(ops: nn.Linear, in_tensor, out_tensor):
    # per output element
    total_mul = ops.in_features
    # total_add = ops.in_features - 1
    # total_add += 1 if ops.bias is not None else 0
    num_elements = out_tensor.numel()
    total_ops = total_mul * num_elements

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_bn(ops: nn.BatchNorm2d, in_tensor, out_tensor):
    """
    如何计算CNN中batch normalization的计算复杂度（FLOPs）？ - 采石工的回答 - 知乎
    https://www.zhihu.com/question/400039617/answer/1270642900
    PyTorch 源码解读之 BN & SyncBN：BN 与 多卡同步 BN 详解 - OpenMMLab的文章 - 知乎
    https://zhuanlan.zhihu.com/p/337732517
    pytorch BatchNorm参数详解，计算过程:
    https://blog.csdn.net/weixin_39228381/article/details/107896863
    https://github.com/sovrasov/flops-counter.pytorch/blob/469b7430ec7c6aa8c258da1bca2c04de81fc9613/ptflops/flops_counter.py#L285-L291
    :param base_ops: bn ops
    :param num_elements: C in_tensor H in_tensor W
    :return:
    """
    in_tensor = in_tensor[0]

    total_ops = in_tensor.numel()
    if ops.affine:
        # subtract, divide, gamma, beta
        # self.weight = Parameter(torch.Tensor(num_features))
        # self.bias = Parameter(torch.Tensor(num_features))
        total_ops *= 2

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_ln(ops: nn.LayerNorm, in_tensor, out_tensor):
    in_tensor = in_tensor[0]

    total_ops = in_tensor.numel()
    if ops.elementwise_affine:
        # self.weight = Parameter(torch.Tensor(*self.normalized_shape))
        # self.bias = Parameter(torch.Tensor(*self.normalized_shape))
        total_ops *= 2

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_in(ops: nn.InstanceNorm2d, in_tensor, out_tensor):
    in_tensor = in_tensor[0]

    total_ops = in_tensor.numel()
    if ops.affine:
        # subtract, divide, gamma, beta
        # self.weight = Parameter(torch.Tensor(num_features))
        # self.bias = Parameter(torch.Tensor(num_features))
        total_ops *= 2

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


def count_mha(ops: nn.MultiheadAttention, in_tensor, out_tensor):
    query, key, value, *_ = in_tensor
    total_ops = 0

    q_l, q_n, q_e = query.shape
    num_heads = ops.num_heads
    head_dim = ops.head_dim
    embed_dim = ops.embed_dim
    assert embed_dim == q_e

    # in_proj
    if ops._qkv_same_embed_dim:
        k_l = v_l = q_l
        k_n = v_n = q_n
        k_e = v_e = q_e
        total_ops += q_l * ops.in_proj_weight.shape[1] * ops.in_proj_weight.shape[0]
    else:
        k_l, k_n, k_e = key.shape
        v_l, v_n, v_e = value.shape
        assert q_e == k_e and k_n == v_n
        total_ops += q_l * ops.q_proj_weight.shape[1] * ops.q_proj_weight.shape[0]
        total_ops += k_l * ops.k_proj_weight.shape[1] * ops.k_proj_weight.shape[0]
        total_ops += v_l * ops.v_proj_weight.shape[1] * ops.v_proj_weight.shape[0]

    if ops.in_proj_weight is not None:
        total_ops += ops.in_proj_bias.numel()
    if ops.bias_k is not None:
        total_ops += ops.bias_k.numel()
    if ops.bias_v is not None:
        total_ops += ops.bias_v.numel()

    # attention
    # q, k, v -> l,bs*num_heads,head_dim -> bs*num_heads,l,head_dim
    if ops.add_zero_attn:
        k_l += 1
        v_l += 1
    # attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    total_ops += num_heads * q_l * head_dim * k_l
    # attn_output = torch.bmm(attn_output_weights, v)
    # assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    total_ops += num_heads * q_l * k_l * v_e

    # out_proj
    total_ops += q_l * ops.out_proj.weight.shape[1] * ops.out_proj.weight.shape[0]
    if ops.out_proj.bias is not None:
        total_ops += ops.out_proj.bias.numel()

    ops.total_ops += torch.DoubleTensor([int(total_ops)])


register_hooks = {
    nn.ZeroPad2d: zero_ops,  # padding does not involve any multiplication.
    # Convolution
    nn.Conv1d: count_convNd_ver2,
    nn.Conv2d: count_convNd_ver2,
    nn.Conv3d: count_convNd_ver2,
    nn.ConvTranspose1d: count_convNd_ver2,
    nn.ConvTranspose2d: count_convNd_ver2,
    nn.ConvTranspose3d: count_convNd_ver2,
    nn.Linear: count_linear,
    nn.Identity: zero_ops,
    # Attention
    nn.MultiheadAttention: count_mha,
    # Normalization
    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,
    nn.LayerNorm: count_ln,
    nn.InstanceNorm1d: count_in,
    nn.InstanceNorm2d: count_in,
    nn.InstanceNorm3d: count_in,
    nn.Dropout: zero_ops,
    # Activation
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.PReLU: count_relu_w_params,
    nn.LeakyReLU: count_relu_w_params,
    nn.Softmax: count_softmax,
    # Pooling
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    # Interpolation
    nn.Upsample: zero_ops,
    nn.UpsamplingBilinear2d: zero_ops,
    nn.UpsamplingNearest2d: zero_ops,
}

if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
    logging.warning(
        "You are using an old version PyTorch {version}, which THOP is not going to support in the future.".format(
            version=torch.__version__
        )
    )

if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    register_hooks.update({nn.SyncBatchNorm: count_bn})
