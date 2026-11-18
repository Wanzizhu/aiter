# SPDX-License-Identifier: MIT
# Review baseline: unchanged ROCm/ATOM native functions; imports mapped to this path.
import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.quant.fp8_group32 import (
    FP8_DTYPE,
    quantize_fp8_kernel,
)


def _check_quant_input(x, group):
    if x.ndim < 1 or x.shape[-1] % group:
        raise ValueError(f"Last dimension must be divisible by {group}")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Quantization requires a floating-point CUDA/ROCm tensor")
    return x.contiguous()


def quantize_fp8(x: torch.Tensor, *, dequantize: bool = False):
    """E4M3, group32 E8M0 ceil scales; optionally return QAT values in x.dtype."""
    x = _check_quant_input(x, 32)
    output = torch.empty_like(x, dtype=x.dtype if dequantize else FP8_DTYPE)
    scales = (
        None
        if dequantize
        else torch.empty(
            (*x.shape[:-1], x.shape[-1] // 32),
            device=x.device,
            dtype=torch.float8_e8m0fnu,
        )
    )
    if x.numel():
        quantize_fp8_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            output,
            None if scales is None else scales.view(torch.uint8),
            x.numel(),
            dequantize,
        )
    return output if dequantize else (output, scales)
