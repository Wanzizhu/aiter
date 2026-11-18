# SPDX-License-Identifier: MIT
import torch
import triton
import triton.language as tl
from aiter import dtypes

# The FP8 an activation is quantized to, and the three forms every writer and
# reader of one of those rows needs. gfx942 is FNUZ and every other target is
# OCP E4M3; they differ in exponent bias, so a row written under one encoding
# and read under the other is garbage. Taking all three from one declaration is
# what keeps a clamp, a cast and a bitcast from disagreeing. The two a kernel
# reads are `tl.constexpr`: Triton refuses a plain global inside `@triton.jit`.
FP8_DTYPE = dtypes.fp8
FP8_MAX = tl.constexpr(torch.finfo(FP8_DTYPE).max)
FP8_TL_DTYPE = tl.constexpr(
    tl.float8e4b8 if FP8_DTYPE is torch.float8_e4m3fnuz else tl.float8e4nv
)
# The amax floor of every FP8 activation quant: aiter's MoE input quant's, so
# one model's quant sites all agree
AMAX_FLOOR = tl.constexpr(1e-10)


@triton.jit
def ceil_pow2_code(value):
    bits = value.to(tl.uint32, bitcast=True)
    return (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)


@triton.jit
def quantize_fp8_kernel(X, Y, S, SIZE, DEQUANT: tl.constexpr):
    # SIZE is the token count. A `tl.constexpr` here builds one kernel per
    # prefill chunk length and per batch size, and it only bounds the tiles.
    groups = tl.program_id(0) * 32 + tl.arange(0, 32)
    offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets, offsets < SIZE, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), AMAX_FLOOR)
    code = ceil_pow2_code(amax * (1.0 / FP8_MAX))
    scale = (code << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x / scale[:, None], -FP8_MAX), FP8_MAX).to(FP8_TL_DTYPE)
    if DEQUANT:
        tl.store(Y + offsets, q.to(tl.float32) * scale[:, None], offsets < SIZE)
    else:
        tl.store(Y + offsets, q, offsets < SIZE)
        tl.store(S + groups, code.to(tl.uint8), groups < SIZE // 32)

