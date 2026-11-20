# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MI400-only H16 tests against decoded FP32 and zero-padded H64 S1.

Do not run on the offline integration host. Tolerances follow the existing
H64 suite: rtol=atol=0.01, at most 1% mismatching elements, all outputs finite.
These are acceptance thresholds, not a claim of measured H16 accuracy.
"""

import importlib
import math

import pytest

torch = pytest.importorskip("torch")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm GPU", allow_module_level=True)
_GFX = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
pytestmark = pytest.mark.skipif(_GFX != "gfx1250", reason="H16 PS CO requires gfx1250")
if _GFX != "gfx1250":
    pytest.skip("H16 PS CO requires gfx1250", allow_module_level=True)

from aiter.jit.utils.asm_guard import is_gfx1250_asm_supported  # noqa: E402
from aiter.ops.mla_sparse_prefill import (  # noqa: E402
    get_mla_sparse_prefill_v41_backend,
    mla_sparse_prefill_v41,
)
from aiter.utility import dtypes  # noqa: E402

if not is_gfx1250_asm_supported():
    pytest.skip("requires the gfx1250 ASM stepping gate", allow_module_level=True)

SCALE = 1 / math.sqrt(512)


def _case(n, csa_rows=65, swa_rows=65, *, ragged=False, sink="finite", high_rows=False):
    gen = torch.Generator().manual_seed(41)

    def fp8(shape):
        return (torch.randn(shape, generator=gen) * 0.5).to(torch.float8_e4m3fn)

    def e8m0(shape):
        return torch.randint(125, 129, shape, generator=gen, dtype=torch.uint8)

    def csr(rows):
        indptr, indices = [0], []
        for token in range(n):
            if high_rows:
                selected = [65535, 65536]
            else:
                length = min(rows, (0, 1, 63, 64, 65, rows)[token % 6]) if ragged else rows
                selected = [(i + token * 17) % rows for i in range(length)]
            indices.extend(selected)
            indptr.append(len(indices))
        return torch.tensor(indices, dtype=torch.int32), torch.tensor(indptr, dtype=torch.int32)

    prefix, pptr = csr(csa_rows)
    extend, eptr = csr(swa_rows)
    scale_values = torch.tensor([0.0625, 0.125, 0.25, 0.5], dtype=torch.float32)
    csa_scale = scale_values[
        torch.randint(0, len(scale_values), (csa_rows, 32), generator=gen)
    ].to(torch.float8_e4m3fn)
    attn_sink = torch.linspace(-1, 1, 16, dtype=torch.float32)
    if sink == "mixed":
        attn_sink[::2] = -math.inf
    inputs = {
        "q": fp8((n, 16, 512)),
        "q_scale": e8m0((n, 16, 16)),
        "unified_kv": torch.randint(0, 256, (csa_rows, 256), generator=gen, dtype=torch.uint8),
        "unified_kv_scale": csa_scale.view(torch.uint8),
        "kv_indices_prefix": prefix,
        "kv_indptr_prefix": pptr,
        "kv": fp8((swa_rows, 512)),
        "kv_scale": e8m0((swa_rows, 16)),
        "kv_indices_extend": extend,
        "kv_indptr_extend": eptr,
        "attn_sink": attn_sink,
    }
    inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
    if sink == "none":
        inputs["attn_sink"] = None
    return inputs


def _decoded_reference(inputs):
    """Decode the exact supplied bytes on CPU, independently of AITER quantizers."""
    cpu = {
        name: tensor.detach().cpu() if tensor is not None else None
        for name, tensor in inputs.items()
    }

    def decode_fp8(data, scale):
        exp = scale.view(torch.uint8).to(torch.int32) - 127
        factor = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)
        return data.float() * factor.repeat_interleave(32, dim=-1)

    q = decode_fp8(cpu["q"], cpu["q_scale"])
    swa = decode_fp8(cpu["kv"], cpu["kv_scale"])
    packed = cpu["unified_kv"].view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    csa_scale = cpu["unified_kv_scale"].view(torch.float8_e4m3fn).float()
    csa = e2m1[codes] * csa_scale.repeat_interleave(16, dim=-1)
    sink = cpu["attn_sink"]
    if sink is None:
        sink = torch.full((16,), -math.inf)
    result = torch.zeros(q.shape, dtype=torch.float32)
    pptr, eptr = cpu["kv_indptr_prefix"].tolist(), cpu["kv_indptr_extend"].tolist()
    for token in range(q.shape[0]):
        prefix = cpu["kv_indices_prefix"][pptr[token] : pptr[token + 1]].long()
        extend = cpu["kv_indices_extend"][eptr[token] : eptr[token + 1]].long()
        rows = torch.cat((csa[prefix], swa[extend]))
        if not rows.numel():
            continue
        scores = (q[token] @ rows.T) * SCALE
        maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink[:, None])
        weight = (scores - maximum).exp()
        denom = weight.sum(dim=-1, keepdim=True) + (sink[:, None] - maximum).exp()
        result[token] = (weight / denom) @ rows
    return result


def _h64_oracle(inputs):
    n = inputs["q"].shape[0]
    padded = dict(inputs)
    padded["q"] = torch.zeros(
        (n, 64, 512), dtype=torch.uint8, device="cuda"
    ).view(inputs["q"].dtype)
    padded["q"].view(torch.uint8)[:, :16].copy_(inputs["q"].view(torch.uint8))
    padded["q_scale"] = torch.full((n, 64, 16), 127, dtype=torch.uint8, device="cuda")
    padded["q_scale"][:, :16].copy_(inputs["q_scale"].view(torch.uint8))
    if inputs["attn_sink"] is not None:
        padded["attn_sink"] = torch.full((64,), -math.inf, device="cuda")
        padded["attn_sink"][:16].copy_(inputs["attn_sink"])
    return mla_sparse_prefill_v41(
        **padded, softmax_scale=SCALE, num_kv_splits=1, backend="asm"
    )[:, :16].float().cpu()


def _check(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    actual = actual.float().cpu()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    if not actual.numel():
        return
    close = torch.isclose(actual, expected, rtol=0.01, atol=0.01)
    fraction = (~close).float().mean().item()
    max_abs = (actual - expected).abs().max().item()
    assert fraction <= 0.01, f"mismatch fraction={fraction}, max_abs={max_abs}"


def _invoke_and_check(inputs, *, splits=None, backend=None):
    n = inputs["q"].shape[0]
    storage = torch.full((n * 16 * 512 + 32,), 123, dtype=torch.bfloat16, device="cuda")
    out = storage[16:-16].view(n, 16, 512)
    out.fill_(math.nan)
    actual = mla_sparse_prefill_v41(
        **inputs, softmax_scale=SCALE, out=out, num_kv_splits=splits, backend=backend
    )
    assert actual.data_ptr() == out.data_ptr()
    _check(actual, _decoded_reference(inputs))
    _check(actual, _h64_oracle(inputs))
    assert torch.all(storage[:16] == 123) and torch.all(storage[-16:] == 123)
    return actual


@pytest.mark.parametrize("n", [0, 1, 96, 255, 256, 257, 513, 769])
@pytest.mark.parametrize("splits", [None, 1])
def test_persistent_query_loop(n, splits):
    assert get_mla_sparse_prefill_v41_backend(
        "cuda", (n, 16, 512), unified_kv_dtype=torch.uint8, num_kv_splits=splits
    ) == "asm"
    _invoke_and_check(_case(n), splits=splits)


@pytest.mark.parametrize("csa_rows,swa_rows", [
    (0, 0), (0, 1), (1, 0), (63, 63), (64, 64), (129, 128),
    (512, 128), (1025, 128),
])
def test_sources_and_tile_tails(csa_rows, swa_rows):
    _invoke_and_check(_case(96, csa_rows, swa_rows), backend="asm")


@pytest.mark.parametrize("sink", ["none", "finite", "mixed"])
def test_ragged_csr_and_empty_query_handoffs(sink):
    _invoke_and_check(_case(513, ragged=True, sink=sink))


def test_int32_row_ids_above_65535():
    _invoke_and_check(_case(2, 65537, 65537, high_rows=True))


def test_typed_scale_and_fp4_aliases():
    inputs = _case(96)
    inputs["q_scale"] = inputs["q_scale"].view(dtypes.fp8_e8m0)
    inputs["kv_scale"] = inputs["kv_scale"].view(dtypes.fp8_e8m0)
    inputs["unified_kv"] = inputs["unified_kv"].view(dtypes.fp4x2)
    inputs["unified_kv_scale"] = inputs["unified_kv_scale"].view(torch.float8_e4m3fn)
    _invoke_and_check(inputs)


def test_native_h16_entry_and_no_workspace(monkeypatch):
    native = importlib.import_module("aiter.ops.mla_sparse_prefill_v41_asm")
    entry = native.asm_mla_sparse_prefill_v41_h16_fwd
    calls = []

    def record(*args):
        calls.append(args)
        return entry(*args)

    monkeypatch.setattr(native, "asm_mla_sparse_prefill_v41_h16_fwd", record)
    inputs = _case(96, sink="none")
    _invoke_and_check(inputs)
    assert len(calls) == 1 and len(calls[0]) == 13
    assert calls[0][10].shape == (16,)
    assert torch.isneginf(calls[0][10]).all()
    assert calls[0][2] is inputs["unified_kv"] and calls[0][6] is inputs["kv"]


@pytest.mark.parametrize("backend", [None, "asm"])
def test_split_greater_than_one_fails(backend):
    with pytest.raises(ValueError, match="H16 asm supports S1 only"):
        mla_sparse_prefill_v41(
            **_case(1), softmax_scale=SCALE, num_kv_splits=2, backend=backend
        )


def test_graph_replay():
    inputs = _case(257, ragged=True)
    out = _invoke_and_check(inputs)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        mla_sparse_prefill_v41(**inputs, softmax_scale=SCALE, out=out, backend="asm")
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_sparse_prefill_v41(**inputs, softmax_scale=SCALE, out=out, backend="asm")
    for _ in range(3):
        out.fill_(math.nan)
        graph.replay()
        _check(out, _decoded_reference(inputs))
