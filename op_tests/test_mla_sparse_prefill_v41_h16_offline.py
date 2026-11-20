# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-only contract tests; no torch, ROCm, pytest or AITER import required.

Run: python3 -m unittest op_tests.test_mla_sparse_prefill_v41_h16_offline -v
The production functions are AST-loaded with metadata-only tensor/runtime stubs.
This tests host control flow and ABI, not compiled HIP or GPU correctness.
"""

import __future__
import ast
import ctypes
import hashlib
import itertools
import math
from numbers import Real
from pathlib import Path
import re
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from contextlib import nullcontext


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "aiter/ops/mla_sparse_prefill.py"
BACKEND = ROOT / "aiter/ops/mla_sparse_prefill_v41_asm.py"
CPP = ROOT / "csrc/py_itfs_cu/asm_mla_sparse_prefill_v41.cu"
SYMBOL = "mla_v41_qh16_ps_grid256x_mono_v1"
ASM = ROOT / "hsa/gfx1250/mla_v41" / f"{SYMBOL}.s"
SCALE = 1 / math.sqrt(512)


class DType:
    def __init__(self, name, size=1):
        self.name, self.size = name, size

    def __repr__(self):
        return self.name


DTYPES = SimpleNamespace(
    fp8=DType("fp8"),
    fp4x2=DType("fp4x2"),
    fp8_e8m0=DType("e8m0"),
    uint8=DType("uint8"),
    int32=DType("int32", 4),
    float32=DType("float32", 4),
    bfloat16=DType("bfloat16", 2),
    float16=DType("float16", 2),
)


class Tensor:
    _next_ptr = 4096

    def __init__(self, shape, dtype, device=None, *, contiguous=True, pointer=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or SimpleNamespace(type="cuda", index=0)
        self.contiguous = contiguous
        self.pointer = Tensor._next_ptr if pointer is None else pointer
        Tensor._next_ptr += (self.numel() * self.element_size() + 511) // 256 * 256

    def dim(self):
        return len(self.shape)

    def size(self, dim):
        return self.shape[dim]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.dtype.size

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous

    def stride(self, dim=None):
        strides = tuple(math.prod(self.shape[i + 1 :]) for i in range(self.dim()))
        if not self.contiguous:
            strides = (strides[0] + 32, *strides[1:])
        return strides if dim is None else strides[dim]

    def item(self):
        raise AssertionError("host contract must not read device values")


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    code = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)


class HostContractTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            Tensor=Tensor,
            dtype=DType,
            uint8=DTYPES.uint8,
            int32=DTYPES.int32,
            float32=DTYPES.float32,
            float16=DTYPES.float16,
            bfloat16=DTYPES.bfloat16,
            float8_e4m3fn=DTYPES.fp8,
            version=SimpleNamespace(hip="test"),
            cuda=SimpleNamespace(device=lambda device: nullcontext()),
            empty=Mock(side_effect=lambda shape, **kw: Tensor(shape, **kw)),
            full=Mock(side_effect=lambda shape, fill, **kw: Tensor(shape, **kw)),
        )
        self.ns = {
            "torch": self.torch,
            "dtypes": DTYPES,
            "math": math,
            "struct": struct,
            "Real": Real,
            "_E8M0": (DTYPES.uint8, DTYPES.fp8_e8m0),
            "_OPUS_V41_KERNELS": {
                DTYPES.fp8: ((object(), object()), "v41_a8w8", (DTYPES.uint8, DTYPES.fp8_e8m0)),
                DTYPES.fp4x2: ((object(), None), "v41_a8w4", (DTYPES.uint8, DTYPES.fp8)),
            },
            "_v41_runtime_arch": Mock(return_value="gfx1250"),
            "get_gfx_runtime": Mock(return_value="gfx1250"),
            "is_gfx1250_asm_supported": Mock(return_value=True),
            "require_gfx1250_asm": Mock(),
            "asm_mla_sparse_prefill_v41_h16_fwd": Mock(),
            "_asm_sparse_prefill_v41": Mock(return_value="H64"),
            "_opus_sparse_prefill_v41": Mock(return_value="OPUS"),
        }
        load_functions(
            BACKEND,
            ("_check_tensor", "_check_v41_h16_splits", "_asm_sparse_prefill_v41_h16"),
            self.ns,
        )
        load_functions(
            PUBLIC,
            (
                "_check_backend", "_v41_asm_eligible", "_v41_opus_eligible",
                "get_mla_sparse_prefill_v41_backend", "_v41_tensor_layout",
                "mla_sparse_prefill_v41",
            ),
            self.ns,
        )
        operation = SimpleNamespace(name="mla_sparse_prefill_v41")
        self.ns["_mla_sparse_prefill_v41_routed"] = SimpleNamespace(
            __aiter_plugin_op__=operation
        )
        self.ns["_plugin"] = SimpleNamespace(
            _routed_ops={operation.name: SimpleNamespace(canonicalize=None)},
            _dispatch=lambda op, canonicalize, native, args, kw: native(*args, **kw),
        )

    def args(self, n=96, h=16):
        t = Tensor
        return [
            t((n, h, 512), DTYPES.fp8),
            t((n, h, 16), DTYPES.uint8),
            t((65, 256), DTYPES.uint8),
            t((65, 32), DTYPES.uint8),
            t((n * 65,), DTYPES.int32),
            t((n + 1,), DTYPES.int32),
            t((128, 512), DTYPES.fp8),
            t((128, 16), DTYPES.uint8),
            t((n * 128,), DTYPES.int32),
            t((n + 1,), DTYPES.int32),
            t((h,), DTYPES.float32),
        ]

    def query(self, n=96, h=16, **kw):
        return self.ns["get_mla_sparse_prefill_v41_backend"](
            "cuda:0", (n, h, 512), unified_kv_dtype=DTYPES.uint8, **kw
        )

    def invoke(self, args=None, **kw):
        return self.ns["mla_sparse_prefill_v41"](
            *(self.args() if args is None else args), softmax_scale=SCALE, **kw
        )

    def test_query_counts_and_heads(self):
        for h, n, expected in [
            (16, 0, "asm"), (16, 1, "asm"), (16, 96, "asm"),
            (16, 255, "asm"), (16, 256, "asm"), (16, 257, "asm"),
            (16, 513, "asm"), (16, 65537, "asm"), (16, 2**31 - 1, "asm"),
            (16, 2**31, None), (64, 0, "asm"), (64, 65536, "asm"),
            (64, 65537, None), (32, 96, None), (128, 96, None),
        ]:
            with self.subTest(h=h, n=n):
                self.assertEqual(self.query(n, h), expected)

    def test_h16_split_eligibility_and_clear_failure(self):
        for splits in (None, 1, 2, 16, 0, -1, True, False, 1.0, "1"):
            accepted = splits is None or type(splits) is int and splits == 1
            with self.subTest(splits=splits):
                self.assertEqual(self.query(num_kv_splits=splits), "asm" if accepted else None)
                for backend in (None, "asm"):
                    if accepted:
                        self.invoke(num_kv_splits=splits, backend=backend)
                    else:
                        with self.assertRaisesRegex(ValueError, "H16 asm supports S1 only"):
                            self.invoke(num_kv_splits=splits, backend=backend)
        self.ns["_opus_sparse_prefill_v41"].assert_not_called()
        self.ns["_asm_sparse_prefill_v41"].assert_not_called()

    def test_h64_split_dispatch_is_unchanged(self):
        for splits in (None, 1, 2, 16):
            self.assertEqual(self.query(h=64, num_kv_splits=splits), "asm")
            args = self.args(h=64)
            self.assertEqual(self.invoke(args, num_kv_splits=splits), "H64")
            self.ns["_asm_sparse_prefill_v41"].assert_called_with(*args, SCALE, None, splits)
        self.ns["asm_mla_sparse_prefill_v41_h16_fwd"].assert_not_called()

    def test_gfx950_h16_opus_splits_remain_supported(self):
        self.ns["_v41_runtime_arch"].return_value = "gfx950"
        self.assertEqual(self.query(num_kv_splits=4), "opus")
        self.assertEqual(self.invoke(num_kv_splits=4), "OPUS")

    def test_fp8_prefix_h16_opus_splits_remain_supported(self):
        args = self.args()
        args[2] = Tensor((65, 512), DTYPES.fp8)
        args[3] = Tensor((65, 16), DTYPES.uint8)
        self.assertEqual(self.invoke(args, num_kv_splits=4), "OPUS")

    def test_dtype_rules_for_both_head_counts(self):
        eligible = self.ns["_v41_asm_eligible"]
        base = [DTYPES.fp8, DTYPES.uint8, DTYPES.uint8, DTYPES.uint8, DTYPES.fp8, DTYPES.uint8]
        for h in (16, 64):
            for index in range(6):
                bad = base.copy()
                bad[index] = DTYPES.float16
                with self.subTest(h=h, field=index):
                    self.assertFalse(eligible("gfx1250", (96, h, 512), bad, "contiguous", SCALE))
            for qs, p, ps, ks in itertools.product(
                (DTYPES.uint8, DTYPES.fp8_e8m0),
                (DTYPES.uint8, DTYPES.fp4x2),
                (DTYPES.uint8, DTYPES.fp8),
                (DTYPES.uint8, DTYPES.fp8_e8m0),
            ):
                self.assertTrue(eligible(
                    "gfx1250", (96, h, 512),
                    (DTYPES.fp8, qs, p, ps, DTYPES.fp8, ks), "contiguous", SCALE,
                ))

    def test_arch_layout_scale_and_gate(self):
        for gfx in (None, "gfx942", "gfx950"):
            self.ns["_v41_runtime_arch"].return_value = gfx
            self.assertEqual(self.query(), "opus" if gfx == "gfx950" else None)
        self.ns["_v41_runtime_arch"].return_value = "gfx1250"
        for h in (16, 64):
            self.assertIsNone(self.query(h=h, layout="row_strided"))
            self.assertIsNone(self.query(h=h, softmax_scale=0.04))
            fp32_scale = struct.unpack("<f", struct.pack("<f", SCALE))[0]
            self.assertEqual(self.query(h=h, softmax_scale=fp32_scale), "asm")
            self.ns["is_gfx1250_asm_supported"].return_value = False
            self.assertIsNone(self.query(h=h))
            self.ns["is_gfx1250_asm_supported"].return_value = True
        for scale in (math.nan, math.inf, 1e100):
            with self.assertRaisesRegex(ValueError, "finite FP32"):
                self.query(softmax_scale=scale)
        spec = (DTYPES.fp8, DTYPES.uint8, DTYPES.uint8, DTYPES.uint8, DTYPES.fp8, DTYPES.uint8)
        self.assertFalse(self.ns["_v41_asm_eligible"](
            "gfx1250", (96, 16, 256), spec, "contiguous", SCALE
        ))

    def test_capability_does_not_allocate_or_launch(self):
        self.assertEqual(self.query(), "asm")
        self.torch.empty.assert_not_called()
        self.torch.full.assert_not_called()
        self.ns["asm_mla_sparse_prefill_v41_h16_fwd"].assert_not_called()

    def test_h16_launch_order_and_no_workspace(self):
        args = self.args()
        args[-1] = None
        out = self.invoke(args)
        self.assertEqual(out.shape, (96, 16, 512))
        self.assertIs(out.dtype, DTYPES.bfloat16)
        call = self.ns["asm_mla_sparse_prefill_v41_h16_fwd"].call_args.args
        self.assertEqual(len(call), 13)
        self.assertEqual(call[:10], tuple(args[:10]))
        self.assertEqual(call[10].shape, (16,))
        self.assertEqual(call[11:], (out, SCALE))
        self.torch.empty.assert_called_once()
        self.torch.full.assert_called_once_with(
            (16,), -math.inf, dtype=DTYPES.float32, device=args[0].device
        )
        self.ns["require_gfx1250_asm"].assert_called_once()

    def test_provided_output_and_empty_queries(self):
        for n in (0, 1, 257):
            with self.subTest(n=n):
                args = self.args(n=n)
                out = Tensor((n, 16, 512), DTYPES.bfloat16)
                self.ns["asm_mla_sparse_prefill_v41_h16_fwd"].reset_mock()
                self.assertIs(self.invoke(args, out=out), out)
                self.assertEqual(
                    self.ns["asm_mla_sparse_prefill_v41_h16_fwd"].call_count, int(n > 0)
                )
        self.torch.empty.assert_not_called()

    def test_bad_tensor_metadata_is_rejected(self):
        mutations = [
            (0, "shape", (96, 16, 256), "shape"),
            (1, "shape", (96, 64, 16), "shape"),
            (2, "contiguous", False, "contiguous"),
            (2, "pointer", 257, "aligned"),
            (3, "shape", (64, 32), "CSA data/scale"),
            (4, "dtype", DTYPES.float32, "dtype"),
            (5, "shape", (96,), "shape"),
            (6, "dtype", DTYPES.uint8, "dtype"),
            (7, "shape", (127, 16), "SWA data/scale"),
            (8, "shape", (2**29,), "32-bit CSR byte offsets"),
            (10, "shape", (64,), "shape"),
            (10, "dtype", DTYPES.bfloat16, "dtype"),
            (0, "device", SimpleNamespace(type="cpu"), "HIP device"),
            (1, "device", SimpleNamespace(type="cuda", index=1), "must be on"),
            (0, "shape", (2**31, 16, 512), "N must be"),
        ]
        for index, field, value, message in mutations:
            with self.subTest(index=index, field=field):
                args = self.args()
                setattr(args[index], field, value)
                with self.assertRaisesRegex(ValueError, message):
                    self.invoke(args, backend="asm")
        self.ns["asm_mla_sparse_prefill_v41_h16_fwd"].assert_not_called()

    def test_output_alias_and_shape_are_rejected(self):
        args = self.args()
        for source in (args[0], args[2], args[10]):
            out = Tensor((96, 16, 512), DTYPES.bfloat16, pointer=source.data_ptr())
            with self.assertRaisesRegex(ValueError, "out must not overlap"):
                self.invoke(args, out=out)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.invoke(args, out=Tensor((96, 64, 512), DTYPES.bfloat16))

    def test_explicit_h16_scale_and_runtime_guards(self):
        args = self.args()
        with self.assertRaisesRegex(ValueError, "softmax_scale"):
            self.ns["_asm_sparse_prefill_v41_h16"](*args, 0.04, None)
        self.ns["get_gfx_runtime"].return_value = "gfx950"
        with self.assertRaisesRegex(RuntimeError, "requires gfx1250"):
            self.invoke(args, backend="asm")
        self.ns["get_gfx_runtime"].return_value = "gfx1250"
        self.ns["require_gfx1250_asm"].side_effect = RuntimeError("stepping gate")
        with self.assertRaisesRegex(RuntimeError, "stepping gate"):
            self.invoke(args, backend="asm")

    def test_plugins_keep_first_refusal(self):
        self.ns["_plugin"]._dispatch = lambda *args: "provider"
        self.assertEqual(self.invoke(num_kv_splits=4), "provider")
        self.ns["_v41_runtime_arch"].assert_not_called()


class AbiTests(unittest.TestCase):
    def setUp(self):
        self.cpp = CPP.read_text()
        self.asm = ASM.read_text()
        body = re.search(
            r"struct __attribute__\(\(packed\)\) MlaV41H16Args\s*\{(.*?)\};",
            self.cpp, re.S,
        )[1]
        types = {"const void*": ctypes.c_uint64, "void*": ctypes.c_uint64,
                 "float": ctypes.c_float, "uint32_t": ctypes.c_uint32}
        fields = []
        for line in body.strip().splitlines():
            match = re.fullmatch(r"\s*(const void\*|void\*|float|uint32_t)\s+(\w+);", line)
            self.assertIsNotNone(match, line)
            fields.append((match[2], types[match[1]]))
        self.args_type = type("H16Args", (ctypes.LittleEndianStructure,), {
            "_pack_": 1, "_fields_": fields,
        })

    def test_cpp_layout_matches_assembly_metadata_and_assertions(self):
        self.assertEqual(ctypes.sizeof(self.args_type), 128)
        metadata = re.findall(r"\.name: (\w+), \.size: (\d+), \.offset: (\d+)", self.asm)
        aliases = {"q_nope": "q", "unified_kv_nope": "unified_kv", "kv_nope": "kv"}
        self.assertEqual(len(metadata), 18)
        for (name, size, offset), (field, ctype) in zip(metadata, self.args_type._fields_):
            self.assertEqual(field, aliases.get(name, name))
            self.assertEqual(ctypes.sizeof(ctype), int(size))
            self.assertEqual(getattr(self.args_type, field).offset, int(offset))
            self.assertIn(f"CHECK_H16_OFFSET({field}, {offset});", self.cpp)
        self.assertIn('static_assert(sizeof(MlaV41H16Args) == 128', self.cpp)

    def test_pointer_order_s1_constants_and_raw_packing(self):
        names = [field for field, _ in self.args_type._fields_[:12]]
        init = re.search(r"MlaV41H16Args args\{(.*?)\};", self.cpp, re.S)[1]
        expressions = [part.strip() for part in init.split(",")]
        self.assertEqual(expressions, [
            *(f"{name}->data_ptr()" for name in names),
            "softmax_scale", "1", "nullptr", "nullptr", "static_cast<uint32_t>(n)", "256",
        ])
        pointers = [0x100000000 + i * 0x100 for i in range(12)]
        for n in (0, 96, 257, 2**31 - 1):
            packed = bytes(self.args_type(*pointers, SCALE, 1, 0, 0, n, 256))
            self.assertEqual(packed, struct.pack("<12QfI2Q2I", *pointers, SCALE, 1, 0, 0, n, 256))

    def test_ctypes_signature_matches_cpp_entry(self):
        tree = ast.parse(BACKEND.read_text())
        name = "asm_mla_sparse_prefill_v41_h16_fwd"
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        entry = self.cpp.split(f"AITER_CTYPES_DEFINE_ENTRYPOINT_VOID({name},", 1)[1]
        signature = entry.split("hipStream_t stream)", 1)[0]
        names = re.findall(r"(?:aiter_tensor_t\*|float)\s+(\w+)", signature)
        self.assertEqual(names, [arg.arg for arg in fn.args.args])

    def test_fixed_launch_and_provenance(self):
        self.assertIn(f'static AiterAsmKernel kernel("{SYMBOL}",', self.cpp)
        self.assertIn(f'"mla_v41/{SYMBOL}.co"', self.cpp)
        self.assertIn(
            "kernel.launch_kernel({&args, &arg_size, 256, 1, 1, 128, 1, 1, stream});",
            self.cpp,
        )
        self.assertIn(".amdhsa_group_segment_fixed_size 327680", self.asm)
        self.assertIn(".amdhsa_kernarg_size 128", self.asm)
        self.assertIn("SP3_MD5(excl_stamp)=a679bc89bee9ce5bc31b82a0c5831098", self.asm)
        self.assertEqual(hashlib.sha256(ASM.read_bytes()).hexdigest(),
                         "68e9453672de984e6de9b9a16e217867edcd723f0d623297d8496f205a8604d4")

    def test_code_object_identity(self):
        code_object = ASM.with_suffix(".co")
        self.assertEqual(code_object.stat().st_size, 50680)
        self.assertEqual(
            hashlib.sha256(code_object.read_bytes()).hexdigest(),
            "eb1a574bcab24da7f38cb10268779246387fb70610b74bf6f9e46fd00177bb78",
        )


if __name__ == "__main__":
    unittest.main()
