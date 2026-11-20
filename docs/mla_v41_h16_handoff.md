# A8W4 H16 Integration Handoff

2026-11-20. Base: `a53213dc474521a52ace1cecc079b70e3a0afafb`.
Changes are uncommitted; no commit or push was made.

## What Changed

`mla_sparse_prefill.py` adds a separate H16 eligibility branch and selects
`_asm_sparse_prefill_v41_h16`. The metadata capability API gains an optional
`num_kv_splits` keyword: H16 ASM accepts only None/integer 1. Native gfx1250
H16 A8W4 rejects larger splits explicitly. H16 FP8-prefix OPUS, gfx950 OPUS,
and plugin-first routing retain their existing behavior.

`mla_sparse_prefill_v41_asm.py` adds an H16-only checked wrapper and ctypes
binding. The existing H64 wrapper, raw binding and split heuristic are
unchanged. H16 checks shape, dtype, contiguous layout, device, base
alignment, row counts, output aliasing, fixed scale and the conservative
CSR entry limit. N=0 does not launch.

`asm_mla_sparse_prefill_v41.cu` appends `MlaV41H16Args` and a separate raw
entrypoint. Every field offset and the 128-byte size have static assertions.
The first 96 pointer bytes match H64 exactly; the kernel performs the
SWA-first swap internally. Launch: grid `(256,1,1)`, block `(128,1,1)`,
`passes=1`, null partial pointers, `query_count=N`, `persistent_groups=256`;
no cluster or dynamic LDS. The existing H64 C++ source remains an unchanged
prefix of the file. The existing JIT module config builds both entrypoints,
so no extra build registry is required.

The source-derived H16 query limit is the public API's `2**31-1`, not 256.
The PS loop advances by 256 until it reaches N. Q/output/indptr offsets use
64-bit addressing; this N limit prevents u32 next-query wraparound. Each CSR
index array is conservatively limited to `2**29-1` entries because LTD
scalar-buffer byte offsets are 32-bit. Source-local CSR row IDs remain
int32, not uint16. These are host acceptance limits, not measured GPU limits.

See `hsa/gfx1250/mla_v41/README.md` for the complete ABI, provenance, SHA256
values and reproducible LLVM build commands. The `.s` was copied unchanged;
the new `.co` is 50,680 bytes.

## Verification Status

- LLVM 23.1.5 assembly/link passed; ELF exports the intended symbol and `.kd`.
- Linked metadata: kernarg 128 B, static LDS 327680 B, private 0, wave32,
  SGPR/VGPR counts 106/1024. Block 128 is the source/launcher contract,
  not the metadata's maximum workgroup size of 1024.
- Disassembly confirms the count/group load, `ttmp9` initialization, u32
  query increment and terminal instruction at offset `0x3f48`. LLVM emits
  56 raw `.long` words, including TDM encodings; a full ISA audit is not claimed.
- SP3's excluding-stamp MD5 and the unchanged assembly SHA256 were checked.
  H64 code object SHA256 remains
  `ef2f135b06915be3487958d2f21c06da804cf7e5199772e2d4419ce6801b632e`.
- The dependency-free offline suite executes extracted production Python
  functions with metadata-only stubs. It checks H16/H64/other head counts,
  count/split/dtype/layout/scale/gate rules, default and explicit dispatch,
  OPUS/plugin behavior, validation, output aliasing and no-workspace launch.
  Its ctypes mirror is derived from the actual C++ struct and compared to
  the assembly metadata, initializer order and raw `<12QfI2Q2I` packing.
- All 19 offline tests passed. Python syntax compilation passed for all four
  changed/new Python files. No GPU tests were run on this host.
- No HIP/C++ compiler, PyTorch or pytest is available here. The compiled
  launcher/static assertions, GPU numerical equivalence and latency must
  still pass on MI400; no performance improvement is claimed.

Offline check:

```bash
python3 -m unittest op_tests.test_mla_sparse_prefill_v41_h16_offline -v
python3 -m py_compile \
  aiter/ops/mla_sparse_prefill.py \
  aiter/ops/mla_sparse_prefill_v41_asm.py \
  op_tests/test_mla_sparse_prefill_v41_h16_offline.py \
  op_tests/test_mla_sparse_prefill_v41_h16.py
git -c core.whitespace=cr-at-eol diff --check
git diff --stat
git diff -- aiter/ops/mla_sparse_prefill.py \
  aiter/ops/mla_sparse_prefill_v41_asm.py \
  csrc/py_itfs_cu/asm_mla_sparse_prefill_v41.cu
```

The whitespace check preserves the existing README's CRLF line endings
without treating carriage returns as added trailing whitespace.

## MI400 Acceptance

Use an idle, authorized gfx1250 B0+ device; do not interrupt another user's
workload. Use a complete checkout of this base plus all modified AND new
files, including the untracked `.co`, `.s` and both tests. This local sparse
checkout alone lacks parts of the normal AITER build/install tree. Use the
project's matching ROCm/PyTorch dependencies and install AITER from the
patched checkout using its normal installation flow. Do not disable the
ASM stepping gate.

From that full patched checkout, select the permitted device with
`HIP_VISIBLE_DEVICES`, then build the changed ctypes module in a fresh cache
so an older H64-only `.so` cannot hide the new entrypoint:

```bash
export AITER_META_DIR="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AITER_JIT_DIR="$(mktemp -d /tmp/aiter-h16-jit.XXXXXX)"
python3 -c 'from aiter.jit.core import build_ctypes_module; print(build_ctypes_module("module_mla_sparse_prefill_v41_asm"))'
python3 -m pytest -q op_tests/test_mla_sparse_prefill_v41_h16_offline.py
python3 -m pytest -q op_tests/test_mla_sparse_prefill_v41_h16.py
python3 -m pytest -q op_tests/test_mla_sparse_prefill_v41_asm.py
```

The H16 GPU suite uses an independent CPU decoder of the supplied FP8,
E8M0, packed E2M1 and E4M3 bytes plus FP32 attention. Its second oracle is
the unchanged H64 S1 kernel with zero extra Q heads, unit extra scales and
negative-infinity extra sink heads; compare the first 16 outputs. The
existing H64 acceptance threshold is used: rtol/atol 0.01, at most 1%
mismatching elements, all finite. Do not loosen it to hide failures.

Coverage includes N=0/1/96/255/256/257/513/769, repeated PS handoffs,
empty sources/queries, 63/64/65-row tails, 512/1025-row CSA, finite/no/mixed
sink, output guard regions, source row IDs 65535/65536, typed dtype aliases,
explicit split rejection, actual H16 entry selection and graph replay.
Record both oracle errors and failures before claiming GPU support.

Record compiler/runtime versions, actual loaded `.so` and `.co` identities,
gfx stepping, kernel symbol and grid. Performance measurement is separate:
use identical quantized inputs/CSR and preallocated sink/output, warm up
outside timing, and report eager and graph latency without counting input
preparation or oracle work. No GPU or performance result is supplied here.
