# MLA V4.1 H64 fused A8W4

`mla_sparse_prefill_v41(..., backend="asm")` supports H=64, D=512 on gfx1250 B0+.
Q/SWA use FP8 with E8M0 scales, CSA uses NVFP4 with E4M3 scales, and output is BF16.

- Inputs must be contiguous. CSA data `[P,256]` and scales `[P,32]` are separate
  arrays. Convert interleaved 288-byte rows to these arrays before calling the API.
- `softmax_scale` must equal the FP32 value of `1/sqrt(512)`.
  Sink values must be finite or negative infinity. `attn_sink=None` allocates
  and fills a negative-infinity tensor; prepare the tensor before kernel timing.
- `num_kv_splits` accepts 1..16. With `None`, N > 64 uses 1 split. For N <= 64,
  an average of 512 CSA prefix rows uses 1 split; 1024 uses 4 splits at N <= 32
  and 2 otherwise. Other prefix lengths use the ASM fallback heuristic. SWA
  rows do not count toward the prefix average.
- Grid `(S,N,1)`, block `(128,1,1)`; S>1 uses cluster `(S,1,1)` and fused reduction.
  The API allocates FP32 workspaces `[N,S,65,512]` and `[N,S,64]` for S>1.
- Code object `fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41.co`.
- Symbol `fused_mla_a8w4_qh64_1tg_16mx4_64nx1_sparse_np_v41`; 120-byte arguments, 327680-byte static LDS.
- CO SHA256: `ef2f135b06915be3487958d2f21c06da804cf7e5199772e2d4419ce6801b632e`.

# MLA V4.1 H16 persistent A8W4

`mla_sparse_prefill_v41(..., backend="asm")` also supports H=16, D=512 on
gfx1250 under the same B0+ ASM runtime gate. Automatic native dispatch selects
this H16 kernel for eligible A8W4 inputs; the H64 kernel and its split policy
are unchanged.

- Q `[N,16,512]` and SWA `[R,512]` are E4M3 FP8. Their separate scale arrays
  `[N,16,16]` and `[R,16]` are E8M0 (uint8 or fp8_e8m0), one per 32 values.
- CSA data `[P,256]` is packed E2M1 (uint8 or fp4x2, even value in the low
  nibble). CSA scales `[P,32]` are E4M3 (uint8 or float8_e4m3fn), one per
  16 values. All arrays must be contiguous on the query device, with matching
  data/scale row counts. Interleaved 288-byte CSA or 528-byte SWA rows must be
  explicitly separated before calling. Data/output bases require 16-byte
  alignment, scale/CSR bases require 4-byte alignment.
- Sink `[16]` is FP32, finite or negative infinity; `None` prepares a
  negative-infinity tensor. Output `[N,16,512]` is BF16 and must not overlap
  any input. `softmax_scale` must have the FP32 value of `1/sqrt(512)`.
- S1 only: `num_kv_splits` accepts `None` or integer `1`. An explicit larger
  split raises a clear error in native H16 dispatch, not a fallback.
  There is no split heuristic, cluster, dynamic LDS, or split workspace.
- Grid `(256,1,1)`, block `(128,1,1)`, `persistent_groups=256`. A TG with
  x-index `x` processes queries `x, x+256, x+512, ...` while below
  `query_count=N`. TGs without a query exit before tensor input loads.
  N=0 returns without launching.
- Accepted API range: `0 <= N <= 2147483647` (not N<=256 or the H64 N<=65536
  restriction). The SP3 uses an unsigned 32-bit query counter and next-query
  addition; Q, Q-scale, output and indptr offsets use 64-bit address
  arithmetic. The native capability API's signed-int32 ceiling also keeps
  `last_query+256` below unsigned wraparound. This is a source-derived host
  limit, not a GPU-tested maximum.
- CSR indices/indptr are int32. The caller supplies monotone indptr arrays
  `[N+1]` starting at zero, bounded by the corresponding index array, and
  nonnegative source-local row IDs within the corresponding source. CSR
  determines all visibility, including causality. No device values are read
  during host validation. Each H16 index array is conservatively capped at
  `2**29-1` entries (less than 2 GiB), below the kernel's 32-bit scalar-buffer
  byte-offset ceiling and leaving headroom for speculative LTD lookahead.
  This entry-count limit is not a 16-bit source-row-ID restriction.
- Code object `mla_v41_qh16_ps_grid256x_mono_v1.co`.
- Symbol `mla_v41_qh16_ps_grid256x_mono_v1`; 128-byte arguments,
  327680-byte static LDS; private segment 0, wave32, metadata SGPR/VGPR
  counts 106/1024. Metadata `max_flat_workgroup_size=1024` is an upper bound,
  not the launch block size.
- CO SHA256: `eb1a574bcab24da7f38cb10268779246387fb70610b74bf6f9e46fd00177bb78`.

## H16 ABI

The first 96 bytes have exactly the same pointer order and meaning as
`MlaV41FusedArgs` in the H64 launcher. Bytes 96..119 also have identical
field types and offsets; H16 fixes the split-related values to S1.
The only added fields are `query_count` and `persistent_groups`.
The kernel internally swaps all four CSA/SWA pointer pairs to compute SWA
first. **The host must not swap either data/scales or CSR pointers.**

| Offset (bytes) | Field | Host value |
| --- | --- | --- |
| 0 / 8 | q_nope / q_scale | Q data / E8M0 scales |
| 16 / 24 | unified_kv_nope / unified_kv_scale | CSA FP4 data / E4M3 scales |
| 32 / 40 | kv_nope / kv_scale | SWA FP8 data / E8M0 scales |
| 48 / 56 | attn_sink / out | FP32 sink / BF16 output |
| 64 / 72 | kv_indptr_prefix / kv_indices_prefix | CSA CSR |
| 80 / 88 | kv_indptr_extend / kv_indices_extend | SWA CSR |
| 96 / 100 | softmax_scale f32 / passes u32 | `1/sqrt(512)` / `1` |
| 104 / 112 | partial_r / partial_lse | null / null |
| 120 / 124 | query_count u32 / persistent_groups u32 | `N` / `256` |

## H16 Provenance And Build

Supplied by Chen Ting on 2026-11-20: `source 1.sp3` at 05:07 UTC and its
sp3cvt-generated `.s` at 05:10 UTC. Source package:
`poc_kl_merg/scripts/mi400/mono/7j_ds41_mono_sync/kernels/A06_ps/`.
The `.s` beside the code object is copied byte-for-byte from that package.

- Assembly SHA256: `68e9453672de984e6de9b9a16e217867edcd723f0d623297d8496f205a8604d4`.
- SP3 SHA256: `17fe3822d19b5ddfce79e50384049ace37136c1ac6a4f22d02ded1bd985652fe`.
- SP3 MD5 excluding its first stamp line:
  `a679bc89bee9ce5bc31b82a0c5831098`, also embedded in `.sp3.notes`.
- Query-loop evidence in `source.sp3`: `get_dispatch_index` (line 834),
  `load_kernel_args` (2323), 64-bit indptr offsets (2337),
  `loop_setup_ps` (5526), entry/handoff/empty paths (5680..5757).
  The authoritative package README and `doc/UPGRADE_FP4.md` sections 1.2/6
  establish the source layouts and CSA-prefix/SWA-extend ABI.

Rebuild from this repository root with the integration toolchain:

```bash
LLVM=/home/weizhu15/tmp/llvm23_dl/LLVM-23.1.5-Linux-X64/bin
K=hsa/gfx1250/mla_v41/mla_v41_qh16_ps_grid256x_mono_v1
"$LLVM/llvm-mc" -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj "$K.s" -o /tmp/aiter-a8w4-h16.o
"$LLVM/ld.lld" -shared /tmp/aiter-a8w4-h16.o -o "$K.co"
"$LLVM/llvm-readelf" --notes --symbols "$K.co"
"$LLVM/llvm-objdump" --mcpu=gfx1250 --disassemble "$K.co"
sha256sum "$K.s" "$K.co"
```

Offline assembly/link and metadata checks passed with LLVM 23.1.5.
Disassembly confirms entry `ttmp9 -> s38`, count/group load at kernarg
offset 120, `s92 = s38 + s91`, and the single decoded `s_endpgm` at kernel
offset `0x3f48`. LLVM leaves 56 words as `.long`, including TDM encodings;
this is not a complete instruction-decoding audit. Host contract tests pass;
the HIP launcher build, GPU correctness (including N>256), and performance
remain unverified. See `docs/mla_v41_h16_handoff.md` for acceptance commands.
