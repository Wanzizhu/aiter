# PA_DECODE_BF16 (gfx1250) varlen crash — debug progress / handoff

## TL;DR
`pa_decode_bf16_asm` kernel GPU-faults at **batch=128 + `--varlen`**. batch=1/64 varlen pass.
**Key finding:** the fault is NOT in kv_indices/K/V (phys_blk), as first assumed. It is a
**store to `split_lse` whose LOW 32 address bits are correct (split_lse_base + 0x1200) but the
HIGH 32 bits are garbage** → `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` (access beyond legal VA).
Triggered only when a **thread-group (TG) processes ≥2 work items** (multi-work-per-TG), which
only happens once `#work_items > #TGs(=num_cu=256)`, i.e. batch≥128 with varlen splits.

---

## Environment / access
- Machine: `ssh 450-c06`  (heliosp-1b114-c06-2.mnb.dcgpu), GPU gfx1250 (AMD Radeon, dev 0x75c1)
- Container: **`zw_iter`** (running `sleep infinity`). Enter with:
  ```
  docker exec -it zw_iter bash
  ```
  If it's gone, recreate:
  ```
  docker run -d --name zw_iter --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined -v /home/carhuang/weizhu15/aiter:/workspace/aiter \
    -w /workspace/aiter --entrypoint /bin/bash \
    rocm/fw-bringup:gfx1250-atom-dev-vinayak-20260606 -c 'sleep infinity'
  ```
  NOTE: image's default entrypoint is broken (`/usr/bin/bash: cannot execute binary file`),
  always pass `--entrypoint /bin/bash`.
- Paths: host `/home/carhuang/weizhu15/aiter` == container `/workspace/aiter`

## Reproduce the crash
```
cd /workspace/aiter
python op_tests/test_pa_decode_bf16_asm.py -b 128 -kvh 1 -c 1024 -m 0 --scales 1.69 1.7329 1.2276 --varlen
```
- batch 1, 64 → pass;  batch 128, 256 → crash.
- mtp = 0/1/2/3 → all crash (NOT mtp-specific).

## Build sp3 -> .co
- Build dir: `/workspace/aiter/pa/poc_kl/zw/`
  - sp3 source: `PA_DECODE_D64_1TG_4W_PS.sp3`
  - `bash build.sh`  (stage1: sp3cvt embed -> .s ; stage2: amdclang++ -> .co)
  - build output: `pa/poc_kl/zw/pa_decode_bf16_d64_page256_gqa8.co`
- Kernel actually loads: `/workspace/aiter/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8.co`
- Helper I added: `/workspace/aiter/build_and_test.sh`  (modes: `build` / `test <args>` / `restore` / default=build+test). Backs up the original deployed .co to `*.co.bak`.

### ⚠ BUILD CAVEAT (important)
Rebuilds from the current sp3+`kernel_template.s` do NOT byte-match the originally deployed .co,
and one earlier rebuild **regressed** (crashed `-b 1 -c 7` which the deployed .co passes).
`kernel_template.s` was modified Jun 12 01:10 (after the deployed .co was built Jun 11).
**Before trusting any rebuild for debugging, verify baseline:**
- `-b 1 -c 7 --scales 1 1 1`           → must PASS (no coredump)
- `-b 1 -c 1024 --varlen --scales ...`  → must PASS
- `-b 128 -c 1024 --varlen --scales ...`→ must CRASH (the bug)
Original deployed .co md5 is saved in `*.co.bak` (`8d0955d5d08a8db0fcecfda8481d9db5`).
`build_and_test.sh restore` puts it back.

## Test instrumentation ALREADY applied to op_tests/test_pa_decode_bf16_asm.py
(via patch_addr.py; keep these)
1. `random.seed(0)` added right after `torch.manual_seed(0)` → varlen seq_lens deterministic.
2. `[ZW] ... seq_lens_kv=...` print after seq_lens generation.
3. `[ZWADDR]` block right before `run_pa_stage`: prints `data_ptr()` range of every tensor
   (Q,K,V,kv_indices,seq_lens,kv_indptr,qo_indptr,work_indptr,work_info,split_o,split_lse,sink,OUT),
   sorted by address. `run_pa_stage` was given an `out_buf` param so OUT's address is the real one.
   Use this to correlate the GPU fault address with a buffer.

---

## KEY FINDING (the crux) — fault is split_lse, high address word corrupted
Captured fault addr + tensor ranges in the SAME run (ASLR randomizes bases each run):

| run | split_lse base            | fault address        | fault_low − split_lse_low |
|-----|---------------------------|----------------------|---------------------------|
| 1   | 0x7fcdb3c94e00            | 0x732eb3c96000       | **+0x1200**               |
| A   | 0x7f2daa894e00            | 0x6f2eaa896000       | **+0x1200**               |
| B   | (aperture violation: "access beyond largest legal address")                  |

- LOW 32 bits of fault == `split_lse_base + 0x1200` **consistently** (not coincidence; base moves
  with ASLR, offset stays 0x1200).
- HIGH 32 bits are garbage (run1 0x7fcd→0x732e, runA 0x7f2d→0x6f2e; delta not constant).
- `0x1200 = 4608 = 144 * 32`.  split_lse row stride = q_head_num(8)*1*4 = 32 B.
- split_lse store-address formula in `store_partial_results`:
  `off = (((mtp_layer + partial_idx)*kv_nheads + kv_head)*GQA + gqa_id) * 4`
  with kv_nheads=1, kv_head=0, gqa_id=0, mtp_layer=0 → `partial_idx * 32`.
  ⇒ the faulting store is **partial_o_loc = 144**, via `buffer_store_dword ... s_PartialLSE_addr`.
- ⇒ The split path's split_lse store computes a correct low address but a **corrupted high word**.

## Triggering conditions (established with the ORIGINAL deployed .co)
- Needs varlen (uneven seq_lens → KV-split work items, partial_o_loc != -1).
- Needs a **TG with ≥2 work items**. PROVEN by surgical test: take the exact crashing
  `work_info`, change ONLY `work_indptr` so each TG gets 1 work ("spread", grid=#items) →
  **NO crash, correct result**. Original multi-work distribution → crash.
- batch 1/64: #work_items ≤ 256 TGs → ≤1 work/TG → pass.
  batch 128: ~281–288 work_items > 256 TGs → some TGs get 2 → crash.
- A single 2-work TG run in isolation (grid=1) can reproduce the crash for the right item pair.

## Ruled OUT
- NOT bad metadata: work_info is clean — `kv_end` min=2 (never 0), `0<=kv_start<kv_end<=kvip[b+1]<=kvlen`,
  0 anomalies (verified through the real test path across multiple seeds).
- NOT kv_indices content/read: zero-filling kv_indices → still crash; placing kv_indices in the
  middle of a 64 MB buffer (±32 MB mapped) → still crash; canary check on the passing "spread"
  run showed kv_indices/K/V unmodified and no output-buffer canary overflow.
- NOT a simple output overflow: oversizing split_o/split_lse 16× and OUT 16× → still crash
  (because it's a high-word ADDRESS corruption, not a size/end-overflow).
- NOT mtp-specific (0–3 all crash).
- User observation: "forcing phys_blk_idx=0 (removing the kv_indices `s_load`) makes it pass."
  Given the fault is actually split_lse (not kv_indices), this most likely works via code-shift/
  timing, not because phys_blk is the culprit. Treat as a clue, not the root cause.

---

## CURRENT HYPOTHESIS → next steps
The split partial-output store to **split_lse** uses a corrupted HIGH address word, only when a
TG runs a 2nd work item. Prime suspects:
- `s_PartialLSE_addr` (and maybe `s_PartialO_addr`) — the 4-SGPR buffer descriptors — are set
  ONCE by `init_buffer_default(...)` BEFORE the work loop and must survive across work-loop
  iterations. If something clobbers the descriptor's high dword during the 1st work item, the
  2nd work item's split store faults.
- OR the per-store address-high computation in `store_partial_results` goes wrong on re-entry.

### Concrete next actions
1. Audit sp3 for writes to `s_PartialLSE_addr` / `s_PartialO_addr` (and the SGPRs they alias)
   anywhere inside the work loop / store_partial_results / cross_wave_reduce. They should be
   write-once. Any scratch reuse of those SGPR indices = the bug.
2. Instrument the kernel: right before `buffer_store_dword ... s_PartialLSE_addr` in
   `store_partial_results`, write `s_PartialLSE_addr[0:1]` (and the final computed address) to a
   scratch debug buffer; run the minimal 2-work-TG repro and dump it to see the corrupted high word
   and trace which prior instruction set it. (Needs a build that reproduces baseline first — see
   BUILD CAVEAT.)
3. Cross-check `init_buffer_default` (sets descriptor [1]=base_hi&0xffff, [2],[3]) vs how the
   buffer_store consumes the descriptor; confirm base_hi survives to the store.

## sp3 line references
Two sp3 copies exist; they may differ slightly — trust the deployed one for line numbers:
- Deployed/build source: `/workspace/aiter/pa/poc_kl/zw/PA_DECODE_D64_1TG_4W_PS.sp3`
- Analysis copy (downloaded earlier): `PA_DECODE_D64_1TG_4W_PS_qiwan.sp3`

Approx locations (qiwan copy):
- `s_PartialO_addr` / `s_PartialLSE_addr` decl: ~570–571
- `init_buffer_default(s_PartialO_addr / s_PartialLSE_addr)`: ~4157–4158 (deployed; ~4131-4132 qiwan)
- `store_partial_results` (split path; writes split_lse via buffer_store, split_o): ~3751–3890
- `pa_finalize_output` + `R_store_to_global` (direct-O path → OUT): ~3739, ~3893
- `work_loop_start`: ~4169 ; work_info `s_load_dwordx8`: ~4176 ; per-TG range from work_indptr: ~4094
- `K_TDM_lookup_issue` / `_step` (kv_indices read — was suspected, now NOT the fault): ~1054 / ~1094
- ICACHE-GUARD / inst-prefetch NOP guard note: ~4470

## Helper scripts created (in /workspace/aiter, also local copies on the Windows box)
- `build_and_test.sh` — build sp3→co, install, test, restore.
- `patch_addr.py` — applies the [ZWADDR] address-dump instrumentation to the test (already applied).
- (debug_*.py harnesses used during investigation also present.)
