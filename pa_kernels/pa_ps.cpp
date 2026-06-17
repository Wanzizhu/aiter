// Copyright (c) 2026 Advanced Micro Devices, Inc.  All rights reserved.
//----------------------------------------------------------------------------
// Host code for PA_DECODE_D64_1TG_4W_PERSISTENT.sp3
// (Paged Attention Decode Kernel, persistent/split-KV variant)
//
// Adapted from fmha_decode_mxfp8.cpp.
//
// Key differences from FMHA reference:
//   * Paged KV layout with block_table indirection (one TG per batch element).
//   * HeadDim=64, PageSize=256, GQA=8, TileQ=32, FP8 KV cache.
//   * Q/K/V are FP8 inputs with per-tensor scalar scales
//     (query_scale, key_scale, value_scale) instead
//     of per-block scales.
//   * Single-pass (no kv_split): each TG handles one batch element entirely.
//   * Output is bfloat16.
//
// Kernel argument layout (must match s_load_dword offsets in the SP3 main):
//   0x000 R_addr             (uint64)        output base
//   0x010 Q_addr             (uint64)        Q base (FP8)
//   0x020 K_addr             (uint64)        K cache base (FP8)
//   0x030 V_addr             (uint64)        V cache base (FP8)
//   0x040 KVIndices_addr     (uint64)        flattened physical page ids
//   0x050 CtxLen_addr        (uint64)        context lengths
//   0x060 softmax_scale      (float)         attention scale, passed by value
//   0x070 QScale_addr        (uint64)        scalar tensor, per-tensor Q scale
//   0x080 KScale_addr        (uint64)        scalar tensor, per-tensor K scale
//   0x090 VScale_addr        (uint64)        scalar tensor, per-tensor V scale
//   0x0A0 kv_head_num        (uint32)
//   0x0B0 Q_mtp_stride      (uint32)        bytes per MTP layer in FP8 Q
//   0x0C0 K_blk_stride       (uint32)
//   0x0D0 K_head_stride      (uint32)
//   0x0E0 mtp                (uint32)
//   0x0F0 gqa_ratio          (uint32)
//   0x100 QOIndptr_addr      (uint64)        accepted but unused
//   0x110 KVIndptr_addr      (uint64)
//   0x120 WorkPtr_addr       (uint64)
//   0x130 WorkInfo_addr      (uint64)
//   0x140 SplitO_addr        (uint64)
//   0x150 SplitLSE_addr      (uint64)
//   0x160 Sink_addr          (uint64)        per-Q-head fp32 sink logits
//
// Grid: global_size = (available_tgs * 128, 1, 1).
//----------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <sys/time.h>
#include <glob.h>
#include <regex.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <limits>

class StandaloneTestArgs {
public:
    void SetOptions(const std::vector<std::string>& option_list) {
        for (const std::string& option : option_list) {
            size_t equal_pos = option.find('=');
            if (equal_pos != std::string::npos) {
                options_[option.substr(0, equal_pos)] = option.substr(equal_pos + 1);
            }
        }
    }

    template <typename T>
    void GetParam(const std::string& key, T& value) const {
        auto it = options_.find(key);
        if (it == options_.end()) {
            return;
        }
        std::stringstream ss(it->second);
        ss >> value;
    }

    void GetParam(const std::string& key, std::string& value) const {
        auto it = options_.find(key);
        if (it != options_.end()) {
            value = it->second;
        }
    }

private:
    std::map<std::string, std::string> options_;
};

class SysMgr {
public:
    static SysMgr* Instance() {
        static SysMgr instance;
        return &instance;
    }

    StandaloneTestArgs* GetTestArgs() {
        return &args_;
    }

private:
    StandaloneTestArgs args_;
};

#include "common.h"
#include "common_decode.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "common_pa.h"           // pa_shuffle / pa_ref_dev_A16W8 / pa_dumpbin / ...
#include "common_ps.h"
#include "common_pa_ps.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace MI_KL_PA_DECODE_PERSISTENT {
using namespace TACL_gemm_ns;

#define HIP_CALL(call)                                                   \
    do                                                                   \
    {                                                                    \
        hipError_t err = call;                                           \
        if (err != hipSuccess)                                           \
        {                                                                \
            printf("[hiperror](%d) fail to call %s: %s\n",              \
                   (int)err, #call, hipGetErrorString(err));             \
            exit(0);                                                     \
        }                                                                \
    } while (0)

static std::string hip_kernel_name_from_path(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    size_t begin = (slash == std::string::npos) ? 0 : slash + 1;
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || dot < begin) {
        dot = path.size();
    }
    return path.substr(begin, dot - begin);
}

// Kernel-side constants (must match SP3)
static constexpr unsigned int PA_HEAD_DIM    = 64;
static constexpr unsigned int PA_PAGE_SIZE   = 256;
static constexpr unsigned int PA_TILE_Q      = 32;
static constexpr unsigned int PA_GQA_RATIO   = 8;
static constexpr unsigned int PA_WAVES       = 4;
static constexpr unsigned int PA_THREADS_PER_TG = 128;   // 4 waves * 32 threads

// ----------------------------------------------------------------------------
// CPU reference & paged-storage layout choice
// ----------------------------------------------------------------------------
// common_pa.h provides:
//   * pa_shuffle()         - pack dense K/V into [num_blocks, kv_heads, head_dim/x, blk, x]
//   * pa_ref_dev_A16W8()   - paged-attention reference matching that shuffled layout
// Both ASSUME the vLLM-style fine block granularity (assert blk_size/v_split == 16
// for FP8). Our SP3 kernel uses PAGE_SIZE = 256 tokens per *physical page* (one
// 16KB LDS tile = one page) and a much simpler token-major HBM layout:
//
//   K[phys_page, kv_head, token, dim]   (dim is innermost / fastest)
//   V[phys_page, kv_head, token, dim]   (same shape as K)
//
// Because of this granularity mismatch, the GPU input uses the simple SP3 page
// layout below, while the CPU reference builds a separate vLLM-style paged cache
// and validates through pa_ref_dev_A16W8. Mathematically the results are
// identical because block_table only changes physical storage, not values.
//
// We still pull in common_pa.h for helpers like pa_dumpbin used for debugging.
// ----------------------------------------------------------------------------

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static unsigned int ceil_div_u(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// IT = input KV cache element type (uint8 for FP8)
// QT = query element type            (uint8 for FP8)
// OT = output element type           (uint16 for bf16)
template<typename IT, typename QT, typename OT>
int do_compute(const common_args& cmn_args, DATA_TYPE in_type, DATA_TYPE out_type) {
    //-----Parameter initialization starts-------//
    int lds_size      = cmn_args.lds_size;
    int init_pattern  = cmn_args.init_pattern;
    int PerfCNT       = cmn_args.PerfCNT;
    int pte_preld     = cmn_args.pte_preld;
    int dump_result   = cmn_args.dump_result;
    int code_prefetch = cmn_args.code_prefetch;
    int check_mode    = cmn_args.check_mode;
    unsigned int wm   = 0;
    unsigned int total_loop = 1;
    SysMgr::Instance()->GetTestArgs()->GetParam("wm", wm);
    SysMgr::Instance()->GetTestArgs()->GetParam("total_loop", total_loop);
    assert(total_loop > 0 && "total_loop must be greater than 0");
    printf("Debug: Get Param wm=%u, total_loop=%u\n", wm, total_loop);

    // ---- User-controllable params ----
    unsigned int batch = 1;
    SysMgr::Instance()->GetTestArgs()->GetParam("batch", batch);
    printf("Debug: Get Param batch=%d\n", batch);

    unsigned int kv_head_num = 1;
    SysMgr::Instance()->GetTestArgs()->GetParam("kv_head_num", kv_head_num);
    printf("Debug: Get Param kv_head_num=%d\n", kv_head_num);

    unsigned int gqa = PA_GQA_RATIO;
    SysMgr::Instance()->GetTestArgs()->GetParam("gqa", gqa);
    printf("Debug: Get Param gqa=%d (kernel hard-coded GQA=%d)\n", gqa, PA_GQA_RATIO);
    assert(gqa == PA_GQA_RATIO && "PA_DECODE_D64 kernel requires GQA=8");

    unsigned int kv_seq_len = 256;
    SysMgr::Instance()->GetTestArgs()->GetParam("kv_seq_len", kv_seq_len);
    printf("Debug: Get Param kv_seq_len=%d\n", kv_seq_len);

    unsigned int qk_head_dim = PA_HEAD_DIM;
    SysMgr::Instance()->GetTestArgs()->GetParam("head_dim", qk_head_dim);
    printf("Debug: Get Param head_dim=%d (kernel hard-coded HeadDim=%d)\n",
           qk_head_dim, PA_HEAD_DIM);
    assert(qk_head_dim == PA_HEAD_DIM && "PA_DECODE_D64 kernel requires HeadDim=64");

    unsigned int v_head_dim = PA_HEAD_DIM;
    SysMgr::Instance()->GetTestArgs()->GetParam("v_head_dim", v_head_dim);
    assert(v_head_dim == PA_HEAD_DIM);

    unsigned int page_size = PA_PAGE_SIZE;
    SysMgr::Instance()->GetTestArgs()->GetParam("page_size", page_size);
    printf("Debug: Get Param page_size=%d (kernel hard-coded PageSize=%d)\n",
           page_size, PA_PAGE_SIZE);
    assert(page_size == PA_PAGE_SIZE);

    // Standard paged-attention reference block size. This is independent from
    // the kernel's custom 256-token physical page layout; it is used only by
    // pa_ref_dev_A16W8 validation.
    unsigned int block_size = PA_PAGE_SIZE;
    SysMgr::Instance()->GetTestArgs()->GetParam("block_size", block_size);
    printf("Debug: Get Param block_size=%d (CPU pa_ref_dev_A16W8 reference)\n", block_size);
    assert(block_size > 0 && (block_size % 16) == 0);

    // Total reserved physical pages in the cache pool.
    // Default: enough for batch * ceil(kv_seq_len / PAGE_SIZE) (no fragmentation).
    unsigned int num_phys_pages = batch * ceil_div(kv_seq_len, PA_PAGE_SIZE);
    SysMgr::Instance()->GetTestArgs()->GetParam("num_phys_pages", num_phys_pages);
    printf("Debug: Get Param num_phys_pages=%d\n", num_phys_pages);

    // max_blocks_per_seq: width of block_table.
    // Default: ceil(kv_seq_len / PAGE_SIZE).
    unsigned int max_blocks_per_seq = ceil_div(kv_seq_len, PA_PAGE_SIZE);
    SysMgr::Instance()->GetTestArgs()->GetParam("max_blocks_per_seq", max_blocks_per_seq);
    printf("Debug: Get Param max_blocks_per_seq=%d\n", max_blocks_per_seq);
    assert(max_blocks_per_seq >= ceil_div_u(kv_seq_len, PA_PAGE_SIZE));

    cl_float scalar = (float)(1.0 / (double)sqrt((double)qk_head_dim));
    SysMgr::Instance()->GetTestArgs()->GetParam("scalar", scalar);
    printf("Debug: Get Param scalar=%f\n", scalar);

    // Per-tensor dequant scales (FP8 values are interpreted as
    // fp_value = decoded_fp8 * scale). key_scale carries the attention scalar
    // by default, matching the previous kernel contract.
    cl_float key_scale = scalar;        // fold attention scalar into key_scale
    cl_float query_scale = 1.0f;
    cl_float value_scale = 1.0f;
    SysMgr::Instance()->GetTestArgs()->GetParam("key_scale", key_scale);
    SysMgr::Instance()->GetTestArgs()->GetParam("query_scale", query_scale);
    SysMgr::Instance()->GetTestArgs()->GetParam("value_scale", value_scale);
    printf("Debug: Get Param query_scale=%f key_scale=%f value_scale=%f\n",
           query_scale, key_scale, value_scale);

    // Attention softmax scale, passed by value (kernarg 0x160). The kernel folds
    // query_scale * key_scale * softmax_scale * log2(e) into scl_log2e. key_scale
    // already carries `scalar` (1/sqrt(d)) above, so the default 1.0 leaves the
    // existing numerics unchanged; set softmax_scale=N to apply an extra factor.
    cl_float softmax_scale = 1.0f;
    SysMgr::Instance()->GetTestArgs()->GetParam("softmax_scale", softmax_scale);
    printf("Debug: Get Param softmax_scale=%f\n", softmax_scale);

    unsigned int mtp = 0;
    SysMgr::Instance()->GetTestArgs()->GetParam("mtp", mtp);
    printf("Debug: Get Param mtp=%d\n", mtp);
    assert(mtp < (PA_TILE_Q / PA_GQA_RATIO) &&
           "PA_DECODE_D64 supports mtp in [0, 3] (up to 4 MTP layers)");

    unsigned int vgpr_cnt = 1024;
    SysMgr::Instance()->GetTestArgs()->GetParam("vgpr_cnt", vgpr_cnt);
    printf("Debug: Get Param vgpr_cnt =%d\n", vgpr_cnt);

    unsigned int dbg_trace = 0;
    SysMgr::Instance()->GetTestArgs()->GetParam("dbg_trace", dbg_trace);

    unsigned int persistent = 1;
    SysMgr::Instance()->GetTestArgs()->GetParam("persistent", persistent);
    printf("Debug: Get Param persistent=%d\n", persistent);
    assert(persistent == 1 && "This host copy targets the persistent kernel path");

    unsigned int available_tgs = 3;
    SysMgr::Instance()->GetTestArgs()->GetParam("available_tgs", available_tgs);
    printf("Debug: Get Param available_tgs=%d\n", available_tgs);
    assert(available_tgs > 0);

    unsigned int tile_kv = PA_PAGE_SIZE;
    SysMgr::Instance()->GetTestArgs()->GetParam("tile_kv", tile_kv);
    printf("Debug: Get Param tile_kv=%d (persistent split granularity)\n", tile_kv);
    assert(tile_kv == PA_PAGE_SIZE && "D64 persistent kernel currently expects tile_kv == page_size");

    unsigned int max_split_kv = 80;
    SysMgr::Instance()->GetTestArgs()->GetParam("max_split_kv", max_split_kv);
    printf("Debug: Get Param max_split_kv=%d\n", max_split_kv);

    unsigned int is_causal = 0;
    SysMgr::Instance()->GetTestArgs()->GetParam("is_causal", is_causal);
    printf("Debug: Get Param is_causal=%d\n", is_causal);

    unsigned int seed = 0;
    SysMgr::Instance()->GetTestArgs()->GetParam("seed", seed);

    // i_perm = 1 (bhsd / [batch, head, seq, dim]) is the default; the kernel
    // strides are general, but the reference test infrastructure uses these
    // two layouts.
    unsigned int i_perm = 1;
    SysMgr::Instance()->GetTestArgs()->GetParam("i_perm", i_perm);
    printf("Debug: Get Param i_perm =%d (1=bhsd, 0=bshd)\n", i_perm);

    // Match PA.cpp naming: qlen_with_mtp is the number of MTP layers
    // materialized in Q/O. The flattened Q length per KV head is
    // qlen_with_mtp * gqa, with layout [batch, mtp, kv_head, gqa, dim].
    unsigned int qlen_with_mtp = mtp + 1;
    unsigned int q_seq_len_with_mtp = qlen_with_mtp * gqa;
    unsigned int q_head_num_per_mtp = kv_head_num * gqa;
    unsigned int tile_q = ceil_div_u(q_seq_len_with_mtp, 16) * 16;
    SysMgr::Instance()->GetTestArgs()->GetParam("tile_q", tile_q);
    printf("Debug: Get Param tile_q=%d\n", tile_q);

    //----- Attention-sink parameters (GPT-OSS style) -----
    // sink_buf[h] (one fp32 per Q head) is added as a virtual "sink token" in
    // softmax denominator. Values live in the PRE-SCALE raw logit domain --
    // the kernel multiplies by scl_log2e (= query_scale * key_scale * log2e)
    // before exp2. To feed AITER/FlashInfer post-scale sink values, multiply
    // by sqrt(qk_head_dim) before passing to sink_base/sink_span.
    //
    // use_sink=0 disables the merge in-kernel; SinkBuffer is still allocated
    // and filled with -INFINITY so the kernel's exp2(sink-max) collapses to 0
    // even if a future build accidentally pulls the merge in.
    unsigned int use_sink = 0;
    SysMgr::Instance()->GetTestArgs()->GetParam("sink", use_sink);
    printf("Debug: Get Param sink=%u\n", use_sink);
    float sink_base = 0.0f;
    SysMgr::Instance()->GetTestArgs()->GetParam("sink_base", sink_base);
    printf("Debug: Get Param sink_base=%f\n", sink_base);
    float sink_span = 5.0f;
    SysMgr::Instance()->GetTestArgs()->GetParam("sink_span", sink_span);
    printf("Debug: Get Param sink_span=%f\n", sink_span);

    //-----HIP module configuration-----//
    std::string hsaco_path = cmn_args.shaderPath + cmn_args.cs0;
    SysMgr::Instance()->GetTestArgs()->GetParam("hsaco", hsaco_path);
    SysMgr::Instance()->GetTestArgs()->GetParam("hip_module", hsaco_path);

    std::string kernel_name = hip_kernel_name_from_path(cmn_args.cs0);
    SysMgr::Instance()->GetTestArgs()->GetParam("kernel_name", kernel_name);
    SysMgr::Instance()->GetTestArgs()->GetParam("hsa_kernel", kernel_name);

    unsigned int hip_device = 0;
    SysMgr::Instance()->GetTestArgs()->GetParam("hip_device", hip_device);
    printf("Debug: HIP module=%s kernel=%s device=%u\n",
           hsaco_path.c_str(), kernel_name.c_str(), hip_device);

    bool ran_gpu = false;

    //----- Grid setup -----
    // 1 TG per (batch, kv_head). Each TG = 4 waves = 128 threads.
    int blockSizeX = PA_THREADS_PER_TG;
    int blockSizeY = 1;
    int blockSizeZ = 1;

    //----- Host buffer sizing -----
    // Q (FP8):      [batch, mtp, kv_head_num, gqa, qk_head_dim]
    // K (FP8):      [num_phys_pages, kv_head_num, qk_head_dim/16, PAGE_SIZE, 16]  (paged, pre-transposed)
    // V (FP8):      [num_phys_pages, kv_head_num, PAGE_SIZE/16, v_head_dim, 16]   (paged)
    // R (bf16):     [batch, mtp, kv_head_num, gqa, v_head_dim]
    // block_table:  [batch, max_blocks_per_seq] int32
    int sz_Q  = batch * kv_head_num * q_seq_len_with_mtp * qk_head_dim;
    int sz_R  = batch * kv_head_num * q_seq_len_with_mtp * v_head_dim;
    int sz_K  = num_phys_pages * kv_head_num * qk_head_dim * PA_PAGE_SIZE;
    int sz_V  = num_phys_pages * kv_head_num * v_head_dim  * PA_PAGE_SIZE;
    int sz_BT = batch * max_blocks_per_seq;

    QT*     Q_buf  = new QT[sz_Q];
    unsigned short* Q_ref = new unsigned short[sz_Q];
    IT*     K_buf  = new IT[sz_K];
    IT*     V_buf  = new IT[sz_V];
    OT*     cpu_R  = new OT[sz_R];
    OT*     gpu_R  = new OT[sz_R];
    uint32* BT_buf = new uint32[sz_BT];   // common_pa.h convention: uint32 block_tables

    // Sink buffer: float[q_head_num_per_mtp] indexed by global q_head_idx.
    // When use_sink=0, fill with -INFINITY so exp2(sink - max) == 0 in-kernel
    // and the merge is a numerical no-op (defensive against accidental enable).
    float* sink_buf = new float[q_head_num_per_mtp];
    if (use_sink) {
        for (unsigned int h = 0; h < q_head_num_per_mtp; ++h) {
            float rand01 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            sink_buf[h] = sink_base + rand01 * sink_span;
        }
    } else {
        for (unsigned int h = 0; h < q_head_num_per_mtp; ++h) {
            sink_buf[h] = -std::numeric_limits<float>::infinity();
        }
    }

    memset(gpu_R, 0xbe, sz_R * sizeof(OT));
    memset(K_buf, 0,    sz_K * sizeof(IT));
    memset(V_buf, 0,    sz_V * sizeof(IT));

    // ---- Init Q (raw FP8, host layout = [batch, mtp, kv_head, gqa, dim]) ----
    // The kernel (PA_DECODE sp3 Q_compute_addr) and the persistent CPU ref
    // (paged_attention_with_mtp_ps_ref) both interpret Q as mtp-major:
    //   [batch, mtp_layers, kv_head, gqa, head_dim].
    // fmha_batch_init writes [B, H, S, D] with offset b*H*S*D + h*S*D + s*D + d.
    // Calling it as (B=batch*qlen_with_mtp, H=kv_head_num, S=gqa, D=head_dim)
    // produces exactly that mtp-major layout. Passing (batch, kv_head_num,
    // qlen_with_mtp*gqa, dim) would give [batch, kv_head, mtp, gqa, dim]
    // (kv-head-major), which silently disagreed with the kernel/ref for any
    // (kv_head_num > 1 AND mtp > 0) configuration.
    srand(++seed);
    //init_pattern = 0;
    fmha_batch_init(Q_buf, batch * qlen_with_mtp, kv_head_num, gqa, qk_head_dim, FP8, init_pattern,
                    FP8_FMT, /*f8_bias=*/false, 0, 0.0f);

    // CPU references still consume BF16 Q. Keep the GPU input untouched in its
    // original FP8 layout and build a dequantized view only for validation.
    {
        int n_Q = sz_Q;
        int n_Q_fix = 0;
        for (int i = 0; i < n_Q; ++i) {
            if (Q_buf[i] == 0x7F || Q_buf[i] == 0xFF) {
                Q_buf[i] = 0x00;
                ++n_Q_fix;
            }
            float q = IntMapToFloat(f8_to_f32(static_cast<uint8>(Q_buf[i]), FP8_FMT, false));
            q *= query_scale;
            Q_ref[i] = static_cast<unsigned short>(FP32toBFP16(FloatMapToInt(q)));
        }
        printf("Debug: OCP E4M3 NaN sanitize: Q NaN->0 = %d/%d\n", n_Q_fix, n_Q);
    }

    // ---- Build block_table: simple identity allocator (page i of batch b
    // lives in physical page b * num_pages_per_seq + i).
    int num_logical_pages_per_seq = ceil_div(kv_seq_len, PA_PAGE_SIZE);
    for (unsigned int b = 0; b < batch; ++b) {
        for (unsigned int p = 0; p < max_blocks_per_seq; ++p) {
            uint32 phys = (p < (unsigned)num_logical_pages_per_seq)
                          ? (uint32)(b * num_logical_pages_per_seq + p)
                          : 0u;     // pad slots: harmless, kernel clamps via s_min_i32
            BT_buf[b * max_blocks_per_seq + p] = phys;
        }
    }

    // ---- Persistent metadata (PA_PS.cpp-compatible interface) ----
    std::vector<uint32> context_lens(batch, kv_seq_len);
    std::vector<uint32> qo_indptr(batch + 1, 0);
    std::vector<uint32> kv_indptr(batch + 1, 0);
    for (unsigned int b = 0; b < batch; ++b) {
        qo_indptr[b + 1] = qo_indptr[b] + qlen_with_mtp;
        kv_indptr[b + 1] = kv_indptr[b] + (kv_seq_len + PA_PAGE_SIZE - 1) / PA_PAGE_SIZE;
    }

    std::vector<uint32> kv_indices(kv_indptr.back(), 0);
    unsigned int global_used_pages = 0;
    for (unsigned int b = 0; b < batch; ++b) {
        unsigned int local_used_pages = (kv_seq_len + PA_PAGE_SIZE - 1) / PA_PAGE_SIZE;
        for (unsigned int p = 0; p < local_used_pages; ++p) {
            kv_indices[global_used_pages++] = BT_buf[b * max_blocks_per_seq + p];
        }
    }

    ps::META_DATA metadata(
        available_tgs,
        batch * ceil_div_u(qlen_with_mtp, qlen_with_mtp) * kv_head_num * gqa * available_tgs);
    std::vector<uint32> reduce_indptr;
    std::vector<std::vector<uint32>> reduce_final_map;
    std::vector<uint32> reduce_partial_map;
    ps::generate_metadata(qo_indptr,
                          kv_indptr,
                          context_lens,
                          gqa,
                          kv_head_num,
                          gqa,
                          qlen_with_mtp,
                          PA_PAGE_SIZE,
                          PA_PAGE_SIZE,
                          is_causal != 0,
                          metadata,
                          reduce_indptr,
                          reduce_final_map,
                          reduce_partial_map,
                          available_tgs,
                          dbg_trace);
    if (dbg_trace) {
        print_metadata(metadata);
        print_reduce_info(reduce_indptr, reduce_final_map, reduce_partial_map);
    }

    unsigned int valid_split_rows = 0;
    for (const auto& work : metadata.work_info) {
        if (work.partial_o_loc != static_cast<uint32>(-1)) {
            unsigned int row_end = work.partial_o_loc + (work.qo_end - work.qo_start);
            if (row_end > valid_split_rows) {
                valid_split_rows = row_end;
            }
        }
    }
    printf("Debug: valid_split_rows=%u\n", valid_split_rows);

    int sz_SplitO = batch * tile_q * max_split_kv * q_head_num_per_mtp * v_head_dim;
    int sz_SplitLSE = batch * tile_q * max_split_kv * q_head_num_per_mtp;
    unsigned int split_row_capacity = batch * tile_q * max_split_kv;
    if (valid_split_rows > split_row_capacity) {
        printf("Error: valid_split_rows=%u exceeds SplitO/SplitLSE row capacity=%u. "
               "Increase max_split_kv.\n",
               valid_split_rows, split_row_capacity);
        return 1;
    }
    std::vector<float> gpu_SplitO(sz_SplitO, 0.0f);
    std::vector<float> gpu_SplitLSE(sz_SplitLSE, 0.0f);
    std::vector<float> gpu_LSE(batch * q_seq_len_with_mtp * kv_head_num, 0.0f);
    std::vector<OT> cpu_R_ps(sz_R, 0);
    std::vector<float> cpu_SplitO(sz_SplitO, 0.0f);
    std::vector<float> cpu_SplitLSE(sz_SplitLSE, 0.0f);
    std::vector<float> cpu_LSE(batch * q_seq_len_with_mtp * kv_head_num, 0.0f);

    // Q/K/V per-tensor scales are passed as scalar tensors and loaded by the
    // kernel. The host keeps Q in its original FP8 layout.
    std::vector<float> Q_tensor_scale(1, query_scale);
    std::vector<float> K_tensor_scale(1, key_scale);
    std::vector<float> V_tensor_scale(1, value_scale);

    // ---- Init K/V (FP8) into the physical pages referenced by block_table.
    // To keep CPU-vs-GPU comparable, we generate a logical [batch, head, seq, dim]
    // tensor first, then scatter it into paged storage according to block_table.
    //
    // FP8 format = OCP E4M3 (non-FNUZ, bias=7).
    //   - MI400 v_cvt_pk_fp8_f32 and v_wmma_*_f8f6f4 both encode/decode FP8
    //     bytes assuming bias=7 (verified empirically: an in-kernel quant of
    //     fp32 1.0 with scale=240 produces byte 0x77, which is 240 only under
    //     bias=7; under FNUZ bias=8 it would have to be 0x7F).
    //   - Therefore host init must use f8_bias=false (the default = OCP E4M3
    //     non-FNUZ) and the CPU reference must decode with f8_bias=false too,
    //     otherwise host and GPU disagree by 2x on every K/V element (bias=7
    //     reads bias=8-encoded 1.0 byte 0x40 as 2.0 and vice versa).
    IT* K_logical = new IT[batch * kv_head_num * kv_seq_len * qk_head_dim];
    IT* V_logical = new IT[batch * kv_head_num * kv_seq_len * v_head_dim];
    srand(++seed);
    fmha_batch_init(K_logical, batch, kv_head_num, kv_seq_len, qk_head_dim, FP8, init_pattern,
                    FP8_FMT, /*f8_bias=*/false, 0, 0.0f);
    srand(++seed);
    fmha_batch_init(V_logical, batch, kv_head_num, kv_seq_len, v_head_dim,  FP8, init_pattern,
                    FP8_FMT, /*f8_bias=*/false, 0, 0.0f);

    // ---- OCP E4M3 NaN sanitization ----
    // OCP E4M3 (non-FNUZ) NaN encoding = 0x7F (S=0, exp=1111, mant=111) and
    // 0xFF (S=1, ...). Any other byte is finite. The host quantizer can emit
    // 0x7F/0xFF from saturated values; replace with +/- max-finite (0x7E / 0xFE
    // = +/-448) -- but since we just want to avoid NaN propagation through
    // QxK -> softmax -> O on both CPU ref and GPU, we replace with 0x00.
    {
        int n_K = batch * kv_head_num * kv_seq_len * qk_head_dim;
        int n_V = batch * kv_head_num * kv_seq_len * v_head_dim;
        int n_K_fix = 0, n_V_fix = 0;
        for (int i = 0; i < n_K; ++i)
            if (K_logical[i] == 0x7F || K_logical[i] == 0xFF) { K_logical[i] = 0x00; ++n_K_fix; }
        for (int i = 0; i < n_V; ++i)
            if (V_logical[i] == 0x7F || V_logical[i] == 0xFF) { V_logical[i] = 0x00; ++n_V_fix; }
        printf("Debug: OCP E4M3 NaN sanitize: K NaN->0 = %d/%d, V NaN->0 = %d/%d\n",
               n_K_fix, n_K, n_V_fix, n_V);
    }

    // K/V physical page layout (matches SP3 TDM tile views; bytes-per-page
    // unchanged so K_blk_stride / K_head_stride numbers stay the same):
    //
    //   K[phys_page, kv_head, dim/16, token, dim%16]   (PRE-TRANSPOSED, dim-major)
    //     vLLM/AITER-style FP8 K cache: [num_blocks, kv_heads, head_dim/16,
    //     block_size, 16]. One page = 256 tokens * 64 dim = 16 KB (contiguous).
    //     The kernel TDM-copies the page unchanged into LDS and re-derives the
    //     WMMA-A operand from this transposed image (K_LDS_rd_addr_gen).
    //     Byte offset of K[b, h, t, d]:
    //       addr = phys   * (kv_head_num * PAGE_SIZE * head_dim)
    //            + h      * (PAGE_SIZE * head_dim)
    //            + (d/16) * (PAGE_SIZE * 16)
    //            + t      * 16
    //            + (d%16)
    //
    //   V[phys_page, kv_head, t / 16, dim, t % 16]               (16-token tiles)
    //     One page = 256 tokens * 64 dim = 16 KB.
    //     Outer tile = 16 tokens; inside the tile: dim-major with innermost
    //     contiguous = 16 tokens (matches SP3 [16, 64, 16] view).
    //     SP3 TDM: 16 rows * 1024 bytes (= 64 dim * 16 inner-token per row).
    //     Byte offset of V[b, h, t, d]:
    //       addr = phys * (kv_head_num * PAGE_SIZE * head_dim)
    //            + h    * (PAGE_SIZE * head_dim)
    //            + (t / 16) * (head_dim * 16)
    //            + d        * 16
    //            + (t % 16)
    auto k_phys_addr = [&](unsigned int b, unsigned int h, unsigned int t,
                           unsigned int d) -> int {
        int phys = BT_buf[b * max_blocks_per_seq + (t / PA_PAGE_SIZE)];
        int tok  = t % PA_PAGE_SIZE;
        int d_hi = (int)d / 16;
        int d_lo = (int)d % 16;
        return phys  * (int)(kv_head_num * qk_head_dim * PA_PAGE_SIZE)
             + (int)h * (int)(qk_head_dim * PA_PAGE_SIZE)
             + d_hi   * (int)(PA_PAGE_SIZE * 16)
             + tok    * 16
             + d_lo;
    };
    auto v_phys_addr = [&](unsigned int b, unsigned int h, unsigned int t,
                           unsigned int d) -> int {
        int phys     = BT_buf[b * max_blocks_per_seq + (t / PA_PAGE_SIZE)];
        int tok      = t % PA_PAGE_SIZE;
        int tok_outer = tok / 16;
        int tok_inner = tok % 16;
        return phys      * (int)(kv_head_num * v_head_dim * PA_PAGE_SIZE)
             + (int)h    * (int)(v_head_dim * PA_PAGE_SIZE)
             + tok_outer * (int)(v_head_dim * 16)
             + (int)d    * 16
             + tok_inner;
    };

    for (unsigned int b = 0; b < batch; ++b) {
        for (unsigned int h = 0; h < kv_head_num; ++h) {
            for (unsigned int t = 0; t < kv_seq_len; ++t) {
                for (unsigned int d = 0; d < qk_head_dim; ++d) {
                    K_buf[k_phys_addr(b, h, t, d)] =
                        K_logical[((b * kv_head_num + h) * kv_seq_len + t) * qk_head_dim + d];
                }
                for (unsigned int d = 0; d < v_head_dim; ++d) {
                    V_buf[v_phys_addr(b, h, t, d)] =
                        V_logical[((b * kv_head_num + h) * kv_seq_len + t) * v_head_dim + d];
                }
            }
        }
    }

    if (dump_result) {
        fmha_dump_batch_inHex(Q_buf, "Q.hex", batch, kv_head_num, q_seq_len_with_mtp, qk_head_dim, FP8);
        fmha_dump_batch_inHex(Q_ref, "Q_ref_bf16.hex",
                              batch, kv_head_num, q_seq_len_with_mtp, qk_head_dim, BF16);
        fmha_dump_batch_inHex(K_logical, "K_logical.hex",
                              batch, kv_head_num, kv_seq_len, qk_head_dim, FP8);
        fmha_dump_batch_inHex(V_logical, "V_logical.hex",
                              batch, kv_head_num, kv_seq_len, v_head_dim,  FP8);
        // Raw paged binaries (consumable by debug-trace replay tools)
        pa_dumpbin(BT_buf, sz_BT * sizeof(uint32), "block_table.bin");
        pa_dumpbin(K_buf,  sz_K  * sizeof(IT),     "K_paged.bin");
        pa_dumpbin(V_buf,  sz_V  * sizeof(IT),     "V_paged.bin");
    }

    //----- CPU reference -----
    // Standard paged-attention CPU reference check (mirrors PA.cpp flow):
    // build a reference-only vLLM-style paged KV cache and validate with
    // pa_ref_dev_A16W8(block_size, block_tables, context_lens).
    printf("Calculating CPU result ~~~~~\n");
    bool has_cpu_ref = true;

    unsigned int kv_seq_len_ref_ajst = ceil_div_u(kv_seq_len, block_size) * block_size;
    unsigned int max_num_blocks_ref = kv_seq_len_ref_ajst / block_size;
    unsigned int v_split_ref = block_size / 16;   // FP8 pa_shuffle requires block_size / v_split == 16.
    unsigned int s_x_ref = 16;                    // FP8 K cache x dimension.
    // pa_ref_dev_A16W8 interprets q_seq_len/sub_Q as q_len (MTP layers).
    // Use one GQA group per reference Q tile so q_len == qlen_with_mtp and
    // q_seq_len is divisible for every supported mtp value (0..3).
    unsigned int sub_Q_ref = gqa;
    unsigned int sub_KV_ref = PA_PAGE_SIZE;
    if (sub_KV_ref > kv_seq_len_ref_ajst) {
        sub_KV_ref = kv_seq_len_ref_ajst;
    }

    int sz_KV_ref = batch * kv_head_num * kv_seq_len_ref_ajst * qk_head_dim;
    int sz_KV_scale_ref = batch * kv_head_num * kv_seq_len_ref_ajst;
    int sz_BT_ref = batch * max_num_blocks_ref;
    IT* K_ref_dense = new IT[sz_KV_ref];
    IT* V_ref_dense = new IT[sz_KV_ref];
    IT* K_ref_shuffle = new IT[sz_KV_ref];
    IT* V_ref_shuffle = new IT[sz_KV_ref];
    float* K_scale_ref = new float[sz_KV_scale_ref];
    float* V_scale_ref = new float[sz_KV_scale_ref];
    float* K_scale_ref_shuffle = new float[sz_KV_scale_ref];
    float* V_scale_ref_shuffle = new float[sz_KV_scale_ref];
    uint32* BT_ref = new uint32[sz_BT_ref];
    uint32* context_lens_ref = new uint32[batch];

#if defined(CHECK_RESULT)
    memset(K_ref_dense, 0, sz_KV_ref * sizeof(IT));
    memset(V_ref_dense, 0, sz_KV_ref * sizeof(IT));
    memset(K_ref_shuffle, 0, sz_KV_ref * sizeof(IT));
    memset(V_ref_shuffle, 0, sz_KV_ref * sizeof(IT));
    memset(K_scale_ref, 0, sz_KV_scale_ref * sizeof(float));
    memset(V_scale_ref, 0, sz_KV_scale_ref * sizeof(float));
    memset(K_scale_ref_shuffle, 0, sz_KV_scale_ref * sizeof(float));
    memset(V_scale_ref_shuffle, 0, sz_KV_scale_ref * sizeof(float));
    memset(BT_ref, 0, sz_BT_ref * sizeof(uint32));
    memset(context_lens_ref, 0, batch * sizeof(uint32));

    // pa_shuffle expects K as [batch, kv_head, seq, dim] and V as
    // [batch, kv_head, dim, seq]. Pad to block_size multiple for standard PA.
    for (unsigned int b = 0; b < batch; ++b) {
        for (unsigned int h = 0; h < kv_head_num; ++h) {
            for (unsigned int t = 0; t < kv_seq_len; ++t) {
                K_scale_ref[(b * kv_head_num + h) * kv_seq_len_ref_ajst + t] =
                    (query_scale * key_scale) / scalar;
                V_scale_ref[(b * kv_head_num + h) * kv_seq_len_ref_ajst + t] = value_scale;
                for (unsigned int d = 0; d < qk_head_dim; ++d) {
                    K_ref_dense[((b * kv_head_num + h) * kv_seq_len_ref_ajst + t) * qk_head_dim + d] =
                        K_logical[((b * kv_head_num + h) * kv_seq_len + t) * qk_head_dim + d];
                }
                for (unsigned int d = 0; d < v_head_dim; ++d) {
                    IT val = V_logical[((b * kv_head_num + h) * kv_seq_len + t) * v_head_dim + d];
                    V_ref_dense[(b * kv_head_num + h) * v_head_dim * kv_seq_len_ref_ajst
                              + d * kv_seq_len_ref_ajst
                              + t] = val;
                }
            }
        }
    }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
    srand(++seed);
    pa_shuffle(K_ref_dense, V_ref_dense, K_ref_shuffle, V_ref_shuffle, BT_ref, context_lens_ref,
               batch, kv_head_num, kv_seq_len_ref_ajst, qk_head_dim,
               s_x_ref, block_size, max_num_blocks_ref, v_split_ref);
    pa_shuffle_KVScale(K_scale_ref, V_scale_ref, K_scale_ref_shuffle, V_scale_ref_shuffle,
                       BT_ref, context_lens_ref,
                       batch, kv_head_num, kv_seq_len_ref_ajst, 1,
                       block_size, max_num_blocks_ref);

    // pa_ref_dev_A16W8 follows PA.cpp: q/o are [batch, mtp, kv_head, gqa, dim],
    // and q_seq_len is passed as qlen_with_mtp * gqa.
    unsigned int q_seq_len_ref = q_seq_len_with_mtp;

    pa_ref_dev_A16W8(reinterpret_cast<cl_half*>(Q_ref),
                     K_ref_shuffle,
                     V_ref_shuffle,
                     K_scale_ref_shuffle,
                     V_scale_ref_shuffle,
                     reinterpret_cast<cl_half*>(cpu_R),
                     batch,
                     kv_head_num,
                     q_seq_len_ref,
                     kv_seq_len,
                     qk_head_dim,
                     BT_ref,
                     context_lens_ref,
                     s_x_ref,
                     block_size,
                     max_num_blocks_ref,
                     sub_Q_ref,
                     sub_KV_ref,
                     dbg_trace,
                     FP8,
                     LAYOUT_NONE,
                     v_split_ref,
                     FP8_FMT,
                     false,
                     FP8_FMT,
                     false,
                     FP8_FMT,
                     false,
                     0,
                     0,
                     0,
                     0);

    // Persistent CPU stage reference for SplitO/SplitLSE checking. It uses the
    // standard shuffled reference cache plus the same work metadata as the GPU.
    std::vector<uint32> kv_indices_ref(kv_indptr.back(), 0);
    {
        unsigned int ref_used_pages = 0;
        for (unsigned int b = 0; b < batch; ++b) {
            unsigned int local_used_pages = (kv_seq_len + PA_PAGE_SIZE - 1) / PA_PAGE_SIZE;
            for (unsigned int p = 0; p < local_used_pages; ++p) {
                kv_indices_ref[ref_used_pages++] = BT_ref[b * max_num_blocks_ref + p];
            }
        }
    }
    PA_PARAMS_PS<IT> pa_params_ps_ref(batch,
                                      kv_head_num,
                                      qlen_with_mtp,
                                      gqa,
                                      s_x_ref,
                                      kv_seq_len,
                                      v_head_dim,
                                      block_size,
                                      max_num_blocks_ref,
                                      is_causal != 0);
    pa_persistent_fwd_ref<IT>(pa_params_ps_ref,
                              reinterpret_cast<cl_half*>(Q_buf),
                              K_ref_shuffle,
                              V_ref_shuffle,
                              K_scale_ref_shuffle,
                              V_scale_ref_shuffle,
                              qo_indptr.data(),
                              kv_indptr.data(),
                              kv_indices_ref.data(),
                              context_lens.data(),
                              reinterpret_cast<cl_half*>(cpu_R_ps.data()),
                              cpu_LSE.data(),
                              cpu_SplitO.data(),
                              cpu_SplitLSE.data(),
                              metadata,
                              reduce_indptr,
                              reduce_final_map,
                              reduce_partial_map,
                              tile_q,
                              tile_kv,
                              dbg_trace,
                              LAYOUT_NONE,
                              FP8_ARG(),
                              true,
                              1.0f);
    memcpy(cpu_R, cpu_R_ps.data(), sz_R * sizeof(OT));
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifndef DIS_KL
    //----- HIP module setup -----
    hipModule_t module = nullptr;
    hipFunction_t kernel_func = nullptr;
    HIP_CALL(hipSetDevice(static_cast<int>(hip_device)));
    HIP_CALL(hipModuleLoad(&module, hsaco_path.c_str()));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_name.c_str()));

    if (pte_preld || PerfCNT || code_prefetch) {
        printf("Debug: HIP path ignores pte_preld=%d PerfCNT=%d code_prefetch=%d\n",
               pte_preld, PerfCNT, code_prefetch);
    }

    //----- Device buffers -----
    QT* QBuffer = nullptr;
    IT* KBuffer = nullptr;
    IT* VBuffer = nullptr;
    OT* RBuffer = nullptr;
    uint32* KVIndicesBuffer = nullptr;
    uint32* CtxLensBuffer = nullptr;
    float* QScaleBuffer = nullptr;
    float* KScaleBuffer = nullptr;
    float* VScaleBuffer = nullptr;
    uint32* QOIndptrBuffer = nullptr;
    uint32* KVIndptrBuffer = nullptr;
    uint32* WorkPtrBuffer = nullptr;
    ps::WORK_INFO* WorkInfoBuffer = nullptr;
    float* SplitOBuffer = nullptr;
    float* SplitLSEBuffer = nullptr;
    float* SinkBuffer = nullptr;

    size_t size_Q = static_cast<size_t>(sz_Q) * sizeof(QT);
    size_t size_K = static_cast<size_t>(sz_K) * sizeof(IT);
    size_t size_V = static_cast<size_t>(sz_V) * sizeof(IT);
    size_t size_R = static_cast<size_t>(sz_R) * sizeof(OT);
    size_t size_KVIndices = kv_indices.size() * sizeof(uint32);
    size_t size_CtxLens = context_lens.size() * sizeof(uint32);
    size_t size_QScale = Q_tensor_scale.size() * sizeof(float);
    size_t size_KScale = K_tensor_scale.size() * sizeof(float);
    size_t size_VScale = V_tensor_scale.size() * sizeof(float);
    size_t size_QOIndptr = qo_indptr.size() * sizeof(uint32);
    size_t size_KVIndptr = kv_indptr.size() * sizeof(uint32);
    size_t size_WorkPtr = metadata.work_indptr.size() * sizeof(uint32);
    size_t size_WorkInfo = metadata.work_info.size() * sizeof(ps::WORK_INFO);
    size_t size_SplitO = static_cast<size_t>(sz_SplitO) * sizeof(float);
    size_t size_SplitLSE = static_cast<size_t>(sz_SplitLSE) * sizeof(float);
    size_t size_Sink = static_cast<size_t>(q_head_num_per_mtp) * sizeof(float);

    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&QBuffer), size_Q));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&KBuffer), size_K));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&VBuffer), size_V));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&RBuffer), size_R));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&KVIndicesBuffer), size_KVIndices ? size_KVIndices : 1));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&CtxLensBuffer), size_CtxLens ? size_CtxLens : 1));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&QScaleBuffer), size_QScale));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&KScaleBuffer), size_KScale));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&VScaleBuffer), size_VScale));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&QOIndptrBuffer), size_QOIndptr ? size_QOIndptr : 1));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&KVIndptrBuffer), size_KVIndptr ? size_KVIndptr : 1));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&WorkPtrBuffer), size_WorkPtr ? size_WorkPtr : 1));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&WorkInfoBuffer), size_WorkInfo ? size_WorkInfo : 1));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&SplitOBuffer), size_SplitO));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&SplitLSEBuffer), size_SplitLSE));
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&SinkBuffer), size_Sink));

    printf("QBuffer          : start-%p, 0x%zxKB\n", QBuffer, size_Q / 1024);
    printf("KBuffer          : start-%p, 0x%zxKB\n", KBuffer, size_K / 1024);
    printf("VBuffer          : start-%p, 0x%zxKB\n", VBuffer, size_V / 1024);
    printf("RBuffer          : start-%p, 0x%zxKB\n", RBuffer, size_R / 1024);
    printf("KVIndicesBuffer  : start-%p, 0x%zxKB\n", KVIndicesBuffer, size_KVIndices / 1024);
    printf("CtxLensBuffer    : start-%p, 0x%zxKB\n", CtxLensBuffer, size_CtxLens / 1024);
    printf("QScaleBuffer     : start-%p, 0x%zxKB\n", QScaleBuffer, size_QScale / 1024);
    printf("KScaleBuffer     : start-%p, 0x%zxKB\n", KScaleBuffer, size_KScale / 1024);
    printf("VScaleBuffer     : start-%p, 0x%zxKB\n", VScaleBuffer, size_VScale / 1024);
    printf("QOIndptrBuffer   : start-%p, 0x%zxKB\n", QOIndptrBuffer, size_QOIndptr / 1024);
    printf("KVIndptrBuffer   : start-%p, 0x%zxKB\n", KVIndptrBuffer, size_KVIndptr / 1024);
    printf("WorkPtrBuffer    : start-%p, 0x%zxKB\n", WorkPtrBuffer, size_WorkPtr / 1024);
    printf("WorkInfoBuffer   : start-%p, 0x%zxKB\n", WorkInfoBuffer, size_WorkInfo / 1024);
    printf("SplitOBuffer     : start-%p, 0x%zxKB\n", SplitOBuffer, size_SplitO / 1024);
    printf("SplitLSEBuffer   : start-%p, 0x%zxKB\n", SplitLSEBuffer, size_SplitLSE / 1024);
    printf("SinkBuffer       : start-%p, 0x%zxB\n",  SinkBuffer, size_Sink);

    HIP_CALL(hipMemcpy(QBuffer, Q_buf, size_Q, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(KBuffer, K_buf, size_K, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(VBuffer, V_buf, size_V, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(RBuffer, gpu_R, size_R, hipMemcpyHostToDevice));
    if (size_KVIndices) {
        HIP_CALL(hipMemcpy(KVIndicesBuffer, kv_indices.data(), size_KVIndices, hipMemcpyHostToDevice));
    }
    if (size_CtxLens) {
        HIP_CALL(hipMemcpy(CtxLensBuffer, context_lens.data(), size_CtxLens, hipMemcpyHostToDevice));
    }
    HIP_CALL(hipMemcpy(QScaleBuffer, Q_tensor_scale.data(), size_QScale, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(KScaleBuffer, K_tensor_scale.data(), size_KScale, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(VScaleBuffer, V_tensor_scale.data(), size_VScale, hipMemcpyHostToDevice));
    if (size_QOIndptr) {
        HIP_CALL(hipMemcpy(QOIndptrBuffer, qo_indptr.data(), size_QOIndptr, hipMemcpyHostToDevice));
    }
    if (size_KVIndptr) {
        HIP_CALL(hipMemcpy(KVIndptrBuffer, kv_indptr.data(), size_KVIndptr, hipMemcpyHostToDevice));
    }
    if (size_WorkPtr) {
        HIP_CALL(hipMemcpy(WorkPtrBuffer, metadata.work_indptr.data(), size_WorkPtr, hipMemcpyHostToDevice));
    }
    if (size_WorkInfo) {
        HIP_CALL(hipMemcpy(WorkInfoBuffer, metadata.work_info.data(), size_WorkInfo, hipMemcpyHostToDevice));
    }
    HIP_CALL(hipMemcpy(SplitOBuffer, gpu_SplitO.data(), size_SplitO, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(SplitLSEBuffer, gpu_SplitLSE.data(), size_SplitLSE, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(SinkBuffer, sink_buf, size_Sink, hipMemcpyHostToDevice));

    // PA_PS-style input strides. The SP3 derives BF16 output strides from this
    // FP8 Q stride because Q and O have the same logical layout but different
    // element sizes.
    int stride_Q = q_head_num_per_mtp * qk_head_dim * (int)sizeof(QT);
    int stride_KV_head = PA_PAGE_SIZE * qk_head_dim * (int)sizeof(IT);
    int stride_KV_blk = stride_KV_head * kv_head_num;

    struct p3 { unsigned int _p0, _p1, _p2; };
    struct p2 { unsigned int _p0, _p1; };

    // Each kernel argument occupies a 16-byte slot; offsets must match the SP3 s_loads.
    struct __attribute__((packed)) {
        void* ptr_R;              p2 _pad0;    // 0x000
        void* ptr_Q;              p2 _pad1;    // 0x010
        void* ptr_K;              p2 _pad2;    // 0x020
        void* ptr_V;              p2 _pad3;    // 0x030
        void* ptr_KVIndices;      p2 _pad4;    // 0x040
        void* ptr_CtxLen;         p2 _pad5;    // 0x050
        float softmax_scale;      p3 _pad6;    // 0x060 (by-value f32; SP3 s_load_dword 0x60)
        void* ptr_QScale;         p2 _pad7;    // 0x070
        void* ptr_KScale;         p2 _pad8;    // 0x080
        void* ptr_VScale;         p2 _pad9;    // 0x090
        unsigned int kv_head_num; p3 _pad10;   // 0x0A0
        unsigned int stride_Q;    p3 _pad11;   // 0x0B0
        unsigned int stride_K_blk;p3 _pad12;   // 0x0C0
        unsigned int stride_K_head;p3 _pad13;  // 0x0D0
        unsigned int mtp;         p3 _pad14;   // 0x0E0
        unsigned int gqa_ratio;   p3 _pad15;   // 0x0F0
        void* ptr_QOIndptr;       p2 _pad16;   // 0x100
        void* ptr_KVIndptr;       p2 _pad17;   // 0x110
        void* ptr_WorkPtr;        p2 _pad18;   // 0x120
        void* ptr_WorkInfo;       p2 _pad19;   // 0x130
        void* ptr_SplitO;         p2 _pad20;   // 0x140
        void* ptr_SplitLSE;       p2 _pad21;   // 0x150
        void* ptr_Sink;           p2 _pad22;   // 0x160 (matches SP3 s_load_dwordx2 s_SINK_buf)
    } args;
    static_assert(sizeof(void*) == 8, "HIP argument layout assumes 64-bit device pointers");
    static_assert(sizeof(args) == 0x170, "HIP argument block must match SP3 s_load offsets");
    memset(&args, 0, sizeof(args));

    args.ptr_R = (void*)RBuffer;
    args.ptr_Q = (void*)QBuffer;
    args.ptr_K = (void*)KBuffer;
    args.ptr_V = (void*)VBuffer;
    args.ptr_KVIndices = (void*)KVIndicesBuffer;
    args.ptr_CtxLen = (void*)CtxLensBuffer;
    args.ptr_QScale = (void*)QScaleBuffer;
    args.ptr_KScale = (void*)KScaleBuffer;
    args.ptr_VScale = (void*)VScaleBuffer;
    args.kv_head_num = kv_head_num;
    args.stride_Q = stride_Q;
    args.stride_K_blk = stride_KV_blk;
    args.stride_K_head = stride_KV_head;
    args.mtp = mtp;
    args.gqa_ratio = gqa;
    args.ptr_QOIndptr = (void*)QOIndptrBuffer;
    args.ptr_KVIndptr = (void*)KVIndptrBuffer;
    args.ptr_WorkPtr = (void*)WorkPtrBuffer;
    args.ptr_WorkInfo = (void*)WorkInfoBuffer;
    args.ptr_SplitO = (void*)SplitOBuffer;
    args.ptr_SplitLSE = (void*)SplitLSEBuffer;
    args.ptr_Sink = (void*)SinkBuffer;
    args.softmax_scale = softmax_scale;

    size_t arg_size = sizeof(args);
    printf("-------printing kernel args: -------\n");
    printf("stride_Q=%d, stride_KV_blk=%d, stride_KV_head=%d\n",
           args.stride_Q, args.stride_K_blk, args.stride_K_head);
    printf("kv_head_num=%u, mtp=%u, gqa=%u\n", args.kv_head_num, args.mtp, args.gqa_ratio);
    printf("softmax_scale=%f\n", args.softmax_scale);
    printf("argsize: %zu\n", arg_size);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
                      HIP_LAUNCH_PARAM_END};

    // PM4 lds_size is the COMPUTE_PGM_RSRC2 LDS_SIZE value (128 dword units).
    unsigned int lds_bytes = static_cast<unsigned int>(lds_size) * 512u;

    //----- Launch -----
    printf("Start Kernel ~~~~~ \n");
    printf("localX=%d, localY=%d, localZ=%d\n", blockSizeX, blockSizeY, blockSizeZ);
    int grid_size_x = available_tgs;
    int grid_size_y = 1;
    int grid_size_z = 1;
    printf("gridX=%d, gridY=%d, gridZ=%d, dynamic_lds=%u bytes\n",
           grid_size_x, grid_size_y, grid_size_z, lds_bytes);

    auto launch_kernel = [&]() {
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       grid_size_x,
                                       grid_size_y,
                                       grid_size_z,
                                       blockSizeX,
                                       blockSizeY,
                                       blockSizeZ,
                                       0,
                                       nullptr,
                                       nullptr,
                                       (void**)&config));
    };

    //----- Warm up -----
    for (unsigned int i = 0; i < wm; ++i) {
        HIP_CALL(hipMemcpy(RBuffer, gpu_R, size_R, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(SplitOBuffer, gpu_SplitO.data(), size_SplitO, hipMemcpyHostToDevice));
        HIP_CALL(hipMemcpy(SplitLSEBuffer, gpu_SplitLSE.data(), size_SplitLSE, hipMemcpyHostToDevice));
        launch_kernel();
        HIP_CALL(hipDeviceSynchronize());
    }
    if (wm > 0) {
        printf("warm up %u times done\n", wm);
    }

    //----- Timed run -----
    HIP_CALL(hipMemcpy(RBuffer, gpu_R, size_R, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(SplitOBuffer, gpu_SplitO.data(), size_SplitO, hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(SplitLSEBuffer, gpu_SplitLSE.data(), size_SplitLSE, hipMemcpyHostToDevice));
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipEventCreate(&evt_00));
    HIP_CALL(hipEventCreate(&evt_11));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipEventRecord(evt_00, nullptr));
    for (unsigned int i = 0; i < total_loop; ++i) {
        launch_kernel();
    }
    float elapsed_ms = 0.0f;
    HIP_CALL(hipEventRecord(evt_11, nullptr));
    HIP_CALL(hipEventSynchronize(evt_11));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipEventElapsedTime(&elapsed_ms, evt_00, evt_11));
    HIP_CALL(hipEventDestroy(evt_00));
    HIP_CALL(hipEventDestroy(evt_11));

    float time_per_loop = elapsed_ms / total_loop;

    // Count only the two tensor-core matmuls in attention: QK and P*V.
    // Causal mode follows the same row-wise mask limit used by common_pa_ps.h.
    double attended_token_pairs = 0.0;
    for (unsigned int b = 0; b < batch; ++b) {
        int ctx_len = static_cast<int>(context_lens[b]);
        for (unsigned int mtp_idx = 0; mtp_idx < qlen_with_mtp; ++mtp_idx) {
            int valid_kv_len = ctx_len;
            if (is_causal) {
                valid_kv_len = ctx_len - static_cast<int>(mtp) + 1 + static_cast<int>(mtp_idx);
                if (valid_kv_len < 0) {
                    valid_kv_len = 0;
                } else if (valid_kv_len > ctx_len) {
                    valid_kv_len = ctx_len;
                }
            }
            attended_token_pairs += static_cast<double>(valid_kv_len) * kv_head_num * gqa;
        }
    }

    double flops = attended_token_pairs * 2.0 *
                   (static_cast<double>(qk_head_dim) + static_cast<double>(v_head_dim));
    double tflops = flops / time_per_loop / 1e9;

    uint64_t kv_work_blocks = 0;
    double q_read_bytes = 0.0;
    for (int64_t i = 0; i < metadata.num_work; ++i) {
        const auto& work = metadata.work_info[i];
        kv_work_blocks += static_cast<uint64_t>(work.kv_end - work.kv_start);

        unsigned int q_head_start = work.q_head_range & 0xFFFF;
        unsigned int q_head_end = (work.q_head_range >> 16) & 0xFFFF;
        q_read_bytes += static_cast<double>(work.qo_end - work.qo_start) *
                        static_cast<double>(q_head_end - q_head_start) *
                        qk_head_dim * sizeof(QT);
    }

    double kv_cache_bytes = static_cast<double>(kv_work_blocks) * PA_PAGE_SIZE *
                            (static_cast<double>(qk_head_dim) * sizeof(IT) +
                             static_cast<double>(v_head_dim) * sizeof(IT));
    double scale_bytes = static_cast<double>(metadata.num_work) * 3.0 * sizeof(float);
    double output_bytes = static_cast<double>(size_R) +
                          static_cast<double>(valid_split_rows) * kv_head_num * gqa *
                              (static_cast<double>(v_head_dim) + 1.0) * sizeof(float);
    double bandwidth_bytes = q_read_bytes + kv_cache_bytes + scale_bytes + output_bytes;
    double bandwidthT = bandwidth_bytes / time_per_loop / 1e9;

    printf("batch:%u,kv_head_num:%u,kv_seq_len:%u,causal:%u, time: %.3f ms, "
           "tflops:%.3f, bandwidth:%.3fT, total_loop:%u, total: %.3f ms\n",
           batch, kv_head_num, kv_seq_len, is_causal, time_per_loop,
           tflops, bandwidthT, total_loop, elapsed_ms);
    printf("Perf detail: attended_token_pairs=%.0f, kv_work_blocks=%llu, bandwidth_bytes=%.0f\n",
           attended_token_pairs,
           static_cast<unsigned long long>(kv_work_blocks),
           bandwidth_bytes);
    printf("Kernel Finished ~~~~~~ \n");

    //----- Read back -----
    HIP_CALL(hipMemcpy(gpu_R, RBuffer, size_R, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(gpu_SplitO.data(), SplitOBuffer, size_SplitO, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(gpu_SplitLSE.data(), SplitLSEBuffer, size_SplitLSE, hipMemcpyDeviceToHost));
    ran_gpu = true;

    HIP_CALL(hipFree(QBuffer));
    HIP_CALL(hipFree(KBuffer));
    HIP_CALL(hipFree(VBuffer));
    HIP_CALL(hipFree(RBuffer));
    HIP_CALL(hipFree(KVIndicesBuffer));
    HIP_CALL(hipFree(CtxLensBuffer));
    HIP_CALL(hipFree(QScaleBuffer));
    HIP_CALL(hipFree(KScaleBuffer));
    HIP_CALL(hipFree(VScaleBuffer));
    HIP_CALL(hipFree(QOIndptrBuffer));
    HIP_CALL(hipFree(KVIndptrBuffer));
    HIP_CALL(hipFree(WorkPtrBuffer));
    HIP_CALL(hipFree(WorkInfoBuffer));
    HIP_CALL(hipFree(SplitOBuffer));
    HIP_CALL(hipFree(SplitLSEBuffer));
    HIP_CALL(hipFree(SinkBuffer));
    HIP_CALL(hipModuleUnload(module));

    if (reduce_partial_map.size() != 0) {
        PA_PARAMS_PS<IT> pa_params_ps(batch,
                                      kv_head_num,
                                      qlen_with_mtp,
                                      gqa,
                                      16,
                                      kv_seq_len,
                                      v_head_dim,
                                      PA_PAGE_SIZE,
                                      max_blocks_per_seq,
                                      is_causal != 0);
        paged_attention_reduce(pa_params_ps,
                               gpu_SplitO.data(),
                               gpu_SplitLSE.data(),
                               reduce_indptr,
                               reduce_final_map,
                               reduce_partial_map,
                               reinterpret_cast<cl_half*>(gpu_R),
                               gpu_LSE.data(),
                               tile_q,
                               true,
                               "gpu_d64");
    }
#else
    printf("DIS_KL is defined: skipping HIP kernel launch\n");
#endif

    if (dump_result) {
        fmha_dump_batch_inHex(cpu_R, "cpu_O.hex", batch, kv_head_num, q_seq_len_with_mtp, v_head_dim, BF16);
        fmha_dump_batch_inHex(gpu_R, "gpu_O.hex", batch, kv_head_num, q_seq_len_with_mtp, v_head_dim, BF16);
        if (valid_split_rows > 0) {
            fmha_dump_batch_inHex(cpu_SplitLSE.data(),
                                  "cpu_SplitLSE.hex",
                                  valid_split_rows,
                                  kv_head_num,
                                  gqa,
                                  1,
                                  FP32);
            fmha_dump_batch_inHex(gpu_SplitLSE.data(),
                                  "gpu_SplitLSE.hex",
                                  valid_split_rows,
                                  kv_head_num,
                                  gqa,
                                  1,
                                  FP32);
            fmha_dump_batch_inHex(cpu_SplitO.data(),
                                  "cpu_SplitO.hex",
                                  valid_split_rows,
                                  kv_head_num,
                                  gqa,
                                  v_head_dim,
                                  FP32);
            fmha_dump_batch_inHex(gpu_SplitO.data(),
                                  "gpu_SplitO.hex",
                                  valid_split_rows,
                                  kv_head_num,
                                  gqa,
                                  v_head_dim,
                                  FP32);
        }
    }

    int passed = 1;
#if defined(CHECK_RESULT)
    if (use_sink && has_cpu_ref && ran_gpu) {
        // pa_ref_dev_A16W8 / pa_persistent_fwd_ref do not currently take a
        // sink_buf parameter, so the CPU reference omits the sink merge while
        // the GPU kernel applies it. The fmha_check_result deltas would
        // spuriously fail. Skip the golden check (kernel-only sanity run).
        printf("Warning: use_sink=1; CPU reference does not model attention "
               "sink. Skipping golden check (extend pa_ref_dev_A16W8 / "
               "pa_persistent_fwd_ref with sink_buf to re-enable).\n");
    } else if (has_cpu_ref && ran_gpu) {
        if (valid_split_rows > 0) {
            int split_o_passed = fmha_check_result(gpu_SplitO.data(),
                                                   cpu_SplitO.data(),
                                                   valid_split_rows,
                                                   kv_head_num,
                                                   gqa,
                                                   v_head_dim,
                                                   FP32,
                                                   check_mode,
                                                   5e-3);
            int split_lse_passed = fmha_check_result(gpu_SplitLSE.data(),
                                                     cpu_SplitLSE.data(),
                                                     valid_split_rows,
                                                     kv_head_num,
                                                     gqa,
                                                     1,
                                                     FP32,
                                                     check_mode,
                                                     5e-3);
            passed = passed && split_o_passed && split_lse_passed;
        }
        int o_passed = fmha_check_result(gpu_R, cpu_R,
            batch, kv_head_num, q_seq_len_with_mtp, v_head_dim,
            BF16, check_mode, 5e-3);
        passed = passed && o_passed;
    } else if (!ran_gpu) {
        printf("Skip golden check because HIP kernel launch did not run\n");
    }
#endif

    delete[] Q_buf;
    delete[] Q_ref;
    delete[] K_buf;
    delete[] V_buf;
    delete[] cpu_R;
    delete[] gpu_R;
    delete[] BT_buf;
    delete[] sink_buf;
    delete[] K_logical;
    delete[] V_logical;
    delete[] K_ref_dense;
    delete[] V_ref_dense;
    delete[] K_ref_shuffle;
    delete[] V_ref_shuffle;
    delete[] K_scale_ref;
    delete[] V_scale_ref;
    delete[] K_scale_ref_shuffle;
    delete[] V_scale_ref_shuffle;
    delete[] BT_ref;
    delete[] context_lens_ref;

    return !passed;
}

static common_args make_common_args() {
    common_args args;
    args.shaderPath = "./";
    args.cs0 = "PA_DECODE_D64_1TG_4W_PS.co";
    args.lds_size = 320;
    args.init_pattern = 0;
    args.code_prefetch = 0;
    args.PerfCNT = 0;
    args.pte_preld = 0;
    args.dump_result = 1;
    args.check_mode = 1;
    SysMgr::Instance()->GetTestArgs()->GetParam("shaderPath", args.shaderPath);
    SysMgr::Instance()->GetTestArgs()->GetParam("shader_dir", args.shaderPath);
    SysMgr::Instance()->GetTestArgs()->GetParam("cs0", args.cs0);
    SysMgr::Instance()->GetTestArgs()->GetParam("lds_size", args.lds_size);
    SysMgr::Instance()->GetTestArgs()->GetParam("init_pattern", args.init_pattern);
    SysMgr::Instance()->GetTestArgs()->GetParam("code_pfth", args.code_prefetch);
    SysMgr::Instance()->GetTestArgs()->GetParam("code_prefetch", args.code_prefetch);
    SysMgr::Instance()->GetTestArgs()->GetParam("PerfCNT", args.PerfCNT);
    SysMgr::Instance()->GetTestArgs()->GetParam("pte_preld", args.pte_preld);
    SysMgr::Instance()->GetTestArgs()->GetParam("dump_result", args.dump_result);
    SysMgr::Instance()->GetTestArgs()->GetParam("check_mode", args.check_mode);
    printf("Debug: Get Param shaderPath=%s\n", args.shaderPath.c_str());
    printf("Debug: Get Param cs0=%s\n", args.cs0.c_str());
    printf("Debug: Get Param lds_size=%d\n", args.lds_size);
    printf("Debug: Get Param init_pattern=%d\n", args.init_pattern);
    printf("Debug: Get Param code_prefetch=%d\n", args.code_prefetch);
    printf("Debug: Get Param PerfCNT=%d\n", args.PerfCNT);
    printf("Debug: Get Param pte_preld=%d\n", args.pte_preld);
    printf("Debug: Get Param dump_result=%d\n", args.dump_result);
    printf("Debug: Get Param check_mode=%d\n", args.check_mode);
    return args;
}

int MI_KL_PA_DECODE_PERSISTENT_FWD(DATA_TYPE in_type, DATA_TYPE out_type) {
    common_args cmn_args = make_common_args();
    // KV cache = FP8 (uint8), Q = FP8 (uint8), Output = bf16 (uint16)
    return do_compute<cl_char, cl_char, unsigned short>(cmn_args, in_type, out_type);
}

} // namespace MI_KL_PA_DECODE_PERSISTENT

int main(int argc, char** argv) {
    std::vector<std::string> options;
    for (int i = 1; i < argc; ++i) {
        options.push_back(argv[i]);
    }
    SysMgr::Instance()->GetTestArgs()->SetOptions(options);

    unsigned int data_type = TACL_gemm_ns::FP8;
    SysMgr::Instance()->GetTestArgs()->GetParam("data_type", data_type);
    printf("Debug: Get Param data_type =%d\n", data_type);

    if (data_type == TACL_gemm_ns::FP8) {
        return MI_KL_PA_DECODE_PERSISTENT::MI_KL_PA_DECODE_PERSISTENT_FWD(
                   static_cast<TACL_gemm_ns::DATA_TYPE>(data_type),
                   TACL_gemm_ns::BF16);
    } else {
        printf("UNSUPPORTED data type for PA_DECODE_D64!!!!!!!\n");
        return 0;
    }
}
