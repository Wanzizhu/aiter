#!/bin/bash
# Build the gfx1250 PA decode code object and install it into aiter.
#
# Two stages (may be different machines):
#   Stage 1 (CSIM, needs sp3 + libsp3.so with the MI400 backend):
#       sp3 compiles the .sp3 -> .hex, then sp3cvt embeds it into a ready-to-
#       assemble .s using the hand-written kernarg ABI template (kernarg.template.s,
#       which mirrors the 0x160 KernelArgs in asm_pa_decode_bf16.cu / pa_ps.cpp).
#   Stage 2 (ROCm, needs amdclang++ with gfx1250 support):
#       amdclang++ assembles the .s -> .co, which is copied into the aiter tree.
#
# Run with no args to do both stages on a host that has both toolchains, or set
# STAGE=1 / STAGE=2 to split across machines.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AITER_DIR="$(cd "$HERE/.." && pwd)"   # aiter repo root (pa_kernels' parent)

# --- configurable ----------------------------------------------------------
# .wsm.cornerfix = the willa-softmax (.wsm) baseline with the corner-case fix.
# This is the variant the deployed .co is built from; it carries the by-value
# softmax_scale kernarg (0x160) and the sink path, so it serves both sink and
# non-sink callers.
VARIANT="${VARIANT:-$HERE/PA_DECODE_D64_1TG_4W_PS.sp3}"
# Kernel symbol = Itanium mangling of aiter::<co-basename> (fmha_fwd_bf16.csv style).
# MUST match knl_name in pa_decode_bf16.csv.
KERNEL_NAME="${KERNEL_NAME:-_ZN5aiter31pa_decode_bf16_d64_page256_gqa8E}"
WORKBASE="${WORKBASE:-pa_decode_bf16_d64_page256_gqa8}"  # intermediate file stem + .co name
# LDS = 327680 (NOT 163840): the kernel runs in WGP mode (two CUs combine LDS),
# so the group_segment_fixed_size is the combined 320 KB.  Using 163840 + the
# wrong descriptor produces a .co that runs but reads its LDS-staged K/V as zero
# (uniform softmax, all-zero output).
LDS_BYTES="${LDS_BYTES:-327680}"
VGPR="${VGPR:-1024}"
SGPR="${SGPR:-106}"
SP3="${SP3:-$HERE/sp3}"
SP3_LIBDIR="${SP3_LIBDIR:-$HERE}"
# Use the poc_kl sp3cvt (NOT the csim one): its kernel_template.s sets the correct
# gfx1250 descriptor (.amdhsa_wavefront_size32 1, WGP mode, system_vgpr_workitem_id 2,
# inst_pref_size) that the csim template was missing -> that omission was what made
# the kernel compute zeros.  It also auto-patches the 22-arg 0x160 ABI from pa_ps.cpp.
SP3CVT="${SP3CVT:-$HERE/sp3cvt.py}"
PA_PS_CPP="${PA_PS_CPP:-$HERE/pa_ps.cpp}"  # kernarg ABI source
PYTHON="${PYTHON:-python}"                           # sp3cvt needs py>=3.8 (shlex.join)
# Local ROCm 7.2.4 LLVM (LLVM 22, has gfx1250) extracted from the rocm-llvm rpm;
# override with a system ROCm install if present.
AMDCLANG="${AMDCLANG:-/opt/venv/bin/amdclang++}"
# Install target: the .co consumed by aiter (must be named after WORKBASE).
AITER_CO="${AITER_CO:-$AITER_DIR/hsa/gfx1250/pa_decode_bf16/${WORKBASE}.co}"
STAGE="${STAGE:-all}"

WORK="$HERE/work"
mkdir -p "$WORK"
# sp3cvt names every artifact after the input .sp3 stem; drive the work-file
# names off that instead of copying/renaming the .sp3 into work.
STEM="$(basename "$VARIANT" .sp3)"
S_OUT="$WORK/${STEM}.s"

stage1() {
    echo "== Stage 1: poc_kl sp3cvt compile + embed -> $S_OUT (symbol=$KERNEL_NAME) =="
    # Start clean: work holds ONLY freshly generated artifacts (.text/.bin/.hex/.s/.co).
    rm -rf "${WORK:?}"/*
    # sp3cvt reads the .sp3 (VARIANT) and pa_ps.cpp in place (pa_kernels/) and writes
    # all generated artifacts into WORK via --out-dir -> nothing is copied into work.
    # It builds its <stem>.s from kernel_template.s (correct gfx1250 descriptor) and
    # patches the 22-arg 0x160 ABI from the given --cpp.
    ( cd "$WORK" && LD_LIBRARY_PATH="$SP3_LIBDIR" "$PYTHON" "$SP3CVT" \
        -i "$VARIANT" -n "$KERNEL_NAME" --cpp "$PA_PS_CPP" -o "$WORK" \
        -lds "$LDS_BYTES" -v "$VGPR" -s "$SGPR" --sp3 "$SP3" )
    echo "   wrote $S_OUT"
}

stage2() {
    echo "== Stage 2: amdclang++ assemble -> .co =="
    [ -f "$S_OUT" ] || { echo "ERROR: $S_OUT not found; run STAGE=1 first"; exit 1; }
    "$AMDCLANG" -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 \
        "$S_OUT" -o "$WORK/${STEM}.co"
    # Single copy, straight to the aiter install target.
    cp -f "$WORK/${STEM}.co" "$AITER_CO"
    echo "   installed -> $AITER_CO"
}

case "$STAGE" in
    1) stage1 ;;
    2) stage2 ;;
    all) stage1; stage2 ;;
    *) echo "STAGE must be 1, 2, or all"; exit 1 ;;
esac
echo "Done."
