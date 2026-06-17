#!/bin/bash
# Build sp3 -> .co and run pa_decode test.
# Usage:
#   bash build_and_test.sh                   # build + run default test from run.sh
#   bash build_and_test.sh build             # build only
#   bash build_and_test.sh test              # test only (skip build)
#   bash build_and_test.sh "test_args"       # build + run with custom test args
#
# Examples:
#   bash build_and_test.sh "-b 1 -kvh 1 -c 1024 --scales 1.69 1.7329 1.2276 --varlen"
#   bash build_and_test.sh "-b 1 64 128 -kvh 1 -c 1024 --scales 1.69 1.7329 1.2276 --varlen"
#
# SP3 source: edit <aiter>/pa_kernels/PA_DECODE_D64_1TG_4W_PS.sp3
# Output .co replaces: <aiter>/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8.co
set -euo pipefail

# Paths derived from this script's own location (this script lives at the aiter root).
AITER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$AITER_DIR/pa_kernels"
SP3_SRC="$BUILD_DIR/PA_DECODE_D64_1TG_4W_PS.sp3"
CO_TARGET="$AITER_DIR/hsa/gfx1250/pa_decode_bf16/pa_decode_bf16_d64_page256_gqa8.co"
CO_BACKUP="$CO_TARGET.bak"
TEST_SCRIPT="$AITER_DIR/op_tests/test_pa_decode_bf16_asm.py"
DEFAULT_TEST_ARGS="-b 65 -kvh 8 -c 16385 8193 --varlen -m 0 1 2 3 --scales 1.69 1.7329 1.2276 --sink"

MODE="${1:-all}"
TEST_ARGS="${1:-$DEFAULT_TEST_ARGS}"

export PATH="/opt/rocm-7.13-95/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm-7.13-95/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-3}"
export ENABLE_CK="${ENABLE_CK:-0}"
export ENABLE_FLYDSL="${ENABLE_FLYDSL:-0}"
export AITER_REBUILD="${AITER_REBUILD:-1}"

do_build() {
    echo "========================================"
    echo "Building sp3 -> .co"
    echo "  SP3 source: $SP3_SRC"
    echo "========================================"

    # Backup original .co (only first time)
    if [ ! -f "$CO_BACKUP" ]; then
        cp -f "$CO_TARGET" "$CO_BACKUP"
        echo "  Backed up original .co -> $CO_BACKUP"
    fi

    # Run the build. build.sh wipes its own work/ first (only fresh artifacts kept)
    # and installs the resulting .co straight to $CO_TARGET.
    cd "$BUILD_DIR"
    bash build.sh

    echo "  Installed .co -> $CO_TARGET"
    echo "  Build done."
    echo ""
}

do_test() {
    local args="$1"
    echo "========================================"
    echo "Running test: python $TEST_SCRIPT $args"
    echo "  PATH=$PATH"
    echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "  HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES ENABLE_CK=$ENABLE_CK ENABLE_FLYDSL=$ENABLE_FLYDSL AITER_REBUILD=$AITER_REBUILD"
    echo "========================================"
    cd "$AITER_DIR"
    rm -f aiter/jit/module_pa*
    python "$TEST_SCRIPT" $args
    echo ""
    echo "Test done."
}

do_restore() {
    if [ -f "$CO_BACKUP" ]; then
        cp -f "$CO_BACKUP" "$CO_TARGET"
        echo "Restored original .co from backup."
    else
        echo "No backup found at $CO_BACKUP"
    fi
}

case "$MODE" in
    build)
        do_build
        ;;
    test)
        shift
        do_test "${*:-$DEFAULT_TEST_ARGS}"
        ;;
    restore)
        do_restore
        ;;
    all)
        do_build
        do_test "$DEFAULT_TEST_ARGS"
        ;;
    *)
        # Treat $MODE as test args: build + test with those args
        do_build
        do_test "$*"
        ;;
esac
