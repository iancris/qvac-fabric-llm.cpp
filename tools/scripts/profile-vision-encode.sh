#!/bin/bash
# QVAC-18297 F6: Metal vision encode profiling
#
# Captures Metal System Traces for vision encode on Gemma4 E2B and Qwen3.5-2B.
# Run from the llama.cpp build directory after building llama-mtmd-cli.
#
# Usage:
#   ./tools/scripts/profile-vision-encode.sh [output_dir]
#
# Prerequisites:
#   - Xcode Command Line Tools (xcrun xctrace)
#   - Built llama-mtmd-cli binary
#   - Model files in ../models/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build"
OUTPUT_DIR="${1:-${SCRIPT_DIR}/../../results/traces/f6-vision-profile}"
BINARY="${BUILD_DIR}/bin/llama-mtmd-cli"

MODELS_DIR="${SCRIPT_DIR}/../../../models"
ELEPHANT="${SCRIPT_DIR}/../../../assets/elephant.jpg"

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: llama-mtmd-cli not found at $BINARY"
    echo "Build first: cmake --build build --target llama-mtmd-cli -j"
    exit 1
fi

declare -A CONFIGS
CONFIGS["gemma4-e2b"]="${MODELS_DIR}/gemma-4-e2b-it-q4_k_m.gguf|${MODELS_DIR}/mmproj-gemma-4-e2b-it-f16.gguf"
CONFIGS["qwen35-2b"]="${MODELS_DIR}/qwen3.5-vl-2b-instruct-q4_k_m.gguf|${MODELS_DIR}/mmproj-qwen3.5-vl-2b-instruct-f16.gguf"

for config_name in "${!CONFIGS[@]}"; do
    IFS='|' read -r model mmproj <<< "${CONFIGS[$config_name]}"

    if [ ! -f "$model" ] || [ ! -f "$mmproj" ]; then
        echo "SKIP: $config_name (model files not found)"
        continue
    fi

    trace_file="${OUTPUT_DIR}/${config_name}-vision-encode.trace"
    echo "=== Profiling $config_name ==="
    echo "  Model:  $model"
    echo "  Mmproj: $mmproj"
    echo "  Output: $trace_file"

    # Single inference with Metal System Trace
    xcrun xctrace record \
        --template "Metal System Trace" \
        --output "$trace_file" \
        --time-limit 30s \
        --launch -- \
        "$BINARY" \
        -m "$model" \
        --mmproj "$mmproj" \
        --image "$ELEPHANT" \
        -p "Describe this image briefly." \
        -n 16 \
        --ctx-size 4096 \
        --threads 4 \
        2>&1 | tee "${OUTPUT_DIR}/${config_name}-stdout.log"

    echo "  Done. Trace: $trace_file"
    echo ""

    # Cool-down between models
    echo "  Cooling down 30s..."
    sleep 30
done

echo "=== Profiling complete ==="
echo "Traces saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Open each .trace in Instruments.app"
echo "  2. Focus on the GPU track during image encode (first ~1s)"
echo "  3. Identify top-5 kernels by wall time"
echo "  4. Note: kernel dispatch gaps, memory stalls, shader compilation"
