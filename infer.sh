#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/dinov3/instruments/instance/eomt_base_640_ft.yaml"
CKPT="/home/zhouyang/work/gitops/eomt/eomt/vh1webla/checkpoints/last.ckpt"
DEVICE="cuda:0"
BATCH_SIZE=4
IO_WORKERS=4
PREFETCH_BATCHES=3
IMAGE_CACHE_SIZE=256
VIDEO_WORKERS=2

# 直接在这里填写推理输入，不再从命令行传参
MODE="videos"
INPUT_VIDEO=""
INPUT_VIDEO_DIR="/tmp/eomt_videos"
INPUT_VIDEO_LIST=""
INPUT_IMAGE=""
INPUT_IMAGE_DIR=""
OUTPUT_PATH="/tmp/eomt_infer_outputs/single_rendered.mp4"
OUTPUT_DIR="/tmp/eomt_infer_outputs/videos"

COMMON_ARGS=(
    --config "$CONFIG"
    --ckpt "$CKPT"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --io-workers "$IO_WORKERS"
    --prefetch-batches "$PREFETCH_BATCHES"
    --image-cache-size "$IMAGE_CACHE_SIZE"
    --video-workers "$VIDEO_WORKERS"
)

if [[ "$MODE" == "video" ]]; then
    python tools/media_instance_infer.py \
        "${COMMON_ARGS[@]}" \
        --mode video \
        --input-video "$INPUT_VIDEO" \
        --output-path "$OUTPUT_PATH"
elif [[ "$MODE" == "videos" ]]; then
    EXTRA_ARGS=()
    if [[ -n "$INPUT_VIDEO_DIR" ]]; then
        EXTRA_ARGS+=(--input-video-dir "$INPUT_VIDEO_DIR")
    fi
    if [[ -n "$INPUT_VIDEO_LIST" ]]; then
        EXTRA_ARGS+=(--input-video-list "$INPUT_VIDEO_LIST")
    fi
    python tools/media_instance_infer.py \
        "${COMMON_ARGS[@]}" \
        --mode videos \
        "${EXTRA_ARGS[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --output-path "$OUTPUT_DIR"
elif [[ "$MODE" == "image" ]]; then
    python tools/media_instance_infer.py \
        "${COMMON_ARGS[@]}" \
        --mode image \
        --input-image "$INPUT_IMAGE" \
        --output-path "$OUTPUT_PATH"
elif [[ "$MODE" == "images" ]]; then
    python tools/media_instance_infer.py \
        "${COMMON_ARGS[@]}" \
        --mode images \
        --input-dir "$INPUT_IMAGE_DIR" \
        --output-path "$OUTPUT_DIR"
else
    echo "不支持的 MODE: $MODE"
    exit 1
fi
