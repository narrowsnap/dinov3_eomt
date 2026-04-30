#!/usr/bin/env bash
set -euo pipefail

# CONFIG="configs/dinov3/instruments/instance/eomt_base_640_ft.yaml"
CONFIG="configs/dinov3/ultrasound/instance/eomt_base_ft.yaml"
CKPT="eomt/vtvucmxe/checkpoints/last.ckpt"
DEVICE="cuda:0"

MODE="videos"

INPUT_VIDEO=""
INPUT_VIDEO_DIR=""
INPUT_VIDEO_LIST="/mnt/FileExchange/users/zhouyang/instruments/iterupdate/术中超声/batch2/remain/remaining_video_paths_500_700.txt"
INPUT_IMAGE=""
INPUT_IMAGE_DIR=""

OUTPUT_PATH=""
OUTPUT_DIR="/mnt/FileExchange/users/zhouyang/instruments/iterupdate/术中超声/batch2/"

BATCH_SIZE=4
IO_WORKERS=4
PREFETCH_BATCHES=3
VIDEO_WORKERS=2
IMAGE_CACHE_SIZE=256

COMMON_ARGS=(
    --config "$CONFIG"
    --ckpt "$CKPT"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --io-workers "$IO_WORKERS"
    --prefetch-batches "$PREFETCH_BATCHES"
    --video-workers "$VIDEO_WORKERS"
    --image-cache-size "$IMAGE_CACHE_SIZE"
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
    if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
        echo "MODE=videos 时，请设置 INPUT_VIDEO_DIR 或 INPUT_VIDEO_LIST"
        exit 1
    fi
    python tools/media_instance_infer.py \
        "${COMMON_ARGS[@]}" \
        --mode videos \
        "${EXTRA_ARGS[@]}" \
        --output-dir "$OUTPUT_DIR"
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
