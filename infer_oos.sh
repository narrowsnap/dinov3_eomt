#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/dinov3/instruments/instance/eomt_base_640_ft.yaml"
CKPT="/mnt/FileExchange/users/zhouyang/instruments/models/eomt_ep12.ckpt"
DEVICE="cuda:0"
BATCH_SIZE=4
IO_WORKERS=4
PREFETCH_BATCHES=3
IMAGE_CACHE_SIZE=256
VIDEO_WORKERS=2
OOS_ANNOTATION_JSON="/mnt/FileExchange/users/zhouyang/instruments/data/oos_dir_range_clips_30fps_box_dict_light.json"

# 直接在这里填写推理输入，不再从命令行传参
MODE="videos"
INPUT_VIDEO=""
# 多视频目录输入时填写目录；如果输入的是“视频路径列表文件”，请填写 INPUT_VIDEO_LIST
# INPUT_VIDEO_DIR="/mnt/data/projects/instruments/videomt/test/inputs"
INPUT_VIDEO_DIR="/mnt/FileExchange/users/zhouyang/instruments/data/oos_dir_range_clips_30fps"
INPUT_VIDEO_LIST=""
# INPUT_VIDEO_LIST="/mnt/data/projects/instruments/annos/predic_17416_20260425_excluded_random100_paths.txt"
INPUT_IMAGE=""
INPUT_IMAGE_DIR=""
OUTPUT_PATH=""
OUTPUT_DIR="/mnt/FileExchange/users/zhouyang/instruments/test/oos"

COMMON_ARGS=(
    --config "$CONFIG"
    --ckpt "$CKPT"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --io-workers "$IO_WORKERS"
    --prefetch-batches "$PREFETCH_BATCHES"
    --image-cache-size "$IMAGE_CACHE_SIZE"
    --video-workers "$VIDEO_WORKERS"
    --oos-annotation-json "$OOS_ANNOTATION_JSON"
)

if [[ "$MODE" == "video" ]]; then
    python tools/media_instance_infer_oos.py \
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
    python tools/media_instance_infer_oos.py \
        "${COMMON_ARGS[@]}" \
        --mode videos \
        "${EXTRA_ARGS[@]}" \
        --output-dir "$OUTPUT_DIR" \
        --output-path "$OUTPUT_DIR"
elif [[ "$MODE" == "image" ]]; then
    python tools/media_instance_infer_oos.py \
        "${COMMON_ARGS[@]}" \
        --mode image \
        --input-image "$INPUT_IMAGE" \
        --output-path "$OUTPUT_PATH"
elif [[ "$MODE" == "images" ]]; then
    python tools/media_instance_infer_oos.py \
        "${COMMON_ARGS[@]}" \
        --mode images \
        --input-dir "$INPUT_IMAGE_DIR" \
        --output-path "$OUTPUT_DIR"
else
    echo "不支持的 MODE: $MODE"
    exit 1
fi
