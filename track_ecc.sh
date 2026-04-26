#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/dinov3/instruments/instance/eomt_base_640_ft.yaml"
CKPT="/mnt/FileExchange/users/zhouyang/instruments/models/eomt_ep12.ckpt"
DEVICE="cuda:0"

MODE="video"

INPUT_VIDEO="/mnt/FileExchange/users/zhouyang/instruments/data/oos_dir_range_clips_30fps/example.mp4"
INPUT_VIDEO_DIR="/mnt/FileExchange/users/zhouyang/instruments/data/oos_dir_range_clips_30fps"
INPUT_VIDEO_LIST=""

OUTPUT_PATH="/mnt/FileExchange/users/zhouyang/instruments/test/oos_track_ecc/example.mp4"
OUTPUT_DIR="/mnt/FileExchange/users/zhouyang/instruments/test/oos_track_ecc"

BATCH_SIZE=4
IO_WORKERS=4
PREFETCH_BATCHES=3
VIDEO_WORKERS=2

TRACK_THRESH=0.5
MATCH_THRESH=0.2
TRACK_BUFFER=30
MIN_HITS=2
IOU_THRESHOLD=0.3
LAMBDA_IOU=0.5
LAMBDA_MHD=0.25
LAMBDA_SHAPE=0.25
LAMBDA_EMB=1.5
DLO_BOOST_COEF=0.65

SAME_CLASS_ONLY=1
DISABLE_DLO_BOOST=0
DISABLE_DUO_BOOST=0
DISABLE_RICH_S=0
DISABLE_SOFT_BOOST=0
DISABLE_VARYING_TH=0
DISABLE_ECC=0
DISABLE_REID=0

ECC_SCALE=0.15
ECC_MAX_ITER=100
ECC_EPS=0.0001
REID_BATCH_SIZE=64
REID_CACHE_DIR="cache/embeddings_eomt"
ECC_CACHE_DIR="cache/ecc_eomt"

COMMON_ARGS=(
    --config "$CONFIG"
    --ckpt "$CKPT"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --io-workers "$IO_WORKERS"
    --prefetch-batches "$PREFETCH_BATCHES"
    --video-workers "$VIDEO_WORKERS"
    --track-thresh "$TRACK_THRESH"
    --match-thresh "$MATCH_THRESH"
    --track-buffer "$TRACK_BUFFER"
    --min-hits "$MIN_HITS"
    --iou-threshold "$IOU_THRESHOLD"
    --lambda-iou "$LAMBDA_IOU"
    --lambda-mhd "$LAMBDA_MHD"
    --lambda-shape "$LAMBDA_SHAPE"
    --lambda-emb "$LAMBDA_EMB"
    --dlo-boost-coef "$DLO_BOOST_COEF"
    --ecc-scale "$ECC_SCALE"
    --ecc-max-iter "$ECC_MAX_ITER"
    --ecc-eps "$ECC_EPS"
    --reid-batch-size "$REID_BATCH_SIZE"
    --reid-cache-dir "$REID_CACHE_DIR"
    --ecc-cache-dir "$ECC_CACHE_DIR"
)

if [[ "$SAME_CLASS_ONLY" == "1" ]]; then
    COMMON_ARGS+=(--same-class-only)
fi
if [[ "$DISABLE_DLO_BOOST" == "1" ]]; then
    COMMON_ARGS+=(--disable-dlo-boost)
fi
if [[ "$DISABLE_DUO_BOOST" == "1" ]]; then
    COMMON_ARGS+=(--disable-duo-boost)
fi
if [[ "$DISABLE_RICH_S" == "1" ]]; then
    COMMON_ARGS+=(--disable-rich-s)
fi
if [[ "$DISABLE_SOFT_BOOST" == "1" ]]; then
    COMMON_ARGS+=(--disable-soft-boost)
fi
if [[ "$DISABLE_VARYING_TH" == "1" ]]; then
    COMMON_ARGS+=(--disable-varying-th)
fi
if [[ "$DISABLE_ECC" == "1" ]]; then
    COMMON_ARGS+=(--disable-ecc)
fi
if [[ "$DISABLE_REID" == "1" ]]; then
    COMMON_ARGS+=(--disable-reid)
fi

if [[ "$MODE" == "video" ]]; then
    python tools/media_instance_track_ecc_infer.py \
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
    python tools/media_instance_track_ecc_infer.py \
        "${COMMON_ARGS[@]}" \
        --mode videos \
        "${EXTRA_ARGS[@]}" \
        --output-dir "$OUTPUT_DIR"
else
    echo "不支持的 MODE: $MODE"
    echo "track_ecc.sh 仅支持 video / videos"
    exit 1
fi
