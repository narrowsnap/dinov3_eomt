#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/dinov3/instruments/instance/eomt_base_640_ft.yaml}"
CKPT="${CKPT:-/home/zhouyang/work/gitops/eomt/eomt/vh1webla/checkpoints/last.ckpt}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/eomt_infer_outputs}"

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <视频文件|视频目录|视频路径列表文件>"
    echo "可选环境变量: CONFIG CKPT DEVICE BATCH_SIZE OUTPUT_DIR"
    exit 1
fi

INPUT="$1"

run_one_video() {
    local video_path="$1"
    local base_name
    base_name="$(basename "$video_path")"
    local stem="${base_name%.*}"
    local output_path="${OUTPUT_DIR}/${stem}_rendered.mp4"

    mkdir -p "$OUTPUT_DIR"

    echo "推理视频: $video_path"
    echo "输出视频: $output_path"

    python tools/media_instance_infer.py \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --mode video \
        --input-video "$video_path" \
        --output-path "$output_path" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE"
}

if [[ -f "$INPUT" && ( "$INPUT" == *.mp4 || "$INPUT" == *.avi || "$INPUT" == *.mov || "$INPUT" == *.mkv ) ]]; then
    run_one_video "$INPUT"
    exit 0
fi

if [[ -d "$INPUT" ]]; then
    while IFS= read -r video_path; do
        run_one_video "$video_path"
    done < <(find "$INPUT" -maxdepth 1 -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" \) | sort)
    exit 0
fi

if [[ -f "$INPUT" ]]; then
    while IFS= read -r video_path; do
        video_path="${video_path#"${video_path%%[![:space:]]*}"}"
        video_path="${video_path%"${video_path##*[![:space:]]}"}"
        [[ -z "$video_path" ]] && continue
        [[ "$video_path" =~ ^# ]] && continue
        run_one_video "$video_path"
    done < "$INPUT"
    exit 0
fi

echo "无法识别输入: $INPUT"
exit 1
