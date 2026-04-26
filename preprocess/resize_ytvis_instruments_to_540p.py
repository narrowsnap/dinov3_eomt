#!/usr/bin/env python3

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


DEFAULT_INPUT_JSON = Path(
    "/mnt/data2/datasets/instruments_2.0/ytvis_2022_instruments/train.json"
)
DEFAULT_IMAGES_ROOT = Path(
    "/mnt/data2/datasets/instruments_2.0/ytvis_2022_instruments/train/JPEGImages"
)
DEFAULT_MAX_HEIGHT = 540
DEFAULT_WORKERS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 ytvis instruments 原始图片与 train.json 统一修正到 540p，并修复 mask 尺寸与图片不一致的问题。"
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


def decode_segmentation(segmentation, height: int, width: int):
    if isinstance(segmentation, dict):
        return coco_mask.decode(segmentation)
    if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], dict):
        merged = coco_mask.merge(segmentation)
        return coco_mask.decode(merged)
    rles = coco_mask.frPyObjects(segmentation, height, width)
    rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
    return coco_mask.decode(rle)


def encode_mask(mask) -> dict[str, Any]:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype("uint8")))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def resize_mask(mask: np.ndarray, new_hw: tuple[int, int]) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype("uint8") * 255)
    pil_mask = pil_mask.resize((new_hw[1], new_hw[0]), resample=Image.Resampling.NEAREST)
    return (np.array(pil_mask) > 0).astype("uint8")


def resize_image_if_needed(image_path: Path, max_height: int, dry_run: bool) -> tuple[bool, tuple[int, int], tuple[int, int]]:
    with Image.open(image_path) as image:
        width, height = image.size
        old_hw = (height, width)
        if height <= max_height:
            return False, old_hw, old_hw

        scale = max_height / height
        new_width = max(1, round(width * scale))
        new_hw = (max_height, new_width)
        resized = image.resize((new_width, max_height), resample=Image.Resampling.BILINEAR)
        if not dry_run:
            resized.save(image_path, quality=95)
        return True, old_hw, new_hw


def process_video(
    video: dict[str, Any],
    annotations: list[dict[str, Any]],
    images_root: Path,
    max_height: int,
    dry_run: bool,
):
    first_image_path = images_root / video["file_names"][0]
    changed, old_hw, new_hw = resize_image_if_needed(first_image_path, max_height, dry_run)

    for rel_path in video["file_names"][1:]:
        image_path = images_root / rel_path
        img_changed, img_old_hw, img_new_hw = resize_image_if_needed(image_path, max_height, dry_run)
        if img_old_hw != old_hw or img_new_hw != new_hw:
            raise RuntimeError(f"同一视频帧尺寸不一致: {rel_path}")
        changed = changed or img_changed

    mismatch_frames = 0
    for ann in annotations:
        segmentations = ann.get("segmentations", [])
        areas = ann.get("areas", [])
        bboxes = ann.get("bboxes", [])
        for frame_idx, segmentation in enumerate(segmentations):
            if not segmentation:
                continue
            mask = decode_segmentation(segmentation, old_hw[0], old_hw[1])
            if tuple(mask.shape[:2]) != old_hw:
                mismatch_frames += 1
                mask = resize_mask(mask, old_hw)
            if old_hw != new_hw:
                mask = resize_mask(mask, new_hw)
            encoded = encode_mask(mask)
            ann["segmentations"][frame_idx] = encoded
            if frame_idx < len(areas):
                areas[frame_idx] = float(mask.sum())
            if frame_idx < len(bboxes):
                bboxes[frame_idx] = [float(x) for x in coco_mask.toBbox(encoded).tolist()]

    return int(video["id"]), changed, old_hw, new_hw, mismatch_frames


def main() -> None:
    args = parse_args()
    data = load_json(args.input_json)
    videos = sorted(data["videos"], key=lambda item: int(item["id"]))
    video_by_id = {int(video["id"]): video for video in videos}
    annotations_by_video = {}
    for ann in data["annotations"]:
        annotations_by_video.setdefault(int(ann["video_id"]), []).append(ann)

    resized_videos = 0
    mismatch_frames = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for video_id, changed, old_hw, new_hw, mismatch_frames_video in executor.map(
            lambda video: process_video(
                video,
                annotations_by_video.get(int(video["id"]), []),
                args.images_root,
                args.max_height,
                args.dry_run,
            ),
            videos,
        ):
            video = video_by_id[video_id]
            video["height"], video["width"] = new_hw
            if changed:
                resized_videos += 1
            mismatch_frames += mismatch_frames_video

    if not args.dry_run:
        write_json(args.input_json, data)

    print(
        json.dumps(
            {
                "input_json": str(args.input_json),
                "images_root": str(args.images_root),
                "max_height": args.max_height,
                "dry_run": args.dry_run,
                "videos": len(videos),
                "annotations": len(data["annotations"]),
                "resized_videos": resized_videos,
                "fixed_mismatch_frames": mismatch_frames,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
