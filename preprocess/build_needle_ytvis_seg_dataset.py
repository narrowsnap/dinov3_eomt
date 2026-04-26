#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_YTVIS_JSON = "/mnt/data2/datasets/instruments_2.0/ytvis_2022_instruments/train.json"
DEFAULT_IMAGE_ROOT = "/mnt/data2/datasets/instruments_2.0/ytvis_2022_instruments/train/JPEGImages"
DEFAULT_OUTPUT_ROOT = "/mnt/data2/datasets/instruments_2.0/needle_gold_suture_only_seg_coco"
TARGET_CATEGORY_NAME = "缝针"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 YTVIS train.json 导出缝针真分割单类 COCO 子集")
    parser.add_argument("--ytvis-json", default=DEFAULT_YTVIS_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "auto"),
        default="auto",
        help="图片输出方式，auto 表示优先软链接，失败后复制",
    )
    parser.add_argument("--clean", action="store_true", help="重建前清空输出目录")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_non_empty_segmentation(segmentation) -> bool:
    if segmentation is None:
        return False
    if isinstance(segmentation, list):
        return any(part not in (None, [], {}) for part in segmentation)
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        return bool(counts)
    return False


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copyfile(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    args = parse_args()
    ytvis_json = Path(args.ytvis_json)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)

    if args.clean:
        reset_dir(output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    annotations_dir = output_root / "annotations"
    train_dir = output_root / "train"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    with ytvis_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data["categories"]
    videos = data["videos"]
    annotations = data["annotations"]

    target_categories = [cat for cat in categories if cat["name"] == TARGET_CATEGORY_NAME]
    if not target_categories:
        raise RuntimeError(f"没有找到类别: {TARGET_CATEGORY_NAME}")
    target_category = target_categories[0]
    target_category_id = int(target_category["id"])

    videos_by_id = {int(video["id"]): video for video in videos}
    image_records: dict[str, dict] = {}
    annotations_out: list[dict] = []
    missing_images: list[str] = []
    annotation_id = 1
    image_id = 1

    for ann in annotations:
        if int(ann["category_id"]) != target_category_id:
            continue

        video = videos_by_id.get(int(ann["video_id"]))
        if video is None:
            continue

        file_names = video.get("file_names", [])
        segmentations = ann.get("segmentations", [])
        bboxes = ann.get("bboxes", [])
        areas = ann.get("areas", [])
        width = int(video["width"])
        height = int(video["height"])

        for frame_idx, file_name in enumerate(file_names):
            segmentation = segmentations[frame_idx] if frame_idx < len(segmentations) else None
            if not is_non_empty_segmentation(segmentation):
                continue

            bbox = bboxes[frame_idx] if frame_idx < len(bboxes) else None
            area = areas[frame_idx] if frame_idx < len(areas) else None
            if bbox is None or len(bbox) != 4:
                continue
            if area is None:
                area = float(bbox[2] * bbox[3])

            rel_file_name = Path(file_name).as_posix()
            if rel_file_name not in image_records:
                image_records[rel_file_name] = {
                    "id": image_id,
                    "file_name": Path(rel_file_name).name,
                    "width": width,
                    "height": height,
                    "source_path": str(image_root / rel_file_name),
                }
                image_id += 1

            annotations_out.append(
                {
                    "id": annotation_id,
                    "image_id": image_records[rel_file_name]["id"],
                    "category_id": 1,
                    "bbox": [float(v) for v in bbox],
                    "area": float(area),
                    "iscrowd": int(ann.get("iscrowd", 0) or 0),
                    "segmentation": segmentation,
                }
            )
            annotation_id += 1

    linked_images = 0
    final_images = []
    for rel_file_name, record in sorted(image_records.items(), key=lambda item: item[1]["id"]):
        source_path = Path(record["source_path"])
        if not source_path.exists():
            missing_images.append(rel_file_name)
            continue

        target_path = train_dir / record["file_name"]
        safe_link_or_copy(source_path, target_path, args.link_mode)
        linked_images += 1
        final_images.append(
            {
                "id": int(record["id"]),
                "file_name": record["file_name"],
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
        )

    valid_image_ids = {img["id"] for img in final_images}
    annotations_out = [ann for ann in annotations_out if ann["image_id"] in valid_image_ids]

    coco_out = {
        "info": {
            "description": "needle ytvis segmentation-only coco subset",
            "version": "1.0",
        },
        "licenses": [],
        "images": final_images,
        "annotations": annotations_out,
        "categories": [
            {
                "id": 1,
                "name": TARGET_CATEGORY_NAME,
                "supercategory": "instrument",
            }
        ],
    }

    train_json_path = annotations_dir / "instances_train.json"
    train_json_path.write_text(json.dumps(coco_out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "ytvis_json": str(ytvis_json),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "target_category_name": TARGET_CATEGORY_NAME,
        "target_category_id_in_ytvis": target_category_id,
        "num_images": len(final_images),
        "num_annotations": len(annotations_out),
        "num_missing_images": len(missing_images),
        "link_mode": args.link_mode,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "missing_images.json").write_text(
        json.dumps(missing_images, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
