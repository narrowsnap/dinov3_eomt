#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_GOLD_BOX_JSON = "/mnt/FileExchange/users/zhouyang/instruments/data/needle_gold_clip_box_dict.json"
DEFAULT_SOURCE_IMAGE_ROOT = "/mnt/data2/datasets/instruments_2.0/all_ins_data_2025_labelstudio_oos/images"
DEFAULT_OUTPUT_ROOT = "/mnt/FileExchange/users/zhouyang/instruments/data/needle_gold_suture_only_coco"

TARGET_CATEGORY_ID = 1
TARGET_CATEGORY_NAME = "缝针"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出金标准缝针单类 COCO 子集")
    parser.add_argument("--gold-box-json", default=DEFAULT_GOLD_BOX_JSON)
    parser.add_argument("--source-image-root", default=DEFAULT_SOURCE_IMAGE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "auto"),
        default="auto",
        help="输出图片使用软链接还是复制，auto 表示优先软链接失败后自动复制",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清空输出目录下的 train/val/annotations/summary.json/missing_images.json 后重建",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def reset_output_root(output_root: Path) -> None:
    for name in ("train", "val", "annotations"):
        target = output_root / name
        if target.exists():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "missing_images.json"):
        target = output_root / name
        if target.exists():
            target.unlink()


def resolve_source_image(source_root: Path, file_name: str) -> Path | None:
    relative_candidate = source_root / file_name
    if relative_candidate.exists():
        return relative_candidate

    basename_candidate = source_root / Path(file_name).name
    if basename_candidate.exists():
        return basename_candidate

    return None


def normalize_split_and_relative_name(file_name: str) -> tuple[str, str]:
    normalized = file_name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts:
        return "train", Path(file_name).name

    split = parts[0] if parts[0] in {"train", "val"} else "train"
    if parts[0] in {"train", "val"}:
        parts = parts[1:]

    while parts and parts[0] in {"train", "val"}:
        parts = parts[1:]

    relative_name = "/".join(parts) if parts else Path(file_name).name
    return split, relative_name


def clamp_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    x = max(0.0, min(x, float(width)))
    y = max(0.0, min(y, float(height)))
    w = max(0.0, min(w, float(width) - x))
    h = max(0.0, min(h, float(height) - y))
    return [x, y, w, h]


def main() -> None:
    args = parse_args()
    gold_box_json = Path(args.gold_box_json)
    source_root = Path(args.source_image_root)
    output_root = Path(args.output_root)

    if args.clean:
        reset_output_root(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "annotations").mkdir(parents=True, exist_ok=True)

    gold_data = json.loads(gold_box_json.read_text(encoding="utf-8"))

    split_images: dict[str, dict[str, dict]] = defaultdict(dict)
    split_annotations: dict[str, list[dict]] = defaultdict(list)
    missing_images: list[dict] = []
    annotation_id = 1

    for clip_record in gold_data.values():
        for image_record in clip_record.get("images", []):
            file_name = str(image_record["file_name"])
            split, relative_name = normalize_split_and_relative_name(file_name)
            source_path = resolve_source_image(source_root, file_name)
            if source_path is None:
                missing_images.append(
                    {
                        "clip_name": clip_record.get("clip_name"),
                        "file_name": file_name,
                    }
                )
                continue

            split_images[split][file_name] = {
                "id": int(image_record["image_id"]),
                "file_name": relative_name,
                "width": int(image_record["width"]),
                "height": int(image_record["height"]),
                "source_path": str(source_path),
            }

            for box in image_record.get("boxes", []):
                bbox = box.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                width = int(image_record["width"])
                height = int(image_record["height"])
                bbox = clamp_bbox(bbox, width=width, height=height)
                area = float(bbox[2] * bbox[3])
                if area <= 0:
                    continue
                split_annotations[split].append(
                    {
                        "id": annotation_id,
                        "image_id": int(image_record["image_id"]),
                        "category_id": TARGET_CATEGORY_ID,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": int(box.get("iscrowd", 0) or 0),
                        "segmentation": [],
                    }
                )
                annotation_id += 1

    categories = [
        {
            "id": TARGET_CATEGORY_ID,
            "name": TARGET_CATEGORY_NAME,
            "supercategory": "instrument",
        }
    ]

    linked_images = 0
    split_summaries: dict[str, dict] = {}
    for split in ("train", "val"):
        images = sorted(split_images[split].values(), key=lambda item: int(item["id"]))
        annotations = sorted(split_annotations[split], key=lambda item: int(item["id"]))

        instances = {
            "info": {
                "description": "needle gold suture-only subset",
                "version": "1.0",
            },
            "licenses": [],
            "images": [
                {
                    "id": int(item["id"]),
                    "file_name": str(item["file_name"]),
                    "width": int(item["width"]),
                    "height": int(item["height"]),
                }
                for item in images
            ],
            "annotations": annotations,
            "categories": categories,
        }

        annotation_path = output_root / "annotations" / f"instances_{split}.json"
        annotation_path.write_text(json.dumps(instances, ensure_ascii=False, indent=2), encoding="utf-8")

        split_root = output_root / split
        for item in images:
            target_path = split_root / item["file_name"]
            ensure_parent(target_path)
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            if args.link_mode == "copy":
                shutil.copyfile(item["source_path"], target_path)
            elif args.link_mode == "symlink":
                os.symlink(item["source_path"], target_path)
            else:
                try:
                    os.symlink(item["source_path"], target_path)
                except OSError:
                    shutil.copyfile(item["source_path"], target_path)
            linked_images += 1

        split_summaries[split] = {
            "num_images": len(images),
            "num_annotations": len(annotations),
        }

    summary = {
        "gold_box_json": str(gold_box_json),
        "source_image_root": str(source_root),
        "output_root": str(output_root),
        "link_mode": args.link_mode,
        "num_images": sum(item["num_images"] for item in split_summaries.values()),
        "num_annotations": sum(item["num_annotations"] for item in split_summaries.values()),
        "num_linked_images": linked_images,
        "num_missing_images": len(missing_images),
        "splits": split_summaries,
    }

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "missing_images.json").write_text(
        json.dumps(missing_images, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
