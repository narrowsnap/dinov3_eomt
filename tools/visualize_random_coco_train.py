#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

try:
    from pycocotools import mask as coco_mask
except ImportError:  # pragma: no cover
    coco_mask = None


DEFAULT_DATA_ROOT = "/mnt/data2/datasets/instruments_2.0/needle_gold_suture_only_coco"
DEFAULT_OUTPUT_DIR = "/home/zhouyang/work/gitops/eomt/results/random_train_vis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机可视化 COCO 训练集样本")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="数据集根目录")
    parser.add_argument(
        "--annotations",
        default="annotations/instances_train.json",
        help="标注 json，相对 data-root 或绝对路径",
    )
    parser.add_argument(
        "--image-dir",
        default="train",
        help="图片目录，相对 data-root 或绝对路径",
    )
    parser.add_argument("--num-samples", type=int, default=16, help="随机抽样数量")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--alpha", type=float, default=0.35, help="mask 透明度")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录")
    return parser.parse_args()


def resolve_path(root: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return root / path


def color_for_category(category_id: int) -> tuple[int, int, int]:
    palette = [
        "#e53935",
        "#1e88e5",
        "#43a047",
        "#fb8c00",
        "#8e24aa",
        "#00897b",
        "#6d4c41",
        "#fdd835",
        "#3949ab",
        "#d81b60",
    ]
    return ImageColor.getrgb(palette[(category_id - 1) % len(palette)])


def polygon_mask(segmentation: list[list[float]], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        draw.polygon(points, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def bbox_mask(bbox: list[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(0, min(int(round(x)), width))
    y1 = max(0, min(int(round(y)), height))
    x2 = max(x1, min(int(round(x + w)), width))
    y2 = max(y1, min(int(round(y + h)), height))
    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def segmentation_mask(segmentation, width: int, height: int) -> np.ndarray:
    if segmentation is None or segmentation == []:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, list):
        if segmentation and isinstance(segmentation[0], dict):
            if coco_mask is None:
                raise RuntimeError("需要 pycocotools 才能解析 RLE segmentation")
            merged = coco_mask.merge(segmentation)
            return coco_mask.decode(merged).astype(bool)
        return polygon_mask(segmentation, width, height)
    if isinstance(segmentation, dict):
        if coco_mask is None:
            raise RuntimeError("需要 pycocotools 才能解析 RLE segmentation")
        return coco_mask.decode(segmentation).astype(bool)
    return np.zeros((height, width), dtype=bool)


def annotation_mask(annotation: dict, width: int, height: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    mask = segmentation_mask(segmentation, width, height)
    if mask.any():
        return mask
    bbox = annotation.get("bbox")
    if bbox and len(bbox) == 4:
        return bbox_mask(bbox, width, height)
    return np.zeros((height, width), dtype=bool)


def draw_instances(
    image: Image.Image,
    annotations: list[dict],
    category_name_by_id: dict[int, str],
    alpha: float,
) -> Image.Image:
    image_np = np.array(image).astype(np.float32)
    overlay_np = image_np.copy()
    draw = ImageDraw.Draw(image)

    for annotation in annotations:
        category_id = int(annotation["category_id"])
        color = np.array(color_for_category(category_id), dtype=np.float32)
        mask = annotation_mask(annotation, image.width, image.height)
        if mask.any():
            overlay_np[mask] = overlay_np[mask] * (1.0 - alpha) + color * alpha

    composed = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(composed)

    for annotation in annotations:
        category_id = int(annotation["category_id"])
        category_name = category_name_by_id.get(category_id, str(category_id))
        color = color_for_category(category_id)
        bbox = annotation.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = [float(v) for v in bbox]
        x1, y1, x2, y2 = x, y, x + w, y + h
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

        label = category_name
        text_bbox = draw.textbbox((x1, y1), label)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_y1 = max(0, int(y1) - text_h - 6)
        text_y2 = text_y1 + text_h + 6
        draw.rectangle((x1, text_y1, x1 + text_w + 8, text_y2), fill=color)
        draw.text((x1 + 4, text_y1 + 3), label, fill=(255, 255, 255))

    return composed


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data_root = Path(args.data_root)
    annotations_path = resolve_path(data_root, args.annotations)
    image_dir = resolve_path(data_root, args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    category_name_by_id = {
        int(category["id"]): str(category["name"]) for category in categories
    }
    annotations_by_image_id: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image_id[int(annotation["image_id"])].append(annotation)

    valid_images = [
        image for image in images if annotations_by_image_id.get(int(image["id"]))
    ]
    if not valid_images:
        raise RuntimeError(f"没有找到带标注的样本: {annotations_path}")

    sample_count = min(args.num_samples, len(valid_images))
    sampled_images = rng.sample(valid_images, sample_count)

    saved = []
    for index, image_info in enumerate(sampled_images, start=1):
        image_id = int(image_info["id"])
        image_path = image_dir / str(image_info["file_name"])
        if not image_path.exists():
            alt_path = image_dir / Path(str(image_info["file_name"])).name
            if alt_path.exists():
                image_path = alt_path
            else:
                continue

        image = Image.open(image_path).convert("RGB")
        vis = draw_instances(
            image=image,
            annotations=annotations_by_image_id[image_id],
            category_name_by_id=category_name_by_id,
            alpha=args.alpha,
        )
        out_name = f"{index:03d}_{Path(str(image_info['file_name'])).stem}_vis.jpg"
        out_path = output_dir / out_name
        vis.save(out_path, quality=95)
        saved.append(
            {
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "output_path": str(out_path),
                "num_annotations": len(annotations_by_image_id[image_id]),
            }
        )

    summary_path = output_dir / "summary.json"
    summary = {
        "data_root": str(data_root),
        "annotations_path": str(annotations_path),
        "image_dir": str(image_dir),
        "num_samples_requested": int(args.num_samples),
        "num_samples_saved": len(saved),
        "seed": int(args.seed),
        "samples": saved,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
