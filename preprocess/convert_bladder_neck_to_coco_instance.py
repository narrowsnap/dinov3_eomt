#!/usr/bin/env python3

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Iterable

from PIL import Image


DEFAULT_CATEGORIES = ["prostate", "cutting_area", "bladder"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 bladder_neck 的归一化多边形标注转换为 COCO instance 格式，并生成 EoMT 可直接使用的 zip 数据。"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="原始图片目录，例如 /mnt/data2/datasets/bladder_neck/images",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        required=True,
        help="训练集标注 json 路径。",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        required=True,
        help="验证集标注 json 路径。可以直接传 test_name2anno_seg.json。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录，脚本会在该目录下写入 COCO json 与 zip 文件。",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="类别顺序，默认: prostate cutting_area bladder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖。",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, dict[str, list[list[float]]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"标注文件不是字典结构: {path}")
    return data


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def check_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            f"以下输出文件已存在，请删除后重试，或加上 --overwrite:\n{names}"
        )


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def normalized_polygon_to_pixels(
    polygon: list[list[float]], width: int, height: int
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in polygon:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"非法点格式: {item}")
        x = min(max(float(item[0]), 0.0), 1.0) * width
        y = min(max(float(item[1]), 0.0), 1.0) * height
        points.append((x, y))
    return points


def flatten_polygon(points: list[tuple[float, float]]) -> list[float]:
    flattened: list[float] = []
    for x, y in points:
        flattened.extend([round(x, 4), round(y, 4)])
    return flattened


def build_coco_split(
    split_annotations: dict[str, dict[str, list[list[float]]]],
    images_dir: Path,
    categories: list[str],
) -> dict:
    category_to_id = {name: idx + 1 for idx, name in enumerate(categories)}
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx + 1, "name": name, "supercategory": "organ"}
            for idx, name in enumerate(categories)
        ],
    }

    annotation_id = 1

    for image_id, file_name in enumerate(sorted(split_annotations), start=1):
        image_path = images_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"找不到图片: {image_path}")

        with Image.open(image_path) as image:
            width, height = image.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        anns = split_annotations[file_name]
        if not isinstance(anns, dict):
            raise ValueError(f"图片 {file_name} 的标注不是字典结构。")

        for category_name in categories:
            polygon = anns.get(category_name)
            if polygon is None:
                continue
            if len(polygon) < 3:
                continue

            pixel_points = normalized_polygon_to_pixels(polygon, width, height)
            area = polygon_area(pixel_points)
            if area <= 0:
                continue

            xs = [point[0] for point in pixel_points]
            ys = [point[1] for point in pixel_points]
            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[category_name],
                    "segmentation": [flatten_polygon(pixel_points)],
                    "area": round(area, 4),
                    "bbox": [
                        round(x_min, 4),
                        round(y_min, 4),
                        round(x_max - x_min, 4),
                        round(y_max - y_min, 4),
                    ],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return coco


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def write_images_zip(
    zip_path: Path,
    split_name: str,
    split_annotations: dict[str, dict[str, list[list[float]]]],
    images_dir: Path,
) -> None:
    ensure_parent(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for file_name in sorted(split_annotations):
            image_path = images_dir / file_name
            zf.write(image_path, arcname=f"{split_name}/{file_name}")


def write_annotations_zip(zip_path: Path, train_json: Path, val_json: Path) -> None:
    ensure_parent(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(train_json, arcname="annotations/instances_train2017.json")
        zf.write(val_json, arcname="annotations/instances_val2017.json")


def main() -> None:
    args = parse_args()

    train_annotations = load_annotations(args.train_json)
    val_annotations = load_annotations(args.val_json)

    output_dir = args.output_dir
    annotations_dir = output_dir / "annotations"
    train_json_out = annotations_dir / "instances_train2017.json"
    val_json_out = annotations_dir / "instances_val2017.json"
    train_zip_out = output_dir / "train2017.zip"
    val_zip_out = output_dir / "val2017.zip"
    ann_zip_out = output_dir / "annotations_trainval2017.zip"

    check_output_paths(
        [train_json_out, val_json_out, train_zip_out, val_zip_out, ann_zip_out],
        overwrite=args.overwrite,
    )

    train_coco = build_coco_split(train_annotations, args.images_dir, args.categories)
    val_coco = build_coco_split(val_annotations, args.images_dir, args.categories)

    write_json(train_json_out, train_coco)
    write_json(val_json_out, val_coco)
    write_images_zip(train_zip_out, "train2017", train_annotations, args.images_dir)
    write_images_zip(val_zip_out, "val2017", val_annotations, args.images_dir)
    write_annotations_zip(ann_zip_out, train_json_out, val_json_out)

    print("转换完成")
    print(f"train images: {len(train_coco['images'])}")
    print(f"train annotations: {len(train_coco['annotations'])}")
    print(f"val images: {len(val_coco['images'])}")
    print(f"val annotations: {len(val_coco['annotations'])}")
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
