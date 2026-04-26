#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
IMAGE_ID_STRIDE = 100000
DEFAULT_INPUT_JSON = Path(
    "/mnt/data2/datasets/instruments_2.0/ytvis_2022_instruments/train.json"
)
DEFAULT_IMAGES_ROOT = Path(
    "/mnt/data2/datasets/instruments_2.0/ytvis_2022_instruments/train/JPEGImages"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/data2/datasets/instruments_2.0/rfdetr_instruments_coco_softlink"
)
DEFAULT_VALID_RATIO = 0.05
DEFAULT_SEED = 42
DEFAULT_WORKERS = 32


@dataclass(frozen=True)
class VideoRecord:
    video_id: int
    width: int
    height: int
    file_names: list[str]
    split_dir_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 ytvis instruments 的 train.json 同步到 RF-DETR 使用的 COCO 软链接目录，"
            "保留已有 train/valid 划分，并为新增视频做稳定分配。"
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=f"YTVis 标注文件，默认: {DEFAULT_INPUT_JSON}",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help=f"原始图片根目录，默认: {DEFAULT_IMAGES_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"软链接 COCO 输出目录，默认: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=DEFAULT_VALID_RATIO,
        help=f"新增视频划入 valid 的比例，默认: {DEFAULT_VALID_RATIO}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"新增视频稳定划分的种子，默认: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"创建软链接的线程数，默认: {DEFAULT_WORKERS}",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="不删除输出目录中源数据已不存在的旧视频目录和多余图片软链接。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅统计，不实际写入文件。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp_path.replace(path)


def stable_valid_assignment(split_dir_name: str, seed: int, valid_ratio: float) -> bool:
    key = f"{seed}:{split_dir_name}".encode("utf-8")
    digest = hashlib.md5(key).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < valid_ratio


def build_video_records(videos: list[dict[str, Any]]) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    seen_split_dir_names: set[str] = set()
    for video in sorted(videos, key=lambda item: int(item["id"])):
        file_names = video["file_names"]
        if not file_names:
            raise ValueError(f"video_id={video['id']} 没有 file_names")
        split_dir_name = PurePosixPath(file_names[0]).parts[0]
        if split_dir_name in seen_split_dir_names:
            raise ValueError(f"存在重复视频目录名: {split_dir_name}")
        seen_split_dir_names.add(split_dir_name)
        records.append(
            VideoRecord(
                video_id=int(video["id"]),
                width=int(video["width"]),
                height=int(video["height"]),
                file_names=[str(name) for name in file_names],
                split_dir_name=split_dir_name,
            )
        )
    return records


def collect_existing_split_map(output_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    duplicates: list[str] = []
    for split in (TRAIN_SPLIT, VALID_SPLIT):
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name in split_map and split_map[name] != split:
                duplicates.append(name)
                continue
            split_map[name] = split
    if duplicates:
        dup_msg = ", ".join(sorted(duplicates))
        raise ValueError(f"以下视频目录同时存在于 train 和 valid: {dup_msg}")
    return split_map


def assign_splits(
    records: list[VideoRecord],
    existing_split_map: dict[str, str],
    seed: int,
    valid_ratio: float,
) -> tuple[dict[str, str], dict[str, int]]:
    split_map: dict[str, str] = {}
    stats = {
        "existing_train": 0,
        "existing_valid": 0,
        "new_train": 0,
        "new_valid": 0,
    }
    for record in records:
        split = existing_split_map.get(record.split_dir_name)
        if split is not None:
            split_map[record.split_dir_name] = split
            stats[f"existing_{split}"] += 1
            continue
        split = (
            VALID_SPLIT
            if stable_valid_assignment(record.split_dir_name, seed, valid_ratio)
            else TRAIN_SPLIT
        )
        split_map[record.split_dir_name] = split
        stats[f"new_{split}"] += 1
    return split_map, stats


def remove_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


def sync_symlink(src: Path, dst: Path, dry_run: bool) -> str:
    if not src.exists():
        return "missing_source"

    if dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return "unchanged"
        except FileNotFoundError:
            pass
        if not dry_run:
            dst.unlink()
    elif dst.exists():
        if dst.is_dir():
            return "conflict_dir"
        if not dry_run:
            dst.unlink()

    if not dry_run:
        os.symlink(src, dst)
    return "created"


def prune_stale_outputs(
    output_dir: Path,
    expected_split_map: dict[str, str],
    videos_by_name: dict[str, VideoRecord],
    dry_run: bool,
) -> Counter:
    stats: Counter = Counter()
    for split in (TRAIN_SPLIT, VALID_SPLIT):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for child in split_dir.iterdir():
            if not child.is_dir():
                continue
            expected_split = expected_split_map.get(child.name)
            if expected_split != split:
                remove_path(child, dry_run=dry_run)
                stats["removed_video_dirs"] += 1
                continue

            record = videos_by_name[child.name]
            expected_files = {PurePosixPath(name).name for name in record.file_names}
            for frame_path in child.iterdir():
                if frame_path.name in expected_files:
                    continue
                remove_path(frame_path, dry_run=dry_run)
                stats["removed_stale_links"] += 1
    return stats


def iter_link_tasks(
    output_dir: Path,
    images_root: Path,
    records: list[VideoRecord],
    split_map: dict[str, str],
) -> Any:
    for record in records:
        split = split_map[record.split_dir_name]
        video_dir = output_dir / split / record.split_dir_name
        video_dir.mkdir(parents=True, exist_ok=True)
        for rel_file in record.file_names:
            yield images_root / rel_file, output_dir / split / rel_file


def build_annotations_index(
    annotations: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        indexed[int(ann["video_id"])].append(ann)
    for video_id in indexed:
        indexed[video_id].sort(key=lambda ann: int(ann["id"]))
    return indexed


def normalize_bbox(bbox: Any) -> list[float] | None:
    if not bbox or len(bbox) != 4:
        return None
    return [float(value) for value in bbox]


def build_coco_split(
    records: list[VideoRecord],
    split_map: dict[str, str],
    annotations_by_video: dict[int, list[dict[str, Any]]],
    categories: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for record in records:
        if split_map[record.split_dir_name] != split:
            continue

        for frame_idx, rel_file in enumerate(record.file_names):
            images.append(
                {
                    "id": record.video_id * IMAGE_ID_STRIDE + frame_idx,
                    "file_name": rel_file,
                    "width": record.width,
                    "height": record.height,
                }
            )

        for ann in annotations_by_video.get(record.video_id, []):
            segmentations = ann.get("segmentations", [])
            areas = ann.get("areas", [])
            bboxes = ann.get("bboxes", [])
            max_len = min(len(record.file_names), len(segmentations))
            for frame_idx in range(max_len):
                segmentation = segmentations[frame_idx]
                if not segmentation:
                    continue
                bbox = normalize_bbox(bboxes[frame_idx] if frame_idx < len(bboxes) else None)
                area = areas[frame_idx] if frame_idx < len(areas) else None
                if bbox is None or area is None or float(area) <= 0:
                    continue
                annotations.append(
                    {
                        "id": int(ann["id"]) * IMAGE_ID_STRIDE + frame_idx,
                        "image_id": record.video_id * IMAGE_ID_STRIDE + frame_idx,
                        "category_id": int(ann["category_id"]),
                        "segmentation": segmentation,
                        "area": float(area),
                        "bbox": bbox,
                        "iscrowd": int(ann.get("iscrowd", 0)),
                    }
                )

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def main() -> None:
    args = parse_args()
    prune = not args.no_prune

    source = load_json(args.input_json)
    categories = sorted(source["categories"], key=lambda item: int(item["id"]))
    records = build_video_records(source["videos"])
    videos_by_name = {record.split_dir_name: record for record in records}
    annotations_by_video = build_annotations_index(source["annotations"])

    output_dir = args.output_dir
    existing_split_map = collect_existing_split_map(output_dir)
    split_map, split_stats = assign_splits(
        records=records,
        existing_split_map=existing_split_map,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
    )

    prune_stats = Counter()
    if prune:
        prune_stats = prune_stale_outputs(
            output_dir=output_dir,
            expected_split_map=split_map,
            videos_by_name=videos_by_name,
            dry_run=args.dry_run,
        )

    total_links = sum(len(record.file_names) for record in records)
    link_tasks = iter_link_tasks(
        output_dir=output_dir,
        images_root=args.images_root,
        records=records,
        split_map=split_map,
    )

    link_stats: Counter = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for status in executor.map(
            lambda item: sync_symlink(item[0], item[1], dry_run=args.dry_run),
            link_tasks,
        ):
            link_stats[status] += 1

    if link_stats.get("missing_source", 0):
        raise FileNotFoundError(
            f"发现 {link_stats['missing_source']} 个源图片不存在，请先检查 {args.images_root}"
        )
    if link_stats.get("conflict_dir", 0):
        raise RuntimeError(
            f"发现 {link_stats['conflict_dir']} 个目标路径被目录占用，请手动检查输出目录"
        )

    train_coco = build_coco_split(
        records=records,
        split_map=split_map,
        annotations_by_video=annotations_by_video,
        categories=categories,
        split=TRAIN_SPLIT,
    )
    valid_coco = build_coco_split(
        records=records,
        split_map=split_map,
        annotations_by_video=annotations_by_video,
        categories=categories,
        split=VALID_SPLIT,
    )

    summary = {
        "input_json": str(args.input_json),
        "images_root": str(args.images_root),
        "output_dir": str(output_dir),
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "link_workers": args.workers,
        "num_categories": len(categories),
        "train_videos": sum(1 for record in records if split_map[record.split_dir_name] == TRAIN_SPLIT),
        "valid_videos": sum(1 for record in records if split_map[record.split_dir_name] == VALID_SPLIT),
        "train_images": len(train_coco["images"]),
        "valid_images": len(valid_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "valid_annotations": len(valid_coco["annotations"]),
        "sync_mode": "incremental_keep_existing_split",
        "last_sync_time": datetime.now().isoformat(timespec="seconds"),
    }

    sync_summary = {
        "source_videos": len(records),
        "source_annotations": len(source["annotations"]),
        "existing_output_videos": len(existing_split_map),
        "added_videos": split_stats["new_train"] + split_stats["new_valid"],
        "added_train_videos": split_stats["new_train"],
        "added_valid_videos": split_stats["new_valid"],
        "kept_train_videos": split_stats["existing_train"],
        "kept_valid_videos": split_stats["existing_valid"],
        "pruned_video_dirs": prune_stats.get("removed_video_dirs", 0),
        "pruned_stale_links": prune_stats.get("removed_stale_links", 0),
        "total_link_tasks": total_links,
        "created_links": link_stats.get("created", 0),
        "unchanged_links": link_stats.get("unchanged", 0),
        "prune_enabled": prune,
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        write_json_atomic(output_dir / TRAIN_SPLIT / "_annotations.coco.json", train_coco)
        write_json_atomic(output_dir / VALID_SPLIT / "_annotations.coco.json", valid_coco)
        write_json_atomic(output_dir / "conversion_summary.json", summary)
        write_json_atomic(output_dir / "sync_summary.json", sync_summary)

    print("同步完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(sync_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
