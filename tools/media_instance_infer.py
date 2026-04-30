#!/usr/bin/env python3

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
import importlib
import json
from collections import OrderedDict
import os
from pathlib import Path
from queue import Queue
import site
import threading
import sys
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_FONT_PATH = "/home/zhouyang/simhei.ttf"
FONT_CANDIDATES = [
    DEFAULT_FONT_PATH,
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

INSTRUMENT_CATEGORY_NAME_TO_COLOR = {
    "切割闭合器": (230, 25, 75),
    "剪刀": (60, 180, 75),
    "取石器": (0, 130, 200),
    "吸引器": (245, 130, 48),
    "引流管": (145, 30, 180),
    "戳卡": (70, 240, 240),
    "手持钳": (240, 50, 230),
    "持针器": (255, 0, 0),
    "施夹器": (250, 190, 190),
    "术中超声": (0, 128, 128),
    "机器人器械": (230, 190, 255),
    "标本袋": (170, 110, 40),
    "止血绒": (255, 250, 200),
    "狗头夹": (128, 0, 0),
    "电凝": (170, 255, 195),
    "电钩": (128, 128, 0),
    "电铲": (255, 215, 180),
    "百克钳": (0, 0, 128),
    "纱布": (128, 128, 128),
    "结扎速": (220, 20, 60),
    "缝针": (46, 139, 87),
    "肝门阻断带": (65, 105, 225),
    "胆道镜": (255, 140, 0),
    "补片": (106, 90, 205),
    "超声刀": (47, 79, 79),
    "消融针": (0, 170, 255),
    "unknown": (160, 160, 160),
}

DEFAULT_INSTRUMENT_CLASS_NAMES = [
    "切割闭合器",
    "剪刀",
    "取石器",
    "吸引器",
    "引流管",
    "戳卡",
    "手持钳",
    "持针器",
    "施夹器",
    "术中超声",
    "机器人器械",
    "标本袋",
    "止血绒",
    "狗头夹",
    "电凝",
    "电钩",
    "电铲",
    "纱布",
    "结扎速",
    "缝针",
    "肝门阻断带",
    "胆道镜",
    "补片",
    "超声刀",
    "消融针",
]

_VIDEO_READER_CLASS = None
VIDEO_SAME_CLASS_DEDUP_MASK_IOU_THRESH = 0.70
VIDEO_DEDUP_MASK_IOU_THRESH = 0.90
VIDEO_DEDUP_CONTAIN_THRESH = 0.95
VIDEO_TRACK_MAX_CENTER_DIST = 80.0
VIDEO_TRACK_MIN_BBOX_IOU = 0.05
VIDEO_TRACK_MAX_MISS = 5
VIDEO_TRACK_LABEL_SWITCH_FRAMES = 3
VIDEO_TRACK_VELOCITY_MOMENTUM = 0.7


def parse_args():
    parser = argparse.ArgumentParser(description="实例分割推理：视频 / 单图 / 批量图片")
    parser.add_argument("--config", required=True, help="训练配置文件路径")
    parser.add_argument("--ckpt", required=True, help="训练 checkpoint 路径")
    parser.add_argument(
        "--mode",
        choices=["video", "videos", "image", "images"],
        required=True,
        help="推理模式",
    )
    parser.add_argument("--input-video", help="输入视频路径")
    parser.add_argument("--input-video-dir", help="输入视频目录")
    parser.add_argument("--input-video-list", help="输入视频列表文件")
    parser.add_argument("--input-image", help="输入图片路径")
    parser.add_argument("--input-dir", help="输入图片目录")
    parser.add_argument("--output-path", help="输出图片或视频路径")
    parser.add_argument("--output-dir", help="多视频模式输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="分数阈值")
    parser.add_argument("--top-k", type=int, default=64, help="每张图保留的 query 数")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask 叠加透明度")
    parser.add_argument("--font-path", default=DEFAULT_FONT_PATH, help="中文字体路径")
    parser.add_argument("--font-size", type=int, default=22, help="字体大小")
    parser.add_argument("--batch-size", type=int, default=4, help="批量图片推理 batch size")
    parser.add_argument("--io-workers", type=int, default=4, help="异步加载线程数")
    parser.add_argument("--video-workers", type=int, default=2, help="多视频并发数量")
    parser.add_argument("--prefetch-batches", type=int, default=3, help="异步预取批次数")
    parser.add_argument("--image-cache-size", type=int, default=256, help="图片解码 LRU 缓存大小")
    parser.add_argument(
        "--video-same-class-dedup-mask-iou-thresh",
        type=float,
        default=VIDEO_SAME_CLASS_DEDUP_MASK_IOU_THRESH,
        help="视频后处理：同类别同帧 mask 去重 IoU 阈值",
    )
    parser.add_argument(
        "--video-dedup-mask-iou-thresh",
        type=float,
        default=VIDEO_DEDUP_MASK_IOU_THRESH,
        help="视频后处理：同帧 mask 去重 IoU 阈值",
    )
    parser.add_argument(
        "--video-dedup-contain-thresh",
        type=float,
        default=VIDEO_DEDUP_CONTAIN_THRESH,
        help="视频后处理：同帧 mask 包含率阈值",
    )
    parser.add_argument(
        "--video-track-max-center-dist",
        type=float,
        default=VIDEO_TRACK_MAX_CENTER_DIST,
        help="视频后处理：跨帧轨迹最大中心距离",
    )
    parser.add_argument(
        "--video-track-min-bbox-iou",
        type=float,
        default=VIDEO_TRACK_MIN_BBOX_IOU,
        help="视频后处理：跨帧轨迹最小 bbox IoU",
    )
    parser.add_argument(
        "--video-track-max-miss",
        type=int,
        default=VIDEO_TRACK_MAX_MISS,
        help="视频后处理：轨迹允许的最大丢失帧数",
    )
    parser.add_argument(
        "--video-track-label-switch-frames",
        type=int,
        default=VIDEO_TRACK_LABEL_SWITCH_FRAMES,
        help="视频后处理：类别切换所需连续帧数",
    )
    parser.add_argument(
        "--image-suffixes",
        nargs="*",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="批量图片模式的后缀过滤",
    )
    parser.add_argument(
        "--video-suffixes",
        nargs="*",
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        help="多视频模式的视频后缀过滤",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate(config: dict[str, Any], **extra_kwargs):
    cls = import_class(config["class_path"])
    init_args = dict(config.get("init_args", {}))
    init_args.update(extra_kwargs)
    return cls(**init_args)


def load_class_names(config: dict[str, Any]) -> list[str]:
    data_cfg = config.get("data", {})
    data_path = data_cfg.get("init_args", {}).get("path")
    if data_path:
        for candidate in [
            Path(data_path) / "train" / "_annotations.coco.json",
            Path(data_path) / "annotations" / "instances_train2017.json",
        ]:
            if candidate.exists():
                data = json.loads(candidate.read_text(encoding="utf-8"))
                categories = sorted(data["categories"], key=lambda item: item["id"])
                return [item["name"] for item in categories]
    num_classes = int(data_cfg.get("init_args", {}).get("num_classes", 0))
    if num_classes == len(DEFAULT_INSTRUMENT_CLASS_NAMES):
        return DEFAULT_INSTRUMENT_CLASS_NAMES
    return [f"class_{i}" for i in range(num_classes)]


def build_model(config: dict[str, Any], checkpoint_path: str, device: torch.device):
    data_cfg = config["data"]
    data_module = instantiate(data_cfg)
    img_size = tuple(data_module.img_size)
    num_classes = data_module.num_classes

    model_cfg = config["model"]
    model_init_args = dict(model_cfg["init_args"])
    network_cfg = model_init_args.pop("network")
    encoder_cfg = network_cfg["init_args"].pop("encoder")

    encoder = instantiate(encoder_cfg, img_size=img_size)
    network = instantiate(network_cfg, encoder=encoder, num_classes=num_classes)
    model = instantiate(
        model_cfg,
        network=network,
        img_size=img_size,
        num_classes=num_classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "加载训练 checkpoint 失败: "
            f"missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}"
        )

    model.eval()
    model.to(device)
    return model


def choose_font(font_path: str | None, size: int):
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(FONT_CANDIDATES)
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def get_category_color(category_name: str) -> tuple[int, int, int]:
    return INSTRUMENT_CATEGORY_NAME_TO_COLOR.get(
        category_name, INSTRUMENT_CATEGORY_NAME_TO_COLOR["unknown"]
    )


def class_id_to_name(class_id: int, category_names: Sequence[str]) -> str:
    if 0 <= class_id < len(category_names):
        return category_names[class_id]
    return "unknown"


def object_array(items: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    arr[:] = list(items)
    return arr


def build_output_npz_path(output_video_path: Path) -> Path:
    return output_video_path.with_suffix(".npz")


def build_output_video_path(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}_rendered.mp4"


def build_output_npz_path_for_video(video_path: Path, npz_dir: Path) -> Path:
    return npz_dir / f"{video_path.stem}_rendered.npz"


def should_skip_video_inference(output_video_path: Path, output_npz_path: Path) -> bool:
    return output_video_path.exists() and output_npz_path.exists()


def save_dense_outputs_to_npz(
    dense_outputs_per_frame: Sequence[dict[str, np.ndarray]],
    output_path: Path,
    category_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    category_names_per_frame = [
        np.asarray(
            [class_id_to_name(int(class_id), category_names) for class_id in outputs["class_ids"]],
            dtype=object,
        )
        for outputs in dense_outputs_per_frame
    ]
    np.savez_compressed(
        output_path,
        format_version=np.asarray("videomt_inference_v1"),
        obj_ids_per_frame=object_array([outputs["obj_ids"] for outputs in dense_outputs_per_frame]),
        binary_masks_per_frame=object_array(
            [outputs["binary_masks"] for outputs in dense_outputs_per_frame]
        ),
        class_ids_per_frame=object_array(
            [outputs["class_ids"] for outputs in dense_outputs_per_frame]
        ),
        scores_per_frame=object_array([outputs["scores"] for outputs in dense_outputs_per_frame]),
        reid_embeds_per_frame=object_array(
            [
                outputs.get("reid_embeds", np.zeros((len(outputs["obj_ids"]), 0), dtype=np.float32))
                for outputs in dense_outputs_per_frame
            ]
        ),
        motion_vectors_per_frame=object_array(
            [
                outputs.get("motion_vectors", np.zeros((len(outputs["obj_ids"]), 2), dtype=np.float32))
                for outputs in dense_outputs_per_frame
            ]
        ),
        category_names_per_frame=object_array(category_names_per_frame),
        category_names=np.asarray(list(category_names), dtype=object),
    )


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def mask_center(mask: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        if box is None:
            return np.zeros(2, dtype=np.float32)
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return inter / union


def mask_containment(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    area_a = mask_area(mask_a)
    area_b = mask_area(mask_b)
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    inter = int(np.logical_and(mask_a, mask_b).sum())
    return inter / smaller


def box_iou(
    box_a: tuple[int, int, int, int] | None,
    box_b: tuple[int, int, int, int] | None,
) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class VideoDetection:
    mask: np.ndarray
    label: int
    score: float
    box: tuple[int, int, int, int] | None
    center: np.ndarray
    area: int
    track_id: int = -1


@dataclass
class VideoTrack:
    track_id: int
    stable_label: int
    center: np.ndarray
    velocity: np.ndarray
    last_box: tuple[int, int, int, int] | None
    last_mask: np.ndarray
    last_score: float
    miss_count: int = 0
    pending_label: int | None = None
    pending_count: int = 0


class VideoPostProcessor:
    def __init__(
        self,
        same_class_dedup_mask_iou_thresh: float = VIDEO_SAME_CLASS_DEDUP_MASK_IOU_THRESH,
        dedup_mask_iou_thresh: float = VIDEO_DEDUP_MASK_IOU_THRESH,
        dedup_contain_thresh: float = VIDEO_DEDUP_CONTAIN_THRESH,
        max_center_dist: float = VIDEO_TRACK_MAX_CENTER_DIST,
        min_bbox_iou: float = VIDEO_TRACK_MIN_BBOX_IOU,
        max_miss: int = VIDEO_TRACK_MAX_MISS,
        label_switch_frames: int = VIDEO_TRACK_LABEL_SWITCH_FRAMES,
        velocity_momentum: float = VIDEO_TRACK_VELOCITY_MOMENTUM,
    ):
        self.same_class_dedup_mask_iou_thresh = same_class_dedup_mask_iou_thresh
        self.dedup_mask_iou_thresh = dedup_mask_iou_thresh
        self.dedup_contain_thresh = dedup_contain_thresh
        self.max_center_dist = max_center_dist
        self.min_bbox_iou = min_bbox_iou
        self.max_miss = max_miss
        self.label_switch_frames = label_switch_frames
        self.velocity_momentum = velocity_momentum
        self.tracks: list[VideoTrack] = []
        self.next_track_id = 0

    def update(
        self,
        masks: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        detections = self._build_detections(masks, labels, scores)
        detections = self._deduplicate_detections(detections)

        if not detections:
            self._step_unmatched_tracks(set())
            return (
                np.zeros_like(masks),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
            )

        matches, unmatched_track_ids, unmatched_detection_ids = self._match_detections(detections)
        final_detections: list[VideoDetection] = []

        matched_track_ids = {track_idx for track_idx, _ in matches}
        self._step_unmatched_tracks(unmatched_track_ids)

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            self._update_track(track, detection)
            final_detections.append(
                VideoDetection(
                    mask=detection.mask,
                    label=track.stable_label,
                    score=detection.score,
                    track_id=track.track_id,
                    box=detection.box,
                    center=detection.center,
                    area=detection.area,
                )
            )

        for det_idx in unmatched_detection_ids:
            detection = detections[det_idx]
            track = self._create_track(detection)
            matched_track_ids.add(track.track_id)
            final_detections.append(
                VideoDetection(
                    mask=detection.mask,
                    label=track.stable_label,
                    score=detection.score,
                    track_id=track.track_id,
                    box=detection.box,
                    center=detection.center,
                    area=detection.area,
                )
            )

        self.tracks = [track for track in self.tracks if track.miss_count <= self.max_miss]
        final_detections.sort(key=lambda det: det.score, reverse=True)
        return self._detections_to_arrays(final_detections, masks.shape)

    def _build_detections(
        self,
        masks: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> list[VideoDetection]:
        detections: list[VideoDetection] = []
        for mask, label, score in zip(masks, labels, scores):
            box = mask_to_box(mask)
            if box is None:
                continue
            detections.append(
                VideoDetection(
                    mask=mask,
                    label=int(label),
                    score=float(score),
                    box=box,
                    center=mask_center(mask, box),
                    area=mask_area(mask),
                )
            )
        detections.sort(key=lambda det: det.score, reverse=True)
        return detections

    def _deduplicate_detections(self, detections: list[VideoDetection]) -> list[VideoDetection]:
        kept: list[VideoDetection] = []
        for detection in detections:
            is_duplicate = False
            for existing in kept:
                mask_iou_value = mask_iou(detection.mask, existing.mask)
                iou_thresh = (
                    self.same_class_dedup_mask_iou_thresh
                    if detection.label == existing.label
                    else self.dedup_mask_iou_thresh
                )
                if mask_iou_value >= iou_thresh:
                    is_duplicate = True
                    break
                if (
                    mask_containment(detection.mask, existing.mask)
                    >= self.dedup_contain_thresh
                ):
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(detection)
        return kept

    def _match_detections(
        self,
        detections: list[VideoDetection],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not self.tracks:
            return [], set(), set(range(len(detections)))

        candidates: list[tuple[float, int, int]] = []
        for track_idx, track in enumerate(self.tracks):
            predicted_center = track.center + track.velocity
            for det_idx, detection in enumerate(detections):
                center_dist = float(np.linalg.norm(detection.center - predicted_center))
                if center_dist > self.max_center_dist:
                    continue
                bbox_overlap = box_iou(track.last_box, detection.box)
                if (
                    bbox_overlap < self.min_bbox_iou
                    and center_dist > self.max_center_dist * 0.5
                ):
                    continue
                label_penalty = 5.0 if detection.label != track.stable_label else 0.0
                cost = center_dist + 20.0 * (1.0 - bbox_overlap) + label_penalty
                candidates.append((cost, track_idx, det_idx))

        candidates.sort(key=lambda item: item[0])
        matched_track_ids: set[int] = set()
        matched_detection_ids: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _, track_idx, det_idx in candidates:
            if track_idx in matched_track_ids or det_idx in matched_detection_ids:
                continue
            matches.append((track_idx, det_idx))
            matched_track_ids.add(track_idx)
            matched_detection_ids.add(det_idx)

        unmatched_track_ids = set(range(len(self.tracks))) - matched_track_ids
        unmatched_detection_ids = set(range(len(detections))) - matched_detection_ids
        return matches, unmatched_track_ids, unmatched_detection_ids

    def _step_unmatched_tracks(self, unmatched_track_ids: set[int]) -> None:
        for track_idx in unmatched_track_ids:
            track = self.tracks[track_idx]
            track.center = track.center + track.velocity
            track.velocity = track.velocity * self.velocity_momentum
            track.miss_count += 1

    def _update_track(self, track: VideoTrack, detection: VideoDetection) -> None:
        observed_velocity = detection.center - track.center
        track.velocity = (
            track.velocity * self.velocity_momentum
            + observed_velocity * (1.0 - self.velocity_momentum)
        )
        track.center = detection.center
        track.last_box = detection.box
        track.last_mask = detection.mask
        track.last_score = detection.score
        track.miss_count = 0

        if detection.label == track.stable_label:
            track.pending_label = None
            track.pending_count = 0
            return

        if detection.label == track.pending_label:
            track.pending_count += 1
        else:
            track.pending_label = detection.label
            track.pending_count = 1

        if track.pending_count >= self.label_switch_frames:
            track.stable_label = int(detection.label)
            track.pending_label = None
            track.pending_count = 0

    def _create_track(self, detection: VideoDetection) -> VideoTrack:
        track = VideoTrack(
            track_id=self.next_track_id,
            stable_label=int(detection.label),
            center=detection.center.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            last_box=detection.box,
            last_mask=detection.mask,
            last_score=detection.score,
        )
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def _detections_to_arrays(
        self,
        detections: list[VideoDetection],
        mask_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not detections:
            return (
                np.zeros(mask_shape, dtype=bool),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 2), dtype=np.float32),
            )
        masks = np.stack([det.mask for det in detections], axis=0).astype(bool)
        labels = np.asarray([det.label for det in detections], dtype=np.int64)
        scores = np.asarray([det.score for det in detections], dtype=np.float32)
        obj_ids = np.asarray([det.track_id for det in detections], dtype=np.int64)
        motion_vectors = np.stack(
            [self._track_velocity_for_detection(det.track_id) for det in detections],
            axis=0,
        ).astype(np.float32)
        return masks, labels, scores, obj_ids, motion_vectors

    def _track_velocity_for_detection(self, track_id: int) -> np.ndarray:
        for track in self.tracks:
            if track.track_id == track_id:
                return track.velocity.astype(np.float32)
        return np.zeros(2, dtype=np.float32)


def _candidate_video_reader_lib_dirs() -> list[Path]:
    candidates: list[Path] = []
    search_roots = []
    try:
        search_roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        search_roots.append(site.getusersitepackages())
    except AttributeError:
        pass
    search_roots.extend(sys.path)

    for root in search_roots:
        if not root:
            continue
        path = Path(root)
        for libs_dir_name in ("video_reader_rs.libs", "video_reader.libs"):
            libs_dir = path / libs_dir_name
            if libs_dir.exists():
                candidates.append(libs_dir)

    unique_dirs: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(path)
    return unique_dirs


def _try_preload_video_reader_libs(libs_dir: Path) -> None:
    pending = sorted(libs_dir.glob("*.so*"))
    loaded: set[str] = set()
    progress = True
    while pending and progress:
        progress = False
        next_pending = []
        for lib_path in pending:
            lib_key = str(lib_path)
            if lib_key in loaded:
                continue
            try:
                ctypes.CDLL(lib_key, mode=ctypes.RTLD_GLOBAL)
                loaded.add(lib_key)
                progress = True
            except OSError:
                next_pending.append(lib_path)
        pending = next_pending


def get_video_reader_class():
    global _VIDEO_READER_CLASS
    if _VIDEO_READER_CLASS is not None:
        return _VIDEO_READER_CLASS

    last_error: Exception | None = None
    try:
        from video_reader import PyVideoReader

        _VIDEO_READER_CLASS = PyVideoReader
        return _VIDEO_READER_CLASS
    except Exception as exc:
        last_error = exc

    for libs_dir in _candidate_video_reader_lib_dirs():
        current = os.environ.get("LD_LIBRARY_PATH", "")
        libs_dir_str = str(libs_dir)
        if libs_dir_str not in current.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{libs_dir_str}:{current}" if current else libs_dir_str
            )
        _try_preload_video_reader_libs(libs_dir)
        try:
            from video_reader import PyVideoReader

            _VIDEO_READER_CLASS = PyVideoReader
            return _VIDEO_READER_CLASS
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "无法导入 video_reader-rs，请确认已安装 `video-reader-rs`，"
        "且运行环境可访问 `video_reader_rs.libs` 动态库目录。"
    ) from last_error


@torch.inference_mode()
def predict_batch(
    model,
    frames_rgb: list[np.ndarray],
    top_k: int,
    score_thresh: float,
):
    device = model.device
    imgs = [torch.from_numpy(frame).permute(2, 0, 1).to(device=device) for frame in frames_rgb]
    img_sizes = [img.shape[-2:] for img in imgs]

    transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=torch.float16):
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)

    mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)
    class_logits = class_logits_per_layer[-1]

    outputs = []
    for batch_idx in range(len(frames_rgb)):
        scores = class_logits[batch_idx].float().softmax(dim=-1)[:, :-1]
        labels = (
            torch.arange(scores.shape[-1], device=device)
            .unsqueeze(0)
            .repeat(scores.shape[0], 1)
            .flatten(0, 1)
        )
        flat_scores = scores.flatten(0, 1)
        if flat_scores.numel() == 0:
            outputs.append((np.zeros((0, *img_sizes[batch_idx]), dtype=bool), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)))
            continue

        topk_k = min(top_k, flat_scores.numel())
        topk_scores, topk_indices = flat_scores.topk(topk_k, sorted=False)
        labels = labels[topk_indices]
        query_indices = topk_indices // scores.shape[-1]
        sample_mask_logits = mask_logits[batch_idx][query_indices].float()
        masks = sample_mask_logits > 0
        mask_scores = (
            sample_mask_logits.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
        scores_out = torch.nan_to_num(topk_scores * mask_scores, nan=0.0, posinf=0.0, neginf=0.0)
        keep = scores_out >= score_thresh
        if not keep.any():
            outputs.append((np.zeros((0, *img_sizes[batch_idx]), dtype=bool), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)))
            continue

        masks_np = masks[keep].detach().cpu().numpy().astype(bool)
        labels_np = labels[keep].detach().cpu().numpy()
        scores_np = scores_out[keep].detach().cpu().numpy()
        order = np.argsort(-scores_np)
        outputs.append((masks_np[order], labels_np[order], scores_np[order]))
    return outputs


def render_frame(
    frame_bgr: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
    alpha: float,
    font,
) -> np.ndarray:
    overlay = frame_bgr.copy().astype(np.float32)
    text_items: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []

    for mask, label, score in zip(masks, labels, scores):
        label_int = int(label)
        class_name = class_names[label_int] if 0 <= label_int < len(class_names) else f"class_{label_int}"
        color_rgb = get_category_color(class_name)
        color_bgr = np.array(color_rgb[::-1], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color_bgr * alpha

        box = mask_to_box(mask)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color=tuple(int(v) for v in color_bgr.tolist()),
            thickness=2,
        )
        text_items.append(((x1, max(0, y1 - 28)), class_name, color_rgb))

    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    for (x, y), label_text, color in text_items:
        bbox = draw.textbbox((x, y), label_text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x, y), label_text, fill=(255, 255, 255), font=font)
    return np.array(image)[:, :, ::-1]


def render_side_by_side(original_bgr: np.ndarray, rendered_bgr: np.ndarray) -> np.ndarray:
    if original_bgr.shape != rendered_bgr.shape:
        raise ValueError("原图和渲染图尺寸不一致，无法左右拼接")
    return np.concatenate([original_bgr, rendered_bgr], axis=1)


def read_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def write_image_bgr(path: Path, image_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_bgr[:, :, ::-1]).save(path)


class ImageCache:
    def __init__(self, max_items: int):
        self.max_items = max(0, max_items)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if self.max_items <= 0:
            return read_image_rgb(path)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        image = read_image_rgb(path)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            self._cache[key] = image
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return image


def load_image_batch(paths: list[Path], image_cache: ImageCache | None = None) -> list[np.ndarray]:
    if image_cache is None:
        return [read_image_rgb(path) for path in paths]
    return [image_cache.get(path) for path in paths]


def iter_prefetched_image_batches(
    image_paths: list[Path],
    batch_size: int,
    io_workers: int,
    prefetch_batches: int,
    image_cache: ImageCache | None = None,
):
    batches = [
        image_paths[start : start + batch_size]
        for start in range(0, len(image_paths), batch_size)
    ]
    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as executor:
        futures: list[tuple[list[Path], Future]] = []
        next_batch_idx = 0

        while next_batch_idx < len(batches) and len(futures) < max(1, prefetch_batches):
            batch_paths = batches[next_batch_idx]
            futures.append(
                (batch_paths, executor.submit(load_image_batch, batch_paths, image_cache))
            )
            next_batch_idx += 1

        while futures:
            batch_paths, future = futures.pop(0)
            batch_rgb = future.result()
            yield batch_paths, batch_rgb

            if next_batch_idx < len(batches):
                next_batch_paths = batches[next_batch_idx]
                futures.append(
                    (
                        next_batch_paths,
                        executor.submit(load_image_batch, next_batch_paths, image_cache),
                    )
                )
                next_batch_idx += 1


class AsyncRustVideoReader:
    def __init__(self, video_path: Path, batch_size: int, queue_size: int, decode_threads: int):
        self.video_path = video_path
        self.batch_size = max(1, batch_size)
        self.queue: Queue = Queue(maxsize=max(1, queue_size))
        reader_class = get_video_reader_class()
        try:
            self.reader = reader_class(str(video_path), threads=max(0, decode_threads))
        except TypeError:
            self.reader = reader_class(str(video_path))
        self.frame_count = len(self.reader)
        shape = self.reader.get_shape()
        if len(shape) != 3:
            raise RuntimeError(f"video_reader-rs 返回了异常 shape: {shape}")
        _, self.height, self.width = [int(v) for v in shape]
        self.fps = float(self.reader.get_fps())
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            for start in range(0, self.frame_count, self.batch_size):
                end = min(start + self.batch_size, self.frame_count)
                indices = list(range(start, end))
                batch_rgb = self.reader.get_batch(indices)
                self.queue.put(np.asarray(batch_rgb))
        except Exception as exc:
            self._error = exc
        finally:
            self.queue.put(None)

    def read_batch(self) -> np.ndarray | None:
        item = self.queue.get()
        if item is None:
            if self._error is not None:
                raise self._error
            return None
        if self._error is not None:
            raise self._error
        return item

    def close(self):
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def infer_image_paths(args, model, class_names, font):
    if args.mode == "image":
        image_paths = [Path(args.input_image)]
    else:
        suffixes = {suffix.lower() for suffix in args.image_suffixes}
        image_paths = sorted(
            path for path in Path(args.input_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
    if not image_paths:
        raise RuntimeError("没有找到可推理的图片")

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    image_cache = ImageCache(args.image_cache_size)

    for batch_paths, batch_rgb in iter_prefetched_image_batches(
        image_paths=image_paths,
        batch_size=args.batch_size,
        io_workers=args.io_workers,
        prefetch_batches=args.prefetch_batches,
        image_cache=image_cache,
    ):
        outputs = predict_batch(
            model=model,
            frames_rgb=batch_rgb,
            top_k=args.top_k,
            score_thresh=args.score_thresh,
        )
        for path, frame_rgb, (masks, labels, scores) in zip(batch_paths, batch_rgb, outputs):
            original_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            rendered_bgr = render_frame(
                frame_bgr=original_bgr,
                masks=masks,
                labels=labels,
                scores=scores,
                class_names=class_names,
                alpha=args.alpha,
                font=font,
            )
            panel = render_side_by_side(original_bgr, rendered_bgr)
            if args.mode == "image":
                out_path = output_root if output_root.suffix else output_root / f"{path.stem}_rendered.png"
            else:
                out_path = output_root / path.name
            write_image_bgr(out_path, panel)


def infer_video_to_path(
    args,
    model,
    class_names,
    input_video: Path,
    output_path: Path,
    npz_output_path: Path | None = None,
    predict_lock=None,
):
    font = choose_font(args.font_path, args.font_size)
    if npz_output_path is None:
        npz_output_path = build_output_npz_path(output_path)
    dense_outputs_per_frame: list[dict[str, np.ndarray]] = []
    postprocessor = VideoPostProcessor(
        same_class_dedup_mask_iou_thresh=args.video_same_class_dedup_mask_iou_thresh,
        dedup_mask_iou_thresh=args.video_dedup_mask_iou_thresh,
        dedup_contain_thresh=args.video_dedup_contain_thresh,
        max_center_dist=args.video_track_max_center_dist,
        min_bbox_iou=args.video_track_min_bbox_iou,
        max_miss=args.video_track_max_miss,
        label_switch_frames=args.video_track_label_switch_frames,
    )
    reader = AsyncRustVideoReader(
        video_path=input_video,
        batch_size=args.batch_size,
        queue_size=max(1, args.prefetch_batches),
        decode_threads=max(0, args.io_workers),
    )

    fps = reader.fps if reader.fps > 0 else 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (reader.width * 2, reader.height),
    )

    try:
        while True:
            batch_rgb = reader.read_batch()
            if batch_rgb is None or len(batch_rgb) == 0:
                break

            batch_rgb_list = [np.asarray(frame) for frame in batch_rgb]
            if predict_lock is None:
                outputs = predict_batch(
                    model=model,
                    frames_rgb=batch_rgb_list,
                    top_k=args.top_k,
                    score_thresh=args.score_thresh,
                )
            else:
                with predict_lock:
                    outputs = predict_batch(
                        model=model,
                        frames_rgb=batch_rgb_list,
                        top_k=args.top_k,
                        score_thresh=args.score_thresh,
                    )

            for frame_rgb, (masks, labels, scores) in zip(batch_rgb_list, outputs):
                masks, labels, scores, obj_ids, motion_vectors = postprocessor.update(
                    masks, labels, scores
                )
                dense_outputs_per_frame.append(
                    {
                        "obj_ids": obj_ids,
                        "binary_masks": masks,
                        "class_ids": labels,
                        "scores": scores,
                        "motion_vectors": motion_vectors,
                    }
                )
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                rendered = render_frame(
                    frame_bgr=frame_bgr,
                    masks=masks,
                    labels=labels,
                    scores=scores,
                    class_names=class_names,
                    alpha=args.alpha,
                    font=font,
                )
                panel = render_side_by_side(frame_bgr, rendered)
                writer.write(panel)
    finally:
        reader.close()
        writer.release()
    save_dense_outputs_to_npz(
        dense_outputs_per_frame=dense_outputs_per_frame,
        output_path=npz_output_path,
        category_names=class_names,
    )


def collect_video_paths(args) -> list[Path]:
    if args.mode == "video":
        return [Path(args.input_video)]

    suffixes = {suffix.lower() for suffix in args.video_suffixes}
    video_paths: list[Path] = []
    if args.input_video_dir:
        video_paths.extend(
            path
            for path in Path(args.input_video_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
    if args.input_video_list:
        for line in Path(args.input_video_list).read_text(encoding="utf-8").splitlines():
            path_str = line.strip()
            if not path_str or path_str.startswith("#"):
                continue
            path = Path(path_str)
            if path.is_file() and path.suffix.lower() in suffixes:
                video_paths.append(path)

    unique_paths: list[Path] = []
    seen = set()
    for path in sorted(video_paths):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def infer_videos(args, model, class_names):
    video_paths = collect_video_paths(args)
    if not video_paths:
        raise RuntimeError("没有找到可推理的视频")

    output_root = Path(args.output_dir)
    output_video_dir = output_root / "mp4"
    output_npz_dir = output_root / "npz"
    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_npz_dir.mkdir(parents=True, exist_ok=True)
    predict_lock = threading.Lock()
    max_workers = min(max(1, args.video_workers), len(video_paths))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for video_path in video_paths:
            output_path = build_output_video_path(video_path, output_video_dir)
            npz_output_path = build_output_npz_path_for_video(video_path, output_npz_dir)
            if should_skip_video_inference(output_path, npz_output_path):
                print(f"跳过已完成视频: {video_path} -> {output_path} | {npz_output_path}")
                continue
            print(f"提交视频推理: {video_path} -> {output_path} | {npz_output_path}")
            future = executor.submit(
                infer_video_to_path,
                args,
                model,
                class_names,
                video_path,
                output_path,
                npz_output_path,
                predict_lock,
            )
            future_to_video[future] = video_path

        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            future.result()
            print(f"完成视频推理: {video_path}")


def main():
    args = parse_args()
    config = load_yaml(args.config)
    device = torch.device(args.device)
    model = build_model(config, args.ckpt, device)
    class_names = load_class_names(config)
    if args.mode == "video":
        if not args.input_video:
            raise ValueError("--mode video 时必须传 --input-video")
        if not args.output_path:
            raise ValueError("--mode video 时必须传 --output-path")
        infer_video_to_path(
            args=args,
            model=model,
            class_names=class_names,
            input_video=Path(args.input_video),
            output_path=Path(args.output_path),
        )
    elif args.mode == "videos":
        if not args.input_video_dir and not args.input_video_list:
            raise ValueError("--mode videos 时必须传 --input-video-dir 或 --input-video-list")
        if not args.output_dir:
            raise ValueError("--mode videos 时必须传 --output-dir")
        infer_videos(args, model, class_names)
    elif args.mode == "image":
        if not args.input_image:
            raise ValueError("--mode image 时必须传 --input-image")
        if not args.output_path:
            raise ValueError("--mode image 时必须传 --output-path")
        font = choose_font(args.font_path, args.font_size)
        infer_image_paths(args, model, class_names, font)
    else:
        if not args.input_dir:
            raise ValueError("--mode images 时必须传 --input-dir")
        if not args.output_path:
            raise ValueError("--mode images 时必须传 --output-path")
        font = choose_font(args.font_path, args.font_size)
        infer_image_paths(args, model, class_names, font)


if __name__ == "__main__":
    main()
