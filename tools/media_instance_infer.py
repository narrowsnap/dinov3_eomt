#!/usr/bin/env python3

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import ctypes
import importlib
import json
from collections import OrderedDict
import os
from pathlib import Path
from queue import Queue
import site
import threading
import sys
from typing import Any

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
    parser.add_argument("--score-thresh", type=float, default=0.25, help="分数阈值")
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


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


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
        libs_dir = path / "video_reader_rs.libs"
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
        scores = class_logits[batch_idx].softmax(dim=-1)[:, :-1]
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
        sample_mask_logits = mask_logits[batch_idx][query_indices]
        masks = sample_mask_logits > 0
        mask_scores = (
            sample_mask_logits.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
        scores_out = topk_scores * mask_scores
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
    predict_lock: threading.Lock | None = None,
):
    font = choose_font(args.font_path, args.font_size)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predict_lock = threading.Lock()
    max_workers = min(max(1, args.video_workers), len(video_paths))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for video_path in video_paths:
            output_path = output_dir / f"{video_path.stem}_rendered.mp4"
            print(f"提交视频推理: {video_path} -> {output_path}")
            future = executor.submit(
                infer_video_to_path,
                args,
                model,
                class_names,
                video_path,
                output_path,
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
