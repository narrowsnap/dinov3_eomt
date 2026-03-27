import argparse
import importlib
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLORS_BGR = np.array(
    [
        [60, 76, 231],
        [219, 152, 52],
        [113, 204, 46],
        [15, 196, 241],
        [182, 89, 155],
        [34, 126, 230],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="视频实例分割推理并导出可视化视频")
    parser.add_argument("--config", required=True, help="训练配置文件路径")
    parser.add_argument("--ckpt", required=True, help="训练完成后的 Lightning checkpoint")
    parser.add_argument("--input-video", required=True, help="输入视频路径")
    parser.add_argument("--output-video", required=True, help="输出视频路径")
    parser.add_argument("--device", default="cuda:0", help="推理设备，例如 cuda:0 / cpu")
    parser.add_argument("--score-thresh", type=float, default=0.25, help="实例置信度阈值")
    parser.add_argument("--top-k", type=int, default=None, help="每帧保留的候选实例数")
    parser.add_argument("--min-mask-area", type=int, default=200, help="最小 mask 面积")
    parser.add_argument("--nms-iou-thresh", type=float, default=0.7, help="mask NMS 的 IoU 阈值")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask 叠加透明度")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧，用于调试")
    parser.add_argument(
        "--show-label-score",
        action="store_true",
        help="是否在视频上显示类别名和置信度，默认不显示",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="类别名列表，例如 --class-names prostate cutting_area bladder",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="输出原图和预测拼接视频，默认只输出叠加结果",
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


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def filter_instances(
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    min_mask_area: int,
    nms_iou_thresh: float,
):
    keep_masks, keep_labels, keep_scores = [], [], []
    for mask, label, score in zip(masks, labels, scores):
        if int(mask.sum()) < min_mask_area:
            continue
        suppressed = False
        for kept_mask in keep_masks:
            if mask_iou(mask, kept_mask) > nms_iou_thresh:
                suppressed = True
                break
        if suppressed:
            continue
        keep_masks.append(mask)
        keep_labels.append(label)
        keep_scores.append(score)

    if not keep_masks:
        return (
            np.zeros((0, *masks.shape[-2:]), dtype=bool),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    return (
        np.stack(keep_masks),
        np.asarray(keep_labels),
        np.asarray(keep_scores, dtype=np.float32),
    )


def keep_best_instance_per_class(
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
):
    if len(labels) == 0:
        return masks, labels, scores

    best_indices = []
    seen_labels = set()
    for idx, label in enumerate(labels.tolist()):
        if label in seen_labels:
            continue
        seen_labels.add(label)
        best_indices.append(idx)

    return masks[best_indices], labels[best_indices], scores[best_indices]


@torch.inference_mode()
def predict_frame(model, frame_rgb: np.ndarray, top_k: int, score_thresh: float):
    device = model.device
    img = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(device=device)
    img_sizes = [img.shape[-2:]]

    transformed_imgs = model.resize_and_pad_imgs_instance_panoptic([img])
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=torch.float16):
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)

    mask_logits = F.interpolate(mask_logits_per_layer[-1], model.img_size, mode="bilinear")
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(mask_logits, img_sizes)[0]
    class_logits = class_logits_per_layer[-1][0]

    scores = class_logits.softmax(dim=-1)[:, :-1]
    labels = (
        torch.arange(scores.shape[-1], device=device)
        .unsqueeze(0)
        .repeat(scores.shape[0], 1)
        .flatten(0, 1)
    )
    flat_scores = scores.flatten(0, 1)
    if flat_scores.numel() == 0:
        return (
            np.zeros((0, frame_rgb.shape[0], frame_rgb.shape[1]), dtype=bool),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    topk_k = min(top_k, flat_scores.numel())
    topk_scores, topk_indices = flat_scores.topk(topk_k, sorted=False)
    labels = labels[topk_indices]
    query_indices = topk_indices // scores.shape[-1]
    mask_logits = mask_logits[query_indices]

    masks = mask_logits > 0
    mask_scores = (
        mask_logits.sigmoid().flatten(1) * masks.flatten(1)
    ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
    scores = topk_scores * mask_scores
    keep = scores >= score_thresh
    if not keep.any():
        return (
            np.zeros((0, frame_rgb.shape[0], frame_rgb.shape[1]), dtype=bool),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    masks = masks[keep].detach().cpu().numpy().astype(bool)
    labels = labels[keep].detach().cpu().numpy()
    scores = scores[keep].detach().cpu().numpy()

    order = np.argsort(-scores)
    return masks[order], labels[order], scores[order]


def render_frame(
    frame_bgr: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    class_names: list[str],
    side_by_side: bool,
    show_label_score: bool,
):
    overlay = frame_bgr.copy()

    for mask, label, score in zip(masks, labels, scores):
        color = COLORS_BGR[int(label) % len(COLORS_BGR)]
        overlay[mask] = (
            (1 - alpha) * overlay[mask].astype(np.float32)
            + alpha * color.astype(np.float32)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

        if not show_label_score:
            continue

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        class_name = (
            class_names[int(label)]
            if 0 <= int(label) < len(class_names)
            else f"class_{int(label)}"
        )
        text = f"{class_name} {float(score):.2f}"
        org = (int(xs.min()), max(20, int(ys.min()) - 5))
        cv2.putText(
            overlay,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )

    if side_by_side:
        return np.concatenate([frame_bgr, overlay], axis=1)
    return overlay


def format_seconds(seconds: float):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_progress(message: str):
    print(f"\r{message}", end="", flush=True)


def main():
    args = parse_args()

    config = load_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(config, args.ckpt, device)
    top_k = args.top_k or int(model.eval_top_k_instances)

    if args.class_names:
        class_names = args.class_names
    else:
        class_names = [f"class_{i}" for i in range(model.num_classes)]

    input_path = Path(args.input_video)
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开输入视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None
    if args.max_frames is not None:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames
    out_width = width * 2 if args.side_by_side else width

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    frame_idx = 0
    start_time = time.time()
    last_log_time = start_time
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames is not None and frame_idx > args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            masks, labels, scores = predict_frame(
                model=model,
                frame_rgb=frame_rgb,
                top_k=top_k,
                score_thresh=args.score_thresh,
            )
            masks, labels, scores = filter_instances(
            masks=masks,
            labels=labels,
            scores=scores,
            min_mask_area=args.min_mask_area,
            nms_iou_thresh=args.nms_iou_thresh,
        )
            masks, labels, scores = keep_best_instance_per_class(
                masks=masks,
                labels=labels,
                scores=scores,
            )
            output_frame = render_frame(
                frame_bgr=frame_bgr,
                masks=masks,
                labels=labels,
                scores=scores,
                alpha=args.alpha,
                class_names=class_names,
                side_by_side=args.side_by_side,
                show_label_score=args.show_label_score,
            )
            writer.write(output_frame)

            now = time.time()
            should_log = (
                frame_idx == 1
                or frame_idx % 10 == 0
                or (total_frames is not None and frame_idx == total_frames)
                or now - last_log_time >= 2.0
            )
            if should_log:
                elapsed = max(now - start_time, 1e-6)
                avg_fps = frame_idx / elapsed
                if total_frames is not None:
                    progress = min(frame_idx / total_frames, 1.0)
                    eta = (total_frames - frame_idx) / max(avg_fps, 1e-6)
                    print_progress(
                        f"进度: {frame_idx}/{total_frames} "
                        f"({progress * 100:.1f}%) | "
                        f"平均速度: {avg_fps:.2f} 帧/秒 | "
                        f"预计剩余: {format_seconds(eta)}"
                    )
                else:
                    print_progress(
                        f"进度: {frame_idx} 帧 | "
                        f"平均速度: {avg_fps:.2f} 帧/秒 | "
                        f"已耗时: {format_seconds(elapsed)}"
                    )
                last_log_time = now
    finally:
        cap.release()
        writer.release()

    print()
    total_elapsed = time.time() - start_time
    print(
        f"处理完成: {frame_idx} 帧 | 总耗时: {format_seconds(total_elapsed)} | "
        f"平均速度: {frame_idx / max(total_elapsed, 1e-6):.2f} 帧/秒"
    )
    print(f"输出视频已保存到: {output_path}")


if __name__ == "__main__":
    main()
