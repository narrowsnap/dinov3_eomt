#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
import pickle
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from media_instance_infer import (  # noqa: E402
    AsyncRustVideoReader,
    build_model,
    build_output_npz_path,
    build_output_npz_path_for_video,
    build_output_video_path,
    choose_font,
    class_id_to_name,
    collect_video_paths,
    get_category_color,
    load_class_names,
    load_yaml,
    mask_to_box,
    predict_batch,
    render_side_by_side,
    save_dense_outputs_to_npz,
)
from media_instance_track_infer import (  # noqa: E402
    BoostStyleTracker,
    TrackDetection,
)


def parse_args():
    parser = argparse.ArgumentParser(description="实例分割视频推理 + BoostTrack++ + ECC + ReID")
    parser.add_argument("--config", required=True, help="训练配置文件路径")
    parser.add_argument("--ckpt", required=True, help="训练 checkpoint 路径")
    parser.add_argument("--mode", choices=["video", "videos"], default="video", help="推理模式")
    parser.add_argument("--input-video", help="输入视频路径")
    parser.add_argument("--input-video-dir", help="输入视频目录")
    parser.add_argument("--input-video-list", help="输入视频列表文件")
    parser.add_argument("--output-path", help="输出视频路径")
    parser.add_argument("--output-dir", help="多视频输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    parser.add_argument("--score-thresh", type=float, default=0.5, help="实例分数阈值")
    parser.add_argument("--top-k", type=int, default=64, help="每张图最多保留 query 数")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask 叠加透明度")
    parser.add_argument("--font-size", type=int, default=22, help="字体大小")
    parser.add_argument("--font-path", default="/home/zhouyang/simhei.ttf", help="中文字体路径")
    parser.add_argument("--batch-size", type=int, default=4, help="视频推理 batch size")
    parser.add_argument("--io-workers", type=int, default=4, help="视频解码线程数")
    parser.add_argument("--prefetch-batches", type=int, default=3, help="视频预取批次数")
    parser.add_argument("--video-workers", type=int, default=2, help="多视频并发数量")
    parser.add_argument("--track-thresh", type=float, default=0.5, help="创建轨迹的最小分数")
    parser.add_argument("--match-thresh", type=float, default=0.2, help="轨迹匹配阈值")
    parser.add_argument("--track-buffer", type=int, default=30, help="轨迹最大丢失帧数")
    parser.add_argument("--min-hits", type=int, default=2, help="轨迹稳定所需最少命中次数")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="轨迹 IoU 匹配阈值")
    parser.add_argument("--lambda-iou", type=float, default=0.5, help="IoU 代价权重")
    parser.add_argument("--lambda-mhd", type=float, default=0.25, help="马氏距离代价权重")
    parser.add_argument("--lambda-shape", type=float, default=0.25, help="框形状相似度权重")
    parser.add_argument("--lambda-emb", type=float, default=1.5, help="ReID embedding 代价权重")
    parser.add_argument("--dlo-boost-coef", type=float, default=0.65, help="DLO boost 系数")
    parser.add_argument("--disable-dlo-boost", action="store_true", help="关闭 DLO boost")
    parser.add_argument("--disable-duo-boost", action="store_true", help="关闭 DUO boost")
    parser.add_argument("--disable-rich-s", action="store_true", help="关闭 rich similarity")
    parser.add_argument("--disable-soft-boost", action="store_true", help="关闭 soft boost")
    parser.add_argument("--disable-varying-th", action="store_true", help="关闭 varying threshold boost")
    parser.add_argument("--same-class-only", action="store_true", help="只允许同类别轨迹匹配")
    parser.add_argument("--disable-ecc", action="store_true", help="关闭 ECC 相机运动补偿")
    parser.add_argument("--disable-reid", action="store_true", help="关闭 ReID embedding 分支")
    parser.add_argument("--ecc-scale", type=float, default=0.15, help="ECC 缩放比例")
    parser.add_argument("--ecc-max-iter", type=int, default=100, help="ECC 最大迭代次数")
    parser.add_argument("--ecc-eps", type=float, default=1e-4, help="ECC 收敛阈值")
    parser.add_argument("--reid-batch-size", type=int, default=64, help="ReID batch size")
    parser.add_argument("--reid-cache-dir", default="cache/embeddings_eomt", help="ReID 缓存目录")
    parser.add_argument("--ecc-cache-dir", default="cache/ecc_eomt", help="ECC 缓存目录")
    return parser.parse_args()


def ecc(
    src,
    dst,
    warp_mode=cv2.MOTION_EUCLIDEAN,
    eps=1e-5,
    max_iter=100,
    scale=0.1,
    align=False,
):
    assert src.shape == dst.shape, "source 和 target 图像尺寸必须一致"
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    if scale is not None:
        if isinstance(scale, float):
            if scale != 1:
                src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                scale = [scale, scale]
            else:
                src_r, dst_r = src, dst
                scale = None
        else:
            src_r, dst_r = src, dst
    else:
        src_r, dst_r = src, dst

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)
    _, warp_matrix = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)

    if scale is not None:
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

    if align:
        sz = src.shape
        src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    return warp_matrix, None


class ECCCompensator:
    def __init__(
        self,
        cache_dir: str,
        video_name: str,
        scale: float,
        eps: float,
        max_iter: int,
        use_cache: bool = True,
    ):
        self.scale = scale
        self.eps = eps
        self.max_iter = max_iter
        self.prev_image = None
        self.video_name = video_name
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{video_name}.json"
        self.cache = {}
        if self.use_cache and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.cache = {key: np.asarray(value, dtype=np.float32) for key, value in self.cache.items()}
            except Exception:
                self.cache = {}

    def __call__(self, frame_bgr: np.ndarray, frame_id: int) -> np.ndarray:
        if frame_id == 1:
            self.prev_image = deepcopy(frame_bgr)
            return np.eye(3, dtype=np.float32)
        key = str(frame_id)
        if key in self.cache:
            self.prev_image = deepcopy(frame_bgr)
            return self.cache[key]
        result, _ = ecc(
            self.prev_image,
            frame_bgr,
            warp_mode=cv2.MOTION_EUCLIDEAN,
            eps=self.eps,
            max_iter=self.max_iter,
            scale=self.scale,
            align=False,
        )
        self.prev_image = deepcopy(frame_bgr)
        result = np.vstack((result, np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)))
        if self.use_cache:
            self.cache[key] = result
        return result

    def save_cache(self):
        if not self.use_cache:
            return
        serializable = {key: value.tolist() for key, value in self.cache.items()}
        self.cache_path.write_text(json.dumps(serializable), encoding="utf-8")


class ReIDEmbedder:
    def __init__(self, device: torch.device, cache_dir: str, video_name: str, batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size
        self.crop_size = (256, 128)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{video_name}.pkl"
        self.cache = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as fp:
                    self.cache = pickle.load(fp)
            except Exception:
                self.cache = {}

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        backbone = torchvision.models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device)
        self.model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _crop_patch(self, frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        patch = frame_bgr[y1:y2, x1:x2]
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch = cv2.resize(patch, (self.crop_size[1], self.crop_size[0]), interpolation=cv2.INTER_LINEAR)
        patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        return patch

    def compute(self, frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], tag: str) -> np.ndarray:
        if tag in self.cache:
            cached = self.cache[tag]
            if cached.shape[0] == len(boxes):
                return cached
        if not boxes:
            return np.zeros((0, 2048), dtype=np.float32)

        crops = torch.stack([self._crop_patch(frame_bgr, box) for box in boxes], dim=0)
        embeddings = []
        for start in range(0, len(crops), self.batch_size):
            batch = crops[start : start + self.batch_size].to(self.device)
            batch = (batch - self.mean) / self.std
            with torch.no_grad():
                feat = self.model(batch)
                feat = feat.flatten(1)
                feat = F.normalize(feat, dim=1)
            embeddings.append(feat.cpu())
        embs = torch.cat(embeddings, dim=0).numpy().astype(np.float32)
        self.cache[tag] = embs
        return embs

    def save_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)


class BoostStyleTrackerECC(BoostStyleTracker):
    def __init__(self, *args, lambda_emb: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_emb = lambda_emb

    def camera_update(self, transform: np.ndarray) -> None:
        for track in self.tracks:
            x1, y1, x2, y2 = track.get_state()
            x1_, y1_, _ = transform @ np.asarray([x1, y1, 1.0], dtype=np.float32)
            x2_, y2_, _ = transform @ np.asarray([x2, y2, 1.0], dtype=np.float32)
            w, h = x2_ - x1_, y2_ - y1_
            if h <= 1e-6:
                continue
            cx, cy = x1_ + w / 2.0, y1_ + h / 2.0
            track.kalman.x[:4] = np.asarray([cx, cy, h, w / h], dtype=np.float32)

    def associate_with_embeddings(
        self,
        detections: np.ndarray,
        track_confidences: np.ndarray,
        det_embeddings: np.ndarray | None,
        trk_embeddings: np.ndarray | None,
    ):
        tracker_boxes = self.get_tracker_boxes()
        iou_matrix = self.get_iou_matrix(detections[:, :4], buffered=False)
        cost_matrix = deepcopy(iou_matrix)
        det_scores = detections[:, 4]
        conf = det_scores.reshape((-1, 1)) * track_confidences.reshape((1, -1))
        conf[iou_matrix < self.iou_threshold] = 0.0
        cost_matrix += self.lambda_iou * conf * iou_matrix

        mahalanobis_distance = self.get_mh_dist_matrix(detections)
        if mahalanobis_distance.size > 0:
            cost_matrix += self.lambda_mhd * self.mh_similarity(mahalanobis_distance)
            cost_matrix += self.lambda_shape * conf * self.shape_similarity_local(
                detections[:, :4], tracker_boxes[:, :4]
            )
        if det_embeddings is not None and trk_embeddings is not None and len(trk_embeddings) > 0:
            emb_cost = det_embeddings @ trk_embeddings.T
            cost_matrix += self.lambda_emb * emb_cost
        if self.same_class_only:
            det_labels = detections[:, 5].astype(np.int32)
            trk_labels = np.asarray([track.label for track in self.tracks], dtype=np.int32)
            mask = det_labels.reshape((-1, 1)) == trk_labels.reshape((1, -1))
            cost_matrix = np.where(mask, cost_matrix, 0.0)
            iou_matrix = np.where(mask, iou_matrix, 0.0)
        return self.assignment_local(detections, tracker_boxes, iou_matrix, cost_matrix, self.iou_threshold)

    @staticmethod
    def mh_similarity(mahalanobis_distance: np.ndarray) -> np.ndarray:
        from media_instance_track_infer import mh_dist_similarity

        return mh_dist_similarity(mahalanobis_distance)

    @staticmethod
    def shape_similarity_local(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        from media_instance_track_infer import shape_similarity

        return shape_similarity(detects, tracks)

    @staticmethod
    def assignment_local(detections, trackers, iou_matrix, cost_matrix, threshold):
        from media_instance_track_infer import linear_assignment

        return linear_assignment(detections, trackers, iou_matrix, cost_matrix, threshold)


def render_tracked_frame(
    frame_bgr: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    track_ids: np.ndarray,
    class_names: Sequence[str],
    alpha: float,
    font,
) -> np.ndarray:
    overlay = frame_bgr.copy().astype(np.float32)
    text_items = []
    for mask, label, score, track_id in zip(masks, labels, scores, track_ids):
        class_name = class_id_to_name(int(label), class_names)
        color_rgb = get_category_color(class_name)
        color_bgr = np.array(color_rgb[::-1], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color_bgr * alpha
        box = mask_to_box(mask)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(v) for v in color_bgr), 2)
        text_items.append(((x1, max(0, y1 - 28)), f"{class_name} ID:{int(track_id)}", color_rgb))

    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    for (x, y), text, color in text_items:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(image)[:, :, ::-1]


def infer_video_track_ecc_to_path(
    args,
    model,
    class_names: Sequence[str],
    input_video: Path,
    output_path: Path,
    npz_output_path: Path | None = None,
):
    font = choose_font(args.font_path, args.font_size)
    if npz_output_path is None:
        npz_output_path = build_output_npz_path(output_path)
    tracker = BoostStyleTrackerECC(
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        lambda_iou=args.lambda_iou,
        lambda_mhd=args.lambda_mhd,
        lambda_shape=args.lambda_shape,
        use_dlo_boost=not args.disable_dlo_boost,
        use_duo_boost=not args.disable_duo_boost,
        dlo_boost_coef=args.dlo_boost_coef,
        use_rich_s=not args.disable_rich_s,
        use_sb=not args.disable_soft_boost,
        use_vt=not args.disable_varying_th,
        same_class_only=args.same_class_only,
        lambda_emb=args.lambda_emb,
    )
    reader = AsyncRustVideoReader(
        video_path=input_video,
        batch_size=args.batch_size,
        queue_size=max(1, args.prefetch_batches),
        decode_threads=max(0, args.io_workers),
    )
    ecc_comp = None if args.disable_ecc else ECCCompensator(
        cache_dir=args.ecc_cache_dir,
        video_name=input_video.stem,
        scale=args.ecc_scale,
        eps=args.ecc_eps,
        max_iter=args.ecc_max_iter,
    )
    embedder = None if args.disable_reid else ReIDEmbedder(
        device=torch.device(args.device),
        cache_dir=args.reid_cache_dir,
        video_name=input_video.stem,
        batch_size=args.reid_batch_size,
    )
    fps = reader.fps if reader.fps > 0 else 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (reader.width * 2, reader.height),
    )
    dense_outputs_per_frame = []
    frame_id = 0
    try:
        while True:
            batch_rgb = reader.read_batch()
            if batch_rgb is None or len(batch_rgb) == 0:
                break
            batch_rgb_list = [np.asarray(frame) for frame in batch_rgb]
            outputs = predict_batch(
                model=model,
                frames_rgb=batch_rgb_list,
                top_k=args.top_k,
                score_thresh=args.score_thresh,
            )
            for frame_rgb, (masks, labels, scores) in zip(batch_rgb_list, outputs):
                frame_id += 1
                tracker.frame_count += 1
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if ecc_comp is not None:
                    transform = ecc_comp(frame_bgr, frame_id)
                    tracker.camera_update(transform)

                detections = []
                for mask, label, score in zip(masks, labels, scores):
                    box = mask_to_box(mask)
                    if box is None:
                        continue
                    detections.append(
                        TrackDetection(
                            mask=mask,
                            label=int(label),
                            score=float(score),
                            box=box,
                            center=np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32),
                            area=int(mask.sum()),
                        )
                    )

                detections_np = np.asarray(
                    [[det.box[0], det.box[1], det.box[2], det.box[3], det.score, det.label] for det in detections],
                    dtype=np.float32,
                ) if detections else np.zeros((0, 6), dtype=np.float32)

                for track in tracker.tracks:
                    track.predict()

                if detections_np.size > 0 and tracker.use_dlo_boost:
                    detections_np = tracker.dlo_confidence_boost(detections_np)
                if detections_np.size > 0 and tracker.use_duo_boost:
                    detections_np = tracker.duo_confidence_boost(detections_np)

                if detections_np.size > 0:
                    for idx, det in enumerate(detections):
                        det.score = float(detections_np[idx, 4])

                keep_indices = [idx for idx, det in enumerate(detections) if det.score >= tracker.track_thresh]
                filtered_detections = [detections[idx] for idx in keep_indices]
                filtered_np = detections_np[keep_indices] if len(keep_indices) else np.zeros((0, 6), dtype=np.float32)
                track_confidences = (
                    np.asarray([track.get_confidence() for track in tracker.tracks], dtype=np.float32)
                    if tracker.tracks
                    else np.zeros((0,), dtype=np.float32)
                )

                det_embs = None
                trk_embs = None
                if embedder is not None and len(filtered_detections) > 0:
                    det_boxes = [det.box for det in filtered_detections]
                    det_embs = embedder.compute(frame_bgr, det_boxes, f"{input_video.stem}:{frame_id}")
                    trk_embs_list = []
                    for track in tracker.tracks:
                        if hasattr(track, "embedding") and track.embedding is not None:
                            trk_embs_list.append(track.embedding)
                        else:
                            trk_embs_list.append(np.zeros((det_embs.shape[1],), dtype=np.float32))
                    trk_embs = np.asarray(trk_embs_list, dtype=np.float32) if trk_embs_list else np.zeros((0, det_embs.shape[1]), dtype=np.float32)

                if len(filtered_detections) > 0 and tracker.tracks:
                    matches, unmatched_dets, unmatched_trks = tracker.associate_with_embeddings(
                        detections=filtered_np,
                        track_confidences=track_confidences,
                        det_embeddings=det_embs,
                        trk_embeddings=trk_embs,
                    )
                    valid_matches = [(int(det_idx), int(trk_idx)) for det_idx, trk_idx in matches]
                else:
                    valid_matches = []
                    unmatched_dets = np.asarray(list(range(len(filtered_detections))), dtype=np.int32)
                    unmatched_trks = np.asarray(list(range(len(tracker.tracks))), dtype=np.int32)

                for det_idx, trk_idx in valid_matches:
                    tracker.tracks[trk_idx].update(filtered_detections[det_idx])
                    if det_embs is not None:
                        tracker.tracks[trk_idx].embedding = det_embs[det_idx]

                for det_idx in unmatched_dets:
                    det = filtered_detections[int(det_idx)]
                    tracker._create_track(det)
                    if det_embs is not None:
                        tracker.tracks[-1].embedding = det_embs[int(det_idx)]

                alive_tracks = []
                for idx, track in enumerate(tracker.tracks):
                    if idx in set(int(x) for x in unmatched_trks):
                        pass
                    if track.time_since_update <= tracker.track_buffer:
                        alive_tracks.append(track)
                tracker.tracks = alive_tracks

                visible_tracks = [
                    track
                    for track in tracker.tracks
                    if track.time_since_update == 0
                    and (track.hit_streak >= tracker.min_hits or tracker.frame_count <= tracker.min_hits)
                ]
                visible_tracks.sort(key=lambda item: item.track_id)
                tracked_masks = np.asarray([track.mask for track in visible_tracks], dtype=bool) if visible_tracks else np.zeros((0,) + masks.shape[1:], dtype=bool)
                tracked_labels = np.asarray([track.label for track in visible_tracks], dtype=np.int64) if visible_tracks else np.zeros((0,), dtype=np.int64)
                tracked_scores = np.asarray([track.score for track in visible_tracks], dtype=np.float32) if visible_tracks else np.zeros((0,), dtype=np.float32)
                track_ids = np.asarray([track.track_id for track in visible_tracks], dtype=np.int64) if visible_tracks else np.zeros((0,), dtype=np.int64)
                dense_outputs_per_frame.append(
                    {
                        "obj_ids": track_ids,
                        "binary_masks": tracked_masks,
                        "class_ids": tracked_labels,
                        "scores": tracked_scores,
                        "reid_embeds": np.asarray(
                            [getattr(track, "embedding", np.zeros((2048,), dtype=np.float32)) for track in visible_tracks],
                            dtype=np.float32,
                        ) if visible_tracks else np.zeros((0, 2048), dtype=np.float32),
                    }
                )
                rendered = render_tracked_frame(
                    frame_bgr=frame_bgr,
                    masks=tracked_masks,
                    labels=tracked_labels,
                    scores=tracked_scores,
                    track_ids=track_ids,
                    class_names=class_names,
                    alpha=args.alpha,
                    font=font,
                )
                writer.write(render_side_by_side(frame_bgr, rendered))
    finally:
        reader.close()
        writer.release()
        if ecc_comp is not None:
            ecc_comp.save_cache()
        if embedder is not None:
            embedder.save_cache()

    save_dense_outputs_to_npz(
        dense_outputs_per_frame=dense_outputs_per_frame,
        output_path=npz_output_path,
        category_names=class_names,
    )


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
        infer_video_track_ecc_to_path(
            args=args,
            model=model,
            class_names=class_names,
            input_video=Path(args.input_video),
            output_path=Path(args.output_path),
        )
        return

    if not args.input_video_dir and not args.input_video_list:
        raise ValueError("--mode videos 时必须传 --input-video-dir 或 --input-video-list")
    if not args.output_dir:
        raise ValueError("--mode videos 时必须传 --output-dir")

    video_paths = collect_video_paths(args)
    if not video_paths:
        raise RuntimeError("没有找到可推理的视频")
    output_root = Path(args.output_dir)
    output_video_dir = output_root / "mp4"
    output_npz_dir = output_root / "npz"
    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_npz_dir.mkdir(parents=True, exist_ok=True)
    max_workers = min(max(1, args.video_workers), len(video_paths))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {}
        for video_path in video_paths:
            output_video_path = build_output_video_path(video_path, output_video_dir)
            output_npz_path = build_output_npz_path_for_video(video_path, output_npz_dir)
            future = executor.submit(
                infer_video_track_ecc_to_path,
                args,
                model,
                class_names,
                video_path,
                output_video_path,
                output_npz_path,
            )
            future_to_video[future] = (video_path, output_video_path, output_npz_path)
        for future in as_completed(future_to_video):
            video_path, output_video_path, _ = future_to_video[future]
            future.result()
            print(f"完成 ECC+ReID 跟踪推理: {video_path} -> {output_video_path}")


if __name__ == "__main__":
    main()
