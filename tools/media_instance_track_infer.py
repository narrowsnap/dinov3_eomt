#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
import scipy.linalg
from scipy.optimize import linear_sum_assignment
import torch

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
    mask_center,
    mask_to_box,
    predict_batch,
    render_side_by_side,
    save_dense_outputs_to_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(description="实例分割视频推理 + BoostTrack 风格跟踪")
    parser.add_argument("--config", required=True, help="训练配置文件路径")
    parser.add_argument("--ckpt", required=True, help="训练 checkpoint 路径")
    parser.add_argument(
        "--mode",
        choices=["video", "videos"],
        required=True,
        help="推理模式",
    )
    parser.add_argument("--input-video", help="输入单个视频路径")
    parser.add_argument("--input-video-dir", help="输入视频目录")
    parser.add_argument("--input-video-list", help="输入视频列表文件")
    parser.add_argument("--output-path", help="单视频输出路径")
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
    parser.add_argument("--dlo-boost-coef", type=float, default=0.65, help="DLO boost 系数")
    parser.add_argument("--disable-dlo-boost", action="store_true", help="关闭 DLO boost")
    parser.add_argument("--disable-duo-boost", action="store_true", help="关闭 DUO boost")
    parser.add_argument("--disable-rich-s", action="store_true", help="关闭 rich similarity")
    parser.add_argument("--disable-soft-boost", action="store_true", help="关闭 soft boost")
    parser.add_argument("--disable-varying-th", action="store_true", help="关闭 varying threshold boost")
    parser.add_argument("--same-class-only", action="store_true", help="只允许同类别轨迹匹配")
    parser.add_argument(
        "--video-suffixes",
        nargs="*",
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
        help="多视频模式的视频后缀过滤",
    )
    return parser.parse_args()


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    r = w / float(h + 1e-6)
    return np.array([x, y, h, r], dtype=np.float32)


def convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    h = x[2]
    r = x[3]
    w = 0.0 if r <= 0 else r * h
    return np.array(
        [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0],
        dtype=np.float32,
    )


class ConstantNoise:
    def __init__(self, x_dim: int, z_dim: int):
        self.x_dim = x_dim
        self.z_dim = z_dim

    def get_init_state_cov(self, z: np.ndarray) -> np.ndarray:
        covariance = np.eye(self.x_dim, dtype=np.float32)
        covariance[4:, 4:] *= 1000.0
        covariance *= 10.0
        return covariance

    def get_R(self, x: np.ndarray, confidence: float = 0.0) -> np.ndarray:
        return np.diag([1.0, 1.0, 10.0, 0.01]).astype(np.float32)

    def get_Q(self, x: np.ndarray) -> np.ndarray:
        covariance = np.eye(self.x_dim, dtype=np.float32)
        covariance[4:, 4:] *= 0.01
        return covariance


class KalmanFilter:
    def __init__(self, z: np.ndarray, ndim: int = 8, dt: int = 1):
        self.dt = dt
        self.ndim = ndim
        self.cov_update_policy = ConstantNoise(ndim, z.size)
        self._motion_mat = np.eye(ndim, ndim, dtype=np.float32)
        for i in range(4 - (ndim % 2)):
            self._motion_mat[i, i + 4] = dt
        self._update_mat = np.eye(4, ndim, dtype=np.float32)
        self.x = np.zeros((ndim,), dtype=np.float32)
        self.x[:4] = z[:]
        self.covariance = self.cov_update_policy.get_init_state_cov(z)

    def predict(self):
        motion_cov = self.cov_update_policy.get_Q(self.x)
        self.x = np.dot(self._motion_mat, self.x)
        self.covariance = np.linalg.multi_dot(
            (self._motion_mat, self.covariance, self._motion_mat.T)
        ) + motion_cov
        return self.x, self.covariance

    def project(self):
        innovation_cov = self.cov_update_policy.get_R(self.x, 0)
        mean = np.dot(self._update_mat, self.x)
        covariance = np.linalg.multi_dot(
            (self._update_mat, self.covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, z: np.ndarray):
        projected_mean, projected_cov = self.project()
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(self.covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = z - projected_mean
        self.x = self.x + np.dot(innovation, kalman_gain.T)
        self.covariance = self.covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return self.x, self.covariance


@dataclass
class TrackDetection:
    mask: np.ndarray
    label: int
    score: float
    box: tuple[int, int, int, int]
    center: np.ndarray
    area: int


@dataclass
class TrackState:
    track_id: int
    kalman: KalmanFilter
    label: int
    score: float
    mask: np.ndarray
    box: tuple[int, int, int, int]
    center: np.ndarray
    area: int
    time_since_update: int = 0
    hit_streak: int = 1
    age: int = 1
    label_votes: dict[int, int] = field(default_factory=dict)

    def predict(self) -> np.ndarray:
        self.kalman.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, detection: TrackDetection) -> None:
        self.kalman.update(convert_bbox_to_z(np.asarray(detection.box, dtype=np.float32)))
        self.label_votes[detection.label] = self.label_votes.get(detection.label, 0) + 1
        self.label = max(self.label_votes.items(), key=lambda item: item[1])[0]
        self.score = detection.score
        self.mask = detection.mask
        self.box = detection.box
        self.center = detection.center
        self.area = detection.area
        self.time_since_update = 0
        self.hit_streak += 1

    def mark_missed(self) -> None:
        return None

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kalman.x)

    def get_confidence(self, coef: float = 0.9) -> float:
        warmup = 7
        if self.age < warmup:
            return coef ** (warmup - self.age)
        return coef ** max(self.time_since_update - 1, 0)


def iou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)
    boxes2 = np.expand_dims(bboxes2, 0)
    boxes1 = np.expand_dims(bboxes1, 1)
    xx1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    yy1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    xx2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    yy2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    union = (
        (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        + (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        - inter
    )
    return np.where(union > 0, inter / union, 0.0)


def soft_biou_batch(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)
    boxes2 = np.expand_dims(bboxes2, 0)
    boxes1 = np.expand_dims(bboxes1, 1)
    k1 = 0.25
    k2 = 0.5
    b2conf = boxes2[..., 4]
    b1x1 = boxes1[..., 0] - (boxes1[..., 2] - boxes1[..., 0]) * (1 - b2conf) * k1
    b2x1 = boxes2[..., 0] - (boxes2[..., 2] - boxes2[..., 0]) * (1 - b2conf) * k2
    xx1 = np.maximum(b1x1, b2x1)
    b1y1 = boxes1[..., 1] - (boxes1[..., 3] - boxes1[..., 1]) * (1 - b2conf) * k1
    b2y1 = boxes2[..., 1] - (boxes2[..., 3] - boxes2[..., 1]) * (1 - b2conf) * k2
    yy1 = np.maximum(b1y1, b2y1)
    b1x2 = boxes1[..., 2] + (boxes1[..., 2] - boxes1[..., 0]) * (1 - b2conf) * k1
    b2x2 = boxes2[..., 2] + (boxes2[..., 2] - boxes2[..., 0]) * (1 - b2conf) * k2
    xx2 = np.minimum(b1x2, b2x2)
    b1y2 = boxes1[..., 3] + (boxes1[..., 3] - boxes1[..., 1]) * (1 - b2conf) * k1
    b2y2 = boxes2[..., 3] + (boxes2[..., 3] - boxes2[..., 1]) * (1 - b2conf) * k2
    yy2 = np.minimum(b1y2, b2y2)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    union = ((b1x2 - b1x1) * (b1y2 - b1y1)) + ((b2x2 - b2x1) * (b2y2 - b2y1)) - inter
    return np.where(union > 0, inter / union, 0.0)


def shape_similarity(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((len(detects), len(tracks)), dtype=np.float32)
    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(
        -(
            np.abs(dw - tw) / np.maximum(dw, tw)
            + np.abs(dh - th) / np.maximum(dh, th)
        )
    )


def mh_dist_similarity(mahalanobis_distance: np.ndarray, softmax_temp: float = 1.0) -> np.ndarray:
    limit = 13.2767
    mh_distance = deepcopy(mahalanobis_distance)
    mask = mh_distance > limit
    mh_distance[mask] = limit
    mh_distance = limit - mh_distance
    exp_distance = np.exp(mh_distance / softmax_temp)
    denom = np.maximum(exp_distance.sum(0, keepdims=True), 1e-12)
    mh_similarity = exp_distance / denom
    mh_similarity = np.where(mask, 0.0, mh_similarity)
    return mh_similarity


def linear_assignment(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        matches = np.empty((0, 2), dtype=np.int32)
    else:
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)
        matches = np.asarray(list(zip(row_indices, col_indices)), dtype=np.int32)
        if matches.size == 0:
            matches = np.empty((0, 2), dtype=np.int32)

    unmatched_detections = [idx for idx in range(len(detections)) if idx not in matches[:, 0]] if len(matches) else list(range(len(detections)))
    unmatched_trackers = [idx for idx in range(len(trackers)) if idx not in matches[:, 1]] if len(matches) else list(range(len(trackers)))

    valid_matches = []
    for det_idx, trk_idx in matches:
        if iou_matrix[det_idx, trk_idx] >= threshold:
            valid_matches.append([det_idx, trk_idx])
        else:
            unmatched_detections.append(int(det_idx))
            unmatched_trackers.append(int(trk_idx))

    if valid_matches:
        valid_matches_arr = np.asarray(valid_matches, dtype=np.int32)
    else:
        valid_matches_arr = np.empty((0, 2), dtype=np.int32)
    return (
        valid_matches_arr,
        np.asarray(sorted(set(unmatched_detections)), dtype=np.int32),
        np.asarray(sorted(set(unmatched_trackers)), dtype=np.int32),
    )


class BoostStyleTracker:
    def __init__(
        self,
        track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        min_hits: int,
        iou_threshold: float,
        lambda_iou: float,
        lambda_mhd: float,
        lambda_shape: float,
        use_dlo_boost: bool,
        use_duo_boost: bool,
        dlo_boost_coef: float,
        use_rich_s: bool,
        use_sb: bool,
        use_vt: bool,
        same_class_only: bool,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.lambda_iou = lambda_iou
        self.lambda_mhd = lambda_mhd
        self.lambda_shape = lambda_shape
        self.use_dlo_boost = use_dlo_boost
        self.use_duo_boost = use_duo_boost
        self.dlo_boost_coef = dlo_boost_coef
        self.use_rich_s = use_rich_s
        self.use_sb = use_sb
        self.use_vt = use_vt
        self.same_class_only = same_class_only
        self.frame_count = 0
        self.tracks: list[TrackState] = []
        self.next_track_id = 1

    def get_tracker_boxes(self) -> np.ndarray:
        if not self.tracks:
            return np.zeros((0, 5), dtype=np.float32)
        tracker_boxes = np.zeros((len(self.tracks), 5), dtype=np.float32)
        for idx, track in enumerate(self.tracks):
            box = track.get_state()
            tracker_boxes[idx] = [box[0], box[1], box[2], box[3], track.get_confidence()]
        return tracker_boxes

    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        trackers = self.get_tracker_boxes()
        if buffered:
            return soft_biou_batch(detections, trackers)
        return iou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.tracks) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        z = np.zeros((len(detections), n_dims), dtype=np.float32)
        x = np.zeros((len(self.tracks), n_dims), dtype=np.float32)
        sigma_inv = np.zeros_like(x, dtype=np.float32)
        for idx in range(len(detections)):
            z[idx, :n_dims] = convert_bbox_to_z(detections[idx, :4])[:n_dims]
        for idx, track in enumerate(self.tracks):
            x[idx] = track.kalman.x[:n_dims]
            sigma_diag = np.diag(track.kalman.covariance[:n_dims, :n_dims]).astype(np.float32)
            sigma_inv[idx] = np.reciprocal(np.maximum(sigma_diag, 1e-12))
        return (
            (z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2
            * sigma_inv.reshape((1, -1, n_dims))
        ).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        detections = detections.copy()
        if detections.size == 0:
            return detections
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections)
        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)
            mask = (min_mh_dists > limit) & (detections[:, 4] < self.track_thresh)
            boost_detections = detections[mask]
            boost_indices = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections[:, :4], boost_detections[:, :4]) - np.eye(
                    len(boost_detections), dtype=np.float32
                )
                bdiou_max = bdiou.max(axis=1)
                remaining = boost_indices[bdiou_max <= iou_limit]
                overlapping = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for box_idx in overlapping:
                    tmp = np.argwhere(bdiou[box_idx] > iou_limit).reshape((-1,))
                    args_tmp = np.append(
                        np.intersect1d(boost_indices[overlapping], boost_indices[tmp]),
                        boost_indices[box_idx],
                    )
                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_indices[box_idx], 4] == conf_max:
                        remaining = np.asarray(list(remaining) + [boost_indices[box_idx]])
                keep_mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                keep_mask[remaining] = True
                detections[:, 4] = np.where(
                    keep_mask,
                    self.track_thresh + 1e-4,
                    detections[:, 4],
                )
        return detections

    def dlo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        detections = detections.copy()
        sbiou_matrix = self.get_iou_matrix(detections[:, :4], buffered=True)
        if sbiou_matrix.size == 0:
            return detections
        trackers = np.zeros((len(self.tracks), 6), dtype=np.float32)
        for idx, track in enumerate(self.tracks):
            pos = track.get_state()
            trackers[idx] = [pos[0], pos[1], pos[2], pos[3], 0.0, track.time_since_update - 1]

        if self.use_rich_s:
            mhd_sim = mh_dist_similarity(self.get_mh_dist_matrix(detections), 1.0)
            shape_sim = shape_similarity(detections[:, :4], trackers[:, :4])
            sim_matrix = (mhd_sim + shape_sim + sbiou_matrix) / 3.0
        else:
            sim_matrix = self.get_iou_matrix(detections[:, :4], buffered=False)

        if not self.use_sb and not self.use_vt:
            max_sim = sim_matrix.max(1)
            detections[:, 4] = np.maximum(detections[:, 4], max_sim * self.dlo_boost_coef)
            return detections

        if self.use_sb:
            max_sim = sim_matrix.max(1)
            alpha = 0.65
            detections[:, 4] = np.maximum(
                detections[:, 4],
                alpha * detections[:, 4] + (1 - alpha) * max_sim ** 1.5,
            )
        if self.use_vt:
            threshold_s = 0.95
            threshold_e = 0.8
            n_steps = 20
            alpha = (threshold_s - threshold_e) / n_steps
            dynamic_threshold = np.maximum(threshold_s - trackers[:, 5] * alpha, threshold_e)
            tmp = (sim_matrix > dynamic_threshold).max(1)
            scores = deepcopy(detections[:, 4])
            scores[tmp] = np.maximum(scores[tmp], self.track_thresh + 1e-5)
            detections[:, 4] = scores
        return detections

    def associate(
        self,
        detections: np.ndarray,
        track_confidences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tracker_boxes = self.get_tracker_boxes()
        iou_matrix = iou_batch(detections[:, :4], tracker_boxes[:, :4])
        cost_matrix = deepcopy(iou_matrix)
        det_scores = detections[:, 4]
        conf = det_scores.reshape((-1, 1)) * track_confidences.reshape((1, -1))
        conf[iou_matrix < self.iou_threshold] = 0.0
        cost_matrix += self.lambda_iou * conf * iou_batch(detections[:, :4], tracker_boxes[:, :4])

        mahalanobis_distance = self.get_mh_dist_matrix(detections)
        if mahalanobis_distance.size > 0:
            cost_matrix += self.lambda_mhd * mh_dist_similarity(mahalanobis_distance)
            cost_matrix += self.lambda_shape * conf * shape_similarity(
                detections[:, :4], tracker_boxes[:, :4]
            )
        if self.same_class_only:
            det_labels = detections[:, 5].astype(np.int32)
            trk_labels = np.asarray([track.label for track in self.tracks], dtype=np.int32)
            label_mask = det_labels.reshape((-1, 1)) == trk_labels.reshape((1, -1))
            cost_matrix = np.where(label_mask, cost_matrix, 0.0)
            iou_matrix = np.where(label_mask, iou_matrix, 0.0)
        return linear_assignment(
            detections=detections,
            trackers=tracker_boxes,
            iou_matrix=iou_matrix,
            cost_matrix=cost_matrix,
            threshold=self.iou_threshold,
        )

    def _create_track(self, detection: TrackDetection) -> None:
        bbox = np.asarray(detection.box, dtype=np.float32)
        track = TrackState(
            track_id=self.next_track_id,
            kalman=KalmanFilter(convert_bbox_to_z(bbox)),
            label=detection.label,
            score=detection.score,
            mask=detection.mask,
            box=detection.box,
            center=detection.center,
            area=detection.area,
            label_votes={detection.label: 1},
        )
        self.tracks.append(track)
        self.next_track_id += 1

    def update(
        self,
        masks: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.frame_count += 1
        detections: list[TrackDetection] = []
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
                    center=mask_center(mask, box),
                    area=int(mask.sum()),
                )
            )

        detections_np = np.asarray(
            [
                [det.box[0], det.box[1], det.box[2], det.box[3], det.score, det.label]
                for det in detections
            ],
            dtype=np.float32,
        ) if detections else np.zeros((0, 6), dtype=np.float32)

        for track in self.tracks:
            predicted_box = track.predict()
            track.box = (
                int(round(predicted_box[0])),
                int(round(predicted_box[1])),
                int(round(predicted_box[2])),
                int(round(predicted_box[3])),
            )

        track_confidences = (
            np.asarray([track.get_confidence() for track in self.tracks], dtype=np.float32)
            if self.tracks
            else np.zeros((0,), dtype=np.float32)
        )

        if detections_np.size > 0 and self.use_dlo_boost:
            detections_np = self.dlo_confidence_boost(detections_np)
        if detections_np.size > 0 and self.use_duo_boost:
            detections_np = self.duo_confidence_boost(detections_np)

        if detections_np.size > 0:
            for idx, det in enumerate(detections):
                det.score = float(detections_np[idx, 4])

        remain_indices = [
            idx for idx, det in enumerate(detections) if det.score >= self.track_thresh
        ]
        filtered_detections = [detections[idx] for idx in remain_indices]
        filtered_np = detections_np[remain_indices] if len(remain_indices) else np.zeros((0, 6), dtype=np.float32)

        if len(filtered_detections) > 0 and self.tracks:
            matches, unmatched_dets, unmatched_trks = self.associate(
                detections=filtered_np,
                track_confidences=track_confidences,
            )
            valid_matches = [(int(det_idx), int(trk_idx)) for det_idx, trk_idx in matches]
        else:
            valid_matches = []
            unmatched_dets = list(range(len(filtered_detections)))
            unmatched_trks = list(range(len(self.tracks)))

        for det_idx, trk_idx in valid_matches:
            self.tracks[trk_idx].update(filtered_detections[det_idx])

        for det_idx in unmatched_dets:
            detection = filtered_detections[det_idx]
            if detection.score >= self.track_thresh:
                self._create_track(detection)

        alive_tracks: list[TrackState] = []
        for idx, track in enumerate(self.tracks):
            if idx in unmatched_trks:
                track.mark_missed()
            if track.time_since_update <= self.track_buffer:
                alive_tracks.append(track)
        self.tracks = alive_tracks

        visible_tracks = [
            track
            for track in self.tracks
            if track.time_since_update == 0
            and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
        ]
        visible_tracks.sort(key=lambda item: item.track_id)
        if not visible_tracks:
            empty_masks = np.zeros((0,) + masks.shape[1:], dtype=bool)
            return (
                empty_masks,
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        out_masks = np.asarray([track.mask for track in visible_tracks], dtype=bool)
        out_labels = np.asarray([track.label for track in visible_tracks], dtype=np.int64)
        out_scores = np.asarray([track.score for track in visible_tracks], dtype=np.float32)
        out_track_ids = np.asarray([track.track_id for track in visible_tracks], dtype=np.int64)
        return out_masks, out_labels, out_scores, out_track_ids


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
    text_items: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []
    for mask, label, score, track_id in zip(masks, labels, scores, track_ids):
        label_int = int(label)
        class_name = class_id_to_name(label_int, class_names)
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
        text = f"{class_name} ID:{int(track_id)}"
        text_items.append(((x1, max(0, y1 - 28)), text, color_rgb))

    image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    for (x, y), text, color in text_items:
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(image)[:, :, ::-1]


def infer_video_track_to_path(
    args,
    model,
    class_names: Sequence[str],
    input_video: Path,
    output_path: Path,
    npz_output_path: Path | None = None,
    predict_lock=None,
):
    font = choose_font(args.font_path, args.font_size)
    if npz_output_path is None:
        npz_output_path = build_output_npz_path(output_path)
    tracker = BoostStyleTracker(
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
    dense_outputs_per_frame: list[dict[str, np.ndarray]] = []

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
                tracked_masks, tracked_labels, tracked_scores, track_ids = tracker.update(
                    masks, labels, scores
                )
                dense_outputs_per_frame.append(
                    {
                        "obj_ids": track_ids,
                        "binary_masks": tracked_masks,
                        "class_ids": tracked_labels,
                        "scores": tracked_scores,
                    }
                )
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
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

    save_dense_outputs_to_npz(
        dense_outputs_per_frame=dense_outputs_per_frame,
        output_path=npz_output_path,
        category_names=class_names,
    )


def infer_videos(args, model, class_names: Sequence[str]):
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
            output_video_path = build_output_video_path(video_path, output_video_dir)
            output_npz_path = build_output_npz_path_for_video(video_path, output_npz_dir)
            print(f"提交带跟踪推理: {video_path} -> {output_video_path}")
            future = executor.submit(
                infer_video_track_to_path,
                args,
                model,
                class_names,
                video_path,
                output_video_path,
                output_npz_path,
                predict_lock,
            )
            future_to_video[future] = video_path

        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            future.result()
            print(f"完成带跟踪推理: {video_path}")


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
        infer_video_track_to_path(
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
    infer_videos(args, model, class_names)


if __name__ == "__main__":
    main()
