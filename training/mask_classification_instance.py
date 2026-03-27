# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
# ---------------------------------------------------------------


import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationInstance(LightningModule):
    INSTANCE_COLORS = np.array(
        [
            [231, 76, 60],
            [52, 152, 219],
            [46, 204, 113],
            [241, 196, 15],
            [155, 89, 182],
            [230, 126, 34],
        ],
        dtype=np.uint8,
    )

    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        save_predictions_dir: Optional[str] = None,
        save_prediction_overlay: bool = True,
        save_prediction_label_map: bool = True,
        save_prediction_instance_map: bool = True,
        save_prediction_json: bool = True,
        save_prediction_score_thresh: float = 0.0,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes: List[int] = []
        self.eval_top_k_instances = eval_top_k_instances
        self.save_predictions_dir = (
            Path(save_predictions_dir) if save_predictions_dir else None
        )
        self.save_prediction_overlay = save_prediction_overlay
        self.save_prediction_label_map = save_prediction_label_map
        self.save_prediction_instance_map = save_prediction_instance_map
        self.save_prediction_json = save_prediction_json
        self.save_prediction_score_thresh = save_prediction_score_thresh

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_instance(self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)
        self._reset_custom_val_metrics()

    def _reset_custom_val_metrics(self):
        self._val_final_block_preds = []
        self._val_final_block_targets = []
        self._val_dice_intersections = torch.zeros(self.num_classes, dtype=torch.float64)
        self._val_dice_pred_areas = torch.zeros(self.num_classes, dtype=torch.float64)
        self._val_dice_target_areas = torch.zeros(self.num_classes, dtype=torch.float64)

    @staticmethod
    def _encode_mask(mask: np.ndarray):
        rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def _update_dice_state(self, preds, targets):
        for pred, target in zip(preds, targets):
            if pred["masks"].numel() > 0:
                pred_masks = pred["masks"].detach().cpu()
            else:
                pred_masks = torch.zeros(
                    (0, *target["masks"].shape[-2:]), dtype=torch.bool
                )
            pred_labels = pred["labels"].detach().cpu()
            target_masks = target["masks"].detach().cpu()
            target_labels = target["labels"].detach().cpu()

            reference_shape = (
                tuple(target_masks.shape[-2:])
                if target_masks.numel() > 0
                else tuple(pred_masks.shape[-2:])
            )

            for class_idx in range(self.num_classes):
                pred_keep = pred_labels == class_idx
                target_keep = target_labels == class_idx
                pred_class_mask = (
                    pred_masks[pred_keep].any(dim=0)
                    if pred_keep.any()
                    else torch.zeros(reference_shape, dtype=torch.bool)
                )
                target_class_mask = (
                    target_masks[target_keep].any(dim=0)
                    if target_keep.any()
                    else torch.zeros(reference_shape, dtype=torch.bool)
                )

                self._val_dice_intersections[class_idx] += torch.logical_and(
                    pred_class_mask, target_class_mask
                ).sum().item()
                self._val_dice_pred_areas[class_idx] += pred_class_mask.sum().item()
                self._val_dice_target_areas[class_idx] += target_class_mask.sum().item()

    def _store_final_block_predictions(self, preds, targets):
        self._update_dice_state(preds, targets)

        for sample_idx, (pred, target) in enumerate(zip(preds, targets)):
            file_name = target.get(
                "file_name",
                f"sample_{len(self._val_final_block_preds) + sample_idx}.jpg",
            )
            self._val_final_block_targets.append(
                {
                    "file_name": file_name,
                    "masks": target["masks"].detach().cpu().numpy().astype(bool),
                    "labels": target["labels"].detach().cpu().numpy(),
                    "iscrowd": target["is_crowd"].detach().cpu().numpy(),
                }
            )
            self._val_final_block_preds.append(
                {
                    "file_name": file_name,
                    "masks": pred["masks"].detach().cpu().numpy().astype(bool),
                    "labels": pred["labels"].detach().cpu().numpy(),
                    "scores": pred["scores"].detach().cpu().numpy(),
                }
            )

    def _compute_ap50_per_class(self):
        if not self._val_final_block_preds:
            return {}

        images = []
        gt_annotations = []
        pred_annotations = []
        ann_id = 1
        image_name_to_id = {}

        for image_id, target in enumerate(self._val_final_block_targets, start=1):
            image_name_to_id[target["file_name"]] = image_id
            height, width = target["masks"].shape[-2:]
            images.append(
                {
                    "id": image_id,
                    "file_name": target["file_name"],
                    "height": int(height),
                    "width": int(width),
                }
            )
            for mask, label, iscrowd in zip(
                target["masks"], target["labels"], target["iscrowd"]
            ):
                encoded_mask = self._encode_mask(mask)
                gt_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "segmentation": encoded_mask,
                        "area": int(mask.sum()),
                        "bbox": list(coco_mask.toBbox(encoded_mask)),
                        "iscrowd": int(iscrowd),
                    }
                )
                ann_id += 1

        for pred in self._val_final_block_preds:
            image_id = image_name_to_id[pred["file_name"]]
            for mask, label, score in zip(
                pred["masks"], pred["labels"], pred["scores"]
            ):
                pred_annotations.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "segmentation": self._encode_mask(mask),
                        "score": float(score),
                    }
                )

        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": gt_annotations,
            "categories": [
                {"id": class_idx + 1, "name": f"class_{class_idx}"}
                for class_idx in range(self.num_classes)
            ],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(pred_annotations)

        ap50_per_class = {}
        for class_idx in range(self.num_classes):
            coco_eval = COCOeval(coco_gt, coco_dt, "segm")
            coco_eval.params.catIds = [class_idx + 1]
            with redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
            precision = coco_eval.eval["precision"]
            precision_iou_50 = precision[0, :, 0, 0, -1]
            precision_iou_50 = precision_iou_50[precision_iou_50 > -1]
            ap50_per_class[class_idx] = (
                float(precision_iou_50.mean()) if precision_iou_50.size > 0 else -1.0
            )

        return ap50_per_class

    def _compute_dice_per_class(self):
        dice_per_class = {}
        for class_idx in range(self.num_classes):
            denom = (
                self._val_dice_pred_areas[class_idx]
                + self._val_dice_target_areas[class_idx]
            )
            if denom == 0:
                dice = 1.0
            else:
                dice = float(2.0 * self._val_dice_intersections[class_idx] / denom)
            dice_per_class[class_idx] = dice
        return dice_per_class

    def on_validation_epoch_start(self):
        self._reset_custom_val_metrics()

    def _should_save_predictions(self, log_prefix, block_idx, num_blocks):
        return (
            self.save_predictions_dir is not None
            and log_prefix == "val"
            and not self.trainer.sanity_checking
            and block_idx == num_blocks - 1
        )

    def _export_predictions(self, imgs, preds, targets):
        assert self.save_predictions_dir is not None

        output_root = self.save_predictions_dir
        (output_root / "overlay").mkdir(parents=True, exist_ok=True)
        (output_root / "label_map").mkdir(parents=True, exist_ok=True)
        (output_root / "instance_map").mkdir(parents=True, exist_ok=True)
        (output_root / "json").mkdir(parents=True, exist_ok=True)

        for img, pred, target in zip(imgs, preds, targets):
            file_name = str(target.get("file_name", "unknown.png"))
            stem = Path(file_name).stem

            img_np = img.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            masks = pred["masks"].detach().cpu().numpy().astype(bool)
            labels = pred["labels"].detach().cpu().numpy()
            scores = pred["scores"].detach().cpu().numpy()
            gt_masks = target["masks"].detach().cpu().numpy().astype(bool)
            gt_labels = target["labels"].detach().cpu().numpy()

            keep = scores >= self.save_prediction_score_thresh
            masks = masks[keep]
            labels = labels[keep]
            scores = scores[keep]

            order = np.argsort(-scores)
            masks = masks[order]
            labels = labels[order]
            scores = scores[order]

            overlay = img_np.copy()
            gt_overlay = img_np.copy()
            label_map = np.zeros(img_np.shape[:2], dtype=np.uint8)
            instance_map = np.zeros(img_np.shape[:2], dtype=np.uint16)
            instances = []

            for gt_mask, gt_label in zip(gt_masks, gt_labels):
                color = self.INSTANCE_COLORS[int(gt_label) % len(self.INSTANCE_COLORS)]
                gt_overlay[gt_mask] = (
                    0.6 * gt_overlay[gt_mask].astype(np.float32)
                    + 0.4 * color.astype(np.float32)
                ).astype(np.uint8)

            for instance_idx, (mask, label, score) in enumerate(
                zip(masks, labels, scores), start=1
            ):
                color = self.INSTANCE_COLORS[int(label) % len(self.INSTANCE_COLORS)]
                overlay[mask] = (
                    0.6 * overlay[mask].astype(np.float32)
                    + 0.4 * color.astype(np.float32)
                ).astype(np.uint8)
                label_map[mask] = int(label) + 1
                instance_map[mask] = instance_idx
                instances.append(
                    {
                        "instance_id": instance_idx,
                        "label": int(label),
                        "score": float(score),
                        "area": int(mask.sum()),
                    }
                )

            if self.save_prediction_overlay:
                panel = np.concatenate([img_np, gt_overlay, overlay], axis=1)
                Image.fromarray(panel).save(output_root / "overlay" / f"{stem}.png")

            if self.save_prediction_label_map:
                Image.fromarray(label_map).save(output_root / "label_map" / f"{stem}.png")

            if self.save_prediction_instance_map:
                Image.fromarray(instance_map).save(
                    output_root / "instance_map" / f"{stem}.png"
                )

            if self.save_prediction_json:
                (output_root / "json" / f"{stem}.json").write_text(
                    json.dumps(
                        {"file_name": file_name, "instances": instances},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                max_candidates = scores.numel()
                if max_candidates == 0:
                    preds.append(
                        dict(
                            masks=torch.zeros(
                                (0, *mask_logits[j].shape[-2:]),
                                dtype=torch.bool,
                                device=mask_logits[j].device,
                            ),
                            labels=torch.zeros(
                                (0,), dtype=torch.long, device=mask_logits[j].device
                            ),
                            scores=torch.zeros(
                                (0,),
                                dtype=mask_logits[j].dtype,
                                device=mask_logits[j].device,
                            ),
                        )
                    )
                    targets_.append(
                        dict(
                            masks=targets[j]["masks"],
                            labels=targets[j]["labels"],
                            iscrowd=targets[j]["is_crowd"],
                            is_crowd=targets[j]["is_crowd"],
                            file_name=targets[j].get("file_name"),
                        )
                    )
                    continue

                topk_k = min(self.eval_top_k_instances, max_candidates)
                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    topk_k, sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                mask_logits[j] = mask_logits[j][topk_indices]

                masks = mask_logits[j] > 0
                mask_scores = (
                    mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                preds.append(
                    dict(
                        masks=masks,
                        labels=labels,
                        scores=scores,
                    )
                )
                targets_.append(
                    dict(
                        masks=targets[j]["masks"],
                        labels=targets[j]["labels"],
                        iscrowd=targets[j]["is_crowd"],
                        is_crowd=targets[j]["is_crowd"],
                        file_name=targets[j].get("file_name"),
                    )
                )

            if log_prefix == "val" and i == len(mask_logits_per_layer) - 1:
                self._store_final_block_predictions(preds, targets_)

            if self._should_save_predictions(log_prefix, i, len(mask_logits_per_layer)):
                self._export_predictions(imgs, preds, targets)

            self.update_metrics_instance(preds, targets, i)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            for class_idx, ap50 in self._compute_ap50_per_class().items():
                self.log(f"metrics/val_ap_50_class_{class_idx}", ap50)
            for class_idx, dice in self._compute_dice_per_class().items():
                self.log(f"metrics/val_dice_class_{class_idx}", dice)
        self._on_eval_epoch_end_instance("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")
