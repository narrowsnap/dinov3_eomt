# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import tv_tensors
from torch.utils.data import DataLoader

from datasets.coco_instance import COCOInstance
from datasets.dataset import Dataset


class COCOInstanceLocal(COCOInstance):
    def __init__(
        self,
        *args,
        train_image_dir: str = "train",
        val_image_dir: str = "val",
        train_annotations: str = "annotations/instances_train.json",
        val_annotations: str = "annotations/instances_val.json",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_image_dir = train_image_dir
        self.val_image_dir = val_image_dir
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.class_mapping = {
            class_id: class_id - 1 for class_id in range(1, self.num_classes + 1)
        }

    @staticmethod
    def bbox_to_mask(bbox: list[float], width: int, height: int) -> tv_tensors.Mask:
        x, y, w, h = [float(v) for v in bbox]
        x1 = max(0, min(int(round(x)), width))
        y1 = max(0, min(int(round(y)), height))
        x2 = max(x1, min(int(round(x + w)), width))
        y2 = max(y1, min(int(round(y + h)), height))

        mask = Image.new("L", (width, height), 0)
        if x2 > x1 and y2 > y1:
            ImageDraw.Draw(mask).rectangle((x1, y1, x2, y2), outline=1, fill=1)
        return tv_tensors.Mask(torch.from_numpy(np.array(mask)).bool())

    def target_parser(
        self,
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        bboxes_by_id: dict[int, list[float]] | None = None,
        width: int = 0,
        height: int = 0,
        **kwargs,
    ):
        masks, labels, is_crowd = [], [], []
        bboxes_by_id = bboxes_by_id or {}

        for label_id, cls_id in labels_by_id.items():
            mapped_cls = self.class_mapping.get(cls_id)
            if mapped_cls is None:
                continue

            segmentation = polygons_by_id.get(label_id)
            if segmentation:
                if isinstance(segmentation, list):
                    masks.append(self.polygons_to_mask(segmentation, width, height))
                else:
                    rles = self._rle_from_segmentation(segmentation, height, width)
                    masks.append(tv_tensors.Mask(rles, dtype=torch.bool))
            else:
                bbox = bboxes_by_id.get(label_id)
                if not bbox or len(bbox) != 4:
                    continue
                masks.append(self.bbox_to_mask(bbox, width, height))

            labels.append(mapped_cls)
            is_crowd.append(is_crowd_by_id.get(label_id, False))

        return masks, labels, is_crowd

    @staticmethod
    def _rle_from_segmentation(segmentation, height: int, width: int):
        from pycocotools import mask as coco_mask

        if isinstance(segmentation, dict):
            return coco_mask.decode(segmentation)

        if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], dict):
            rle = coco_mask.merge(segmentation)
            return coco_mask.decode(rle)

        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
        return coco_mask.decode(rle)

    def setup(self, stage: Union[str, None] = None) -> COCOInstance:
        root = Path(self.path)
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_parser": self.target_parser,
            "only_annotations_json": True,
            "check_empty_targets": self.check_empty_targets,
        }

        self.train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path(f"./{self.train_image_dir}"),
            annotations_json_path_in_zip=Path(f"./{self.train_annotations}"),
            target_zip_path=root,
            zip_path=root,
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path(f"./{self.val_image_dir}"),
            annotations_json_path_in_zip=Path(f"./{self.val_annotations}"),
            target_zip_path=root,
            zip_path=root,
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )
