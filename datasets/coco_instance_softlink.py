# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union

import torch
from torchvision import tv_tensors
from torch.utils.data import DataLoader

from datasets.coco_instance import COCOInstance
from datasets.dataset import Dataset
from datasets.clip_sampler import (
    ClipFractionSampler,
    DistributedClipFractionSampler,
)


class COCOInstanceSoftlink(COCOInstance):
    def __init__(
        self,
        *args,
        train_clip_sample_ratio: float = 1.0 / 3.0,
        train_sampler_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_clip_sample_ratio = train_clip_sample_ratio
        self.train_sampler_seed = train_sampler_seed
        self.class_mapping = {class_id: class_id - 1 for class_id in range(1, self.num_classes + 1)}

    def target_parser(
        self,
        polygons_by_id: dict[int, list[list[float]]],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        width: int,
        height: int,
        **kwargs,
    ):
        masks, labels, is_crowd = [], [], []

        for label_id, cls_id in labels_by_id.items():
            mapped_cls = self.class_mapping.get(cls_id)
            if mapped_cls is None:
                continue

            segmentation = polygons_by_id[label_id]
            if isinstance(segmentation, list):
                masks.append(self.polygons_to_mask(segmentation, width, height))
            else:
                rles = self._rle_from_segmentation(segmentation, height, width)
                masks.append(tv_tensors.Mask(rles, dtype=torch.bool))
            labels.append(mapped_cls)
            is_crowd.append(is_crowd_by_id[label_id])

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
            img_folder_path_in_zip=Path("./train"),
            annotations_json_path_in_zip=Path("./train/_annotations.coco.json"),
            target_zip_path=root,
            zip_path=root,
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./valid"),
            annotations_json_path_in_zip=Path("./valid/_annotations.coco.json"),
            target_zip_path=root,
            zip_path=root,
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        if self.trainer is not None and self.trainer.world_size > 1:
            sampler = DistributedClipFractionSampler(
                self.train_dataset,
                sample_ratio=self.train_clip_sample_ratio,
                seed=self.train_sampler_seed,
                drop_last=True,
            )
        else:
            sampler = ClipFractionSampler(
                self.train_dataset,
                sample_ratio=self.train_clip_sample_ratio,
                seed=self.train_sampler_seed,
            )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )
