# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class ClipFractionSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        sample_ratio: float = 1.0 / 3.0,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.sample_ratio = sample_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.clip_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, img_path in enumerate(dataset.imgs):
            clip_name = Path(img_path).parent.name
            self.clip_to_indices[clip_name].append(index)

        self.clips = sorted(self.clip_to_indices.keys())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return generator

    def _sample_indices(self) -> list[int]:
        generator = self._generator()
        sampled_indices: list[int] = []
        clip_order = self.clips
        if self.shuffle:
            perm = torch.randperm(len(self.clips), generator=generator).tolist()
            clip_order = [self.clips[idx] for idx in perm]

        for clip_name in clip_order:
            clip_indices = list(self.clip_to_indices[clip_name])
            if self.shuffle:
                perm = torch.randperm(len(clip_indices), generator=generator).tolist()
                clip_indices = [clip_indices[idx] for idx in perm]
            keep_count = max(1, math.floor(len(clip_indices) * self.sample_ratio))
            sampled_indices.extend(clip_indices[:keep_count])

        return sampled_indices

    def __iter__(self) -> Iterator[int]:
        return iter(self._sample_indices())

    def __len__(self) -> int:
        return sum(
            max(1, math.floor(len(indices) * self.sample_ratio))
            for indices in self.clip_to_indices.values()
        )


class DistributedClipFractionSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        sample_ratio: float = 1.0 / 3.0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        super().__init__(
            dataset=dataset,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )
        self.base_sampler = ClipFractionSampler(
            dataset=dataset,
            sample_ratio=sample_ratio,
            shuffle=shuffle,
            seed=seed,
        )

    def __iter__(self) -> Iterator[int]:
        self.base_sampler.set_epoch(self.epoch)
        indices = self.base_sampler._sample_indices()

        if self.drop_last and len(indices) % self.num_replicas != 0:
            indices = indices[: len(indices) - (len(indices) % self.num_replicas)]
        elif not self.drop_last and len(indices) % self.num_replicas != 0:
            padding_size = self.num_replicas - (len(indices) % self.num_replicas)
            indices += indices[:padding_size]

        self.num_samples = len(indices) // self.num_replicas
        self.total_size = len(indices)
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        base_len = len(self.base_sampler)
        if self.drop_last:
            return base_len // self.num_replicas
        return math.ceil(base_len / self.num_replicas)
