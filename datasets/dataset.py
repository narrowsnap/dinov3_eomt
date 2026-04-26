# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import re
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional
from typing import Tuple
import torch
import warnings
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class DirectoryInfo:
    def __init__(self, filename: str):
        self.filename = filename

    def is_dir(self):
        return False


class DirectoryArchive:
    def __init__(self, root: Path):
        self.root = root

    def open(self, member: str, mode: str = "r"):
        if mode != "r":
            raise ValueError(f"Unsupported mode for DirectoryArchive: {mode}")
        return (self.root / member).open("rb")

    def namelist(self):
        return [
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        ]

    def infolist(self):
        return [DirectoryInfo(filename) for filename in self.namelist()]

    def close(self):
        return None


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zip_path: Path,
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_zip_path: Optional[Path] = None,
        target_zip_path_in_zip: Optional[Path] = None,
        target_instance_zip_path: Optional[Path] = None,
        img_folder_path_in_zip: Path = Path("./"),
        target_folder_path_in_zip: Path = Path("./"),
        target_instance_folder_path_in_zip: Path = Path("./"),
        annotations_json_path_in_zip: Optional[Path] = None,
        item_retry_times: int = 8,
    ):
        self.zip_path = zip_path
        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_zip_path = target_zip_path
        self.target_zip_path_in_zip = target_zip_path_in_zip
        self.target_instance_zip_path = target_instance_zip_path
        self.target_folder_path_in_zip = target_folder_path_in_zip
        self.target_instance_folder_path_in_zip = target_instance_folder_path_in_zip
        self.item_retry_times = item_retry_times

        self.zip = None
        self.target_zip = None
        self.target_instance_zip = None
        img_zip, target_zip, target_instance_zip = self._load_zips()

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}
        self.bboxes_by_id = {}

        if annotations_json_path_in_zip is not None:
            source_path = target_zip_path or zip_path
            if source_path.is_dir():
                with (source_path / annotations_json_path_in_zip).open(
                    "r", encoding="utf-8"
                ) as file:
                    annotation_data = json.load(file)
            else:
                with zipfile.ZipFile(source_path) as outer_target_zip:
                    with outer_target_zip.open(
                        str(annotations_json_path_in_zip), "r"
                    ) as file:
                        annotation_data = json.load(file)

            image_id_to_file_name = {
                image["id"]: image["file_name"] for image in annotation_data["images"]
            }

            for annotation in annotation_data["annotations"]:
                img_filename = image_id_to_file_name[annotation["image_id"]]

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}

                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}

                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}

                    if img_filename not in self.bboxes_by_id:
                        self.bboxes_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation[
                        "category_id"
                    ]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation[
                        "segmentation"
                    ]
                    self.bboxes_by_id[img_filename][annotation["id"]] = annotation.get(
                        "bbox"
                    )
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(
                        annotation["iscrowd"]
                    )

        self.imgs = []
        self.annotation_keys = []
        self.targets = []
        self.targets_instance = []

        target_zip_filenames = target_zip.namelist()

        for img_info in sorted(img_zip.infolist(), key=self._sort_key):
            if not self.valid_member(
                img_info, img_folder_path_in_zip, img_stem_suffix, img_suffix
            ):
                continue

            img_path = Path(img_info.filename)
            annotation_key = self._annotation_key(img_path, img_folder_path_in_zip)
            if not only_annotations_json:
                rel_path = img_path.relative_to(img_folder_path_in_zip)
                target_parent = target_folder_path_in_zip / rel_path.parent
                target_stem = rel_path.stem.replace(img_stem_suffix, target_stem_suffix)

                target_filename = (target_parent / f"{target_stem}{target_suffix}").as_posix()

            if self.labels_by_id:
                if annotation_key not in self.labels_by_id:
                    continue

                if not self.labels_by_id[annotation_key]:
                    continue
            else:
                if target_filename not in target_zip_filenames:
                    continue

                if check_empty_targets:
                    with target_zip.open(target_filename) as target_file:
                        min_val, max_val = Image.open(target_file).getextrema()
                        if min_val == max_val:
                            continue

            if target_instance_zip is not None:
                target_instance_filename = (
                    target_instance_folder_path_in_zip / (target_stem + target_suffix)
                ).as_posix()

                if check_empty_targets:
                    with target_instance_zip.open(
                        target_instance_filename
                    ) as target_instance:
                        extrema = Image.open(target_instance).getextrema()
                        if all(min_val == max_val for min_val, max_val in extrema):
                            _, labels, _ = self.target_parser(
                                target=tv_tensors.Mask(
                                    Image.open(target_zip.open(target_filename))
                                ),
                                target_instance=tv_tensors.Mask(
                                    Image.open(target_instance)
                                ),
                                stuff_classes=self.stuff_classes,
                            )
                            if not labels:
                                continue

            self.imgs.append(img_path.as_posix())
            self.annotation_keys.append(annotation_key)

            if not only_annotations_json:
                self.targets.append(target_filename)

            if target_instance_zip is not None:
                self.targets_instance.append(target_instance_filename)

    def _get_item_impl(self, index: int):
        img_zip, target_zip, target_instance_zip = self._load_zips()

        with img_zip.open(self.imgs[index]) as img:
            img = tv_tensors.Image(Image.open(img).convert("RGB"))

        target = None
        if not self.only_annotations_json:
            with target_zip.open(self.targets[index]) as target_file:
                target = tv_tensors.Mask(Image.open(target_file), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        target_instance = None
        if self.targets_instance:
            with target_instance_zip.open(
                self.targets_instance[index]
            ) as target_instance:
                target_instance = tv_tensors.Mask(
                    Image.open(target_instance), dtype=torch.long
                )

        masks, labels, is_crowd = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(self.annotation_keys[index], {}),
            labels_by_id=self.labels_by_id.get(self.annotation_keys[index], {}),
            is_crowd_by_id=self.is_crowd_by_id.get(self.annotation_keys[index], {}),
            bboxes_by_id=self.bboxes_by_id.get(self.annotation_keys[index], {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )
        if not masks:
            raise RuntimeError(
                f"样本没有可用实例，请检查标注解析或类别映射: {self.imgs[index]}"
            )

        masks = [
            self._resize_mask_to_shape(mask, img.shape[-2:])
            if tuple(mask.shape[-2:]) != tuple(img.shape[-2:])
            else mask
            for mask in masks
        ]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target["file_name"] = Path(self.imgs[index]).name
        target["image_path"] = self.imgs[index]

        return img, target

    def __getitem__(self, index: int):
        last_error = None
        tried_indices = {index}

        for attempt in range(self.item_retry_times + 1):
            try:
                return self._get_item_impl(index)
            except Exception as exc:
                last_error = exc
                if attempt >= self.item_retry_times or len(tried_indices) >= len(self.imgs):
                    break

                next_index = index
                while next_index in tried_indices:
                    next_index = int(torch.randint(len(self.imgs), size=(1,)).item())
                tried_indices.add(next_index)
                warnings.warn(
                    f"读取样本失败，改为随机重试: {self.imgs[index]} -> {self.imgs[next_index]} ({exc})",
                    RuntimeWarning,
                )
                index = next_index

        raise RuntimeError(
            f"样本重试失败，最后一次索引: {index}, 原始错误: {last_error}"
        ) from last_error

    @staticmethod
    def _annotation_key(
        img_path: Path, img_folder_path_in_zip: Optional[Path] = None
    ) -> str:
        if img_folder_path_in_zip is not None:
            rel_path = img_path.relative_to(img_folder_path_in_zip)
            return rel_path.as_posix()
        return img_path.as_posix()

    @staticmethod
    def _resize_mask_to_shape(
        mask: tv_tensors.Mask, shape: tuple[int, int]
    ) -> tv_tensors.Mask:
        resized = F.resize(
            mask,
            list(shape),
            interpolation=F.InterpolationMode.NEAREST,
        )
        return tv_tensors.Mask(resized, dtype=torch.bool)

    def _load_zips(
        self,
    ) -> Tuple[zipfile.ZipFile, zipfile.ZipFile, Optional[zipfile.ZipFile]]:
        worker = get_worker_info()
        worker = worker.id if worker else None

        if self.zip is None:
            self.zip = {}
        if self.target_zip is None:
            self.target_zip = {}
        if self.target_instance_zip is None and self.target_instance_zip_path:
            self.target_instance_zip = {}

        if worker not in self.zip:
            if self.zip_path.is_dir():
                self.zip[worker] = DirectoryArchive(self.zip_path)
            else:
                self.zip[worker] = zipfile.ZipFile(self.zip_path)
        if worker not in self.target_zip:
            if self.target_zip_path:
                if self.target_zip_path.is_dir():
                    if self.target_zip_path_in_zip:
                        raise ValueError(
                            "Directory target sources do not support target_zip_path_in_zip"
                        )
                    self.target_zip[worker] = DirectoryArchive(self.target_zip_path)
                else:
                    self.target_zip[worker] = zipfile.ZipFile(self.target_zip_path)
                if self.target_zip_path_in_zip:
                    with self.target_zip[worker].open(
                        str(self.target_zip_path_in_zip)
                    ) as target_zip_stream:
                        nested_zip_data = BytesIO(target_zip_stream.read())
                    self.target_zip[worker].close()
                    self.target_zip[worker] = zipfile.ZipFile(nested_zip_data)
            else:
                self.target_zip[worker] = self.zip[worker]
        if (
            self.target_instance_zip_path is not None
            and worker not in self.target_instance_zip
        ):
            if self.target_instance_zip_path.is_dir():
                self.target_instance_zip[worker] = DirectoryArchive(
                    self.target_instance_zip_path
                )
            else:
                self.target_instance_zip[worker] = zipfile.ZipFile(
                    self.target_instance_zip_path
                )

        return (
            self.zip[worker],
            self.target_zip[worker],
            self.target_instance_zip[worker] if self.target_instance_zip_path else None,
        )

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        match = re.search(r"\d+", m.filename)

        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def valid_member(
        img_info: zipfile.ZipInfo,
        img_folder_path_in_zip: Path,
        img_stem_suffix: str,
        img_suffix: str,
    ):
        return (
            Path(img_info.filename).is_relative_to(img_folder_path_in_zip)
            and img_info.filename.endswith(img_stem_suffix + img_suffix)
            and not img_info.is_dir()
        )

    def __len__(self):
        return len(self.imgs)

    def close(self):
        if self.zip is not None:
            for item in self.zip.values():
                item.close()
            self.zip = None

        if self.target_zip is not None:
            for item in self.target_zip.values():
                item.close()
            self.target_zip = None

        if self.target_instance_zip is not None:
            for item in self.target_instance_zip.values():
                item.close()
            self.target_instance_zip = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["zip"] = None
        state["target_zip"] = None
        state["target_instance_zip"] = None
        return state
