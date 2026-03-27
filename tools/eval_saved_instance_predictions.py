import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description="根据导出的 instance_map/json 结果计算 COCO segm 指标")
    parser.add_argument("--pred-dir", required=True, help="预测结果目录，包含 json/ 和 instance_map/")
    parser.add_argument(
        "--gt-json",
        default="/mnt/data2/datasets/bladder_neck/coco_instance_v1/annotations/instances_val2017.json",
        help="GT COCO 标注 json",
    )
    parser.add_argument(
        "--save-coco-json",
        default=None,
        help="可选，保存转换后的 COCO prediction json 路径",
    )
    return parser.parse_args()


def encode_binary_mask(mask: np.ndarray):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def convert_predictions(pred_dir: Path, coco_gt: COCO):
    json_dir = pred_dir / "json"
    instance_map_dir = pred_dir / "instance_map"
    if not json_dir.is_dir() or not instance_map_dir.is_dir():
        raise FileNotFoundError(f"{pred_dir} 下缺少 json/ 或 instance_map/ 目录")

    image_name_to_id = {
        image["file_name"]: image["id"] for image in coco_gt.dataset["images"]
    }

    predictions = []
    missing_images = []
    for pred_json_path in sorted(json_dir.glob("*.json")):
        pred_obj = json.loads(pred_json_path.read_text(encoding="utf-8"))
        file_name = pred_obj["file_name"]
        if file_name not in image_name_to_id:
            missing_images.append(file_name)
            continue

        image_id = image_name_to_id[file_name]
        instance_map_path = instance_map_dir / f"{Path(file_name).stem}.png"
        if not instance_map_path.is_file():
            raise FileNotFoundError(f"缺少 instance_map 文件: {instance_map_path}")

        instance_map = np.array(Image.open(instance_map_path))
        for instance in pred_obj.get("instances", []):
            instance_id = int(instance["instance_id"])
            mask = instance_map == instance_id
            if not mask.any():
                continue

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(instance["label"]) + 1,
                    "segmentation": encode_binary_mask(mask),
                    "score": float(instance["score"]),
                }
            )

    if missing_images:
        print(f"警告: {len(missing_images)} 个预测文件在 GT 中找不到对应图片，已跳过")

    return predictions


def summarize_eval(coco_eval: COCOeval, prefix: str):
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    print(
        f"{prefix} AP={stats[0] * 100:.2f} | "
        f"AP50={stats[1] * 100:.2f} | "
        f"AP75={stats[2] * 100:.2f} | "
        f"APm={stats[4] * 100:.2f} | "
        f"APl={stats[5] * 100:.2f}"
    )


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    gt_json = Path(args.gt_json)

    coco_gt = COCO(str(gt_json))
    coco_gt.dataset.setdefault("info", {})
    coco_gt.dataset.setdefault("licenses", [])
    predictions = convert_predictions(pred_dir, coco_gt)
    if not predictions:
        raise ValueError("没有可评估的预测结果")

    if args.save_coco_json:
        save_path = Path(args.save_coco_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(predictions), encoding="utf-8")
        print(f"COCO prediction json 已保存到: {save_path}")

    coco_dt = coco_gt.loadRes(predictions)

    print(f"预测实例数: {len(predictions)}")
    print("整体指标:")
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    summarize_eval(coco_eval, "All")

    print("\n每类指标:")
    for category in coco_gt.dataset["categories"]:
        coco_eval_cat = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval_cat.params.catIds = [category["id"]]
        summarize_eval(coco_eval_cat, f"{category['name']:<16}")


if __name__ == "__main__":
    main()
