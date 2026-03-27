# 自定义数据转 COCO Instance

当前仓库的 `datasets.coco_instance.COCOInstance` 直接读取如下三个 zip 文件：

- `train2017.zip`
- `val2017.zip`
- `annotations_trainval2017.zip`

其中：

- `train2017.zip` 内部路径应为 `train2017/<图片名>`
- `val2017.zip` 内部路径应为 `val2017/<图片名>`
- `annotations_trainval2017.zip` 内部路径应为：
  - `annotations/instances_train2017.json`
  - `annotations/instances_val2017.json`

## bladder_neck 数据转换

仓库提供了脚本 [convert_bladder_neck_to_coco_instance.py](/home/zhouyang/work/gitops/eomt/preprocess/convert_bladder_neck_to_coco_instance.py)，用于把如下格式的标注：

```python
{
  "xxx.jpg": {
    "prostate": [[x1, y1], [x2, y2], ...],
    "cutting_area": [[x1, y1], [x2, y2], ...],
    "bladder": [[x1, y1], [x2, y2], ...]
  }
}
```

转换为 COCO instance。

坐标要求：

- 输入坐标是归一化坐标，范围 `[0, 1]`
- 每个类别对应一个多边形
- 脚本会自动读取原图尺寸，并换算成 COCO 所需的像素坐标

## 使用命令

```bash
python preprocess/convert_bladder_neck_to_coco_instance.py \
  --images-dir /mnt/data2/datasets/bladder_neck/images \
  --train-json /mnt/data2/datasets/bladder_neck/annos/v1/train_name2anno_seg.json \
  --val-json /mnt/data2/datasets/bladder_neck/annos/v1/test_name2anno_seg.json \
  --output-dir /mnt/data2/datasets/bladder_neck/coco_instance_v1
```

输出目录会生成：

```text
/mnt/data2/datasets/bladder_neck/coco_instance_v1/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── annotations_trainval2017.zip
├── train2017.zip
└── val2017.zip
```

## 训练接入

生成完成后，可以直接复用现有 COCO instance 数据模块，只需要把类别数改成 `3`：

```bash
python main.py fit \
  -c configs/dinov3/coco/instance/eomt_base_640.yaml \
  --data.path /mnt/data2/datasets/bladder_neck/coco_instance_v1 \
  --data.num_classes 3 \
  --data.batch_size 4 \
  --trainer.devices 1
```

如果只想先验证数据是否能正常读取，也可以先执行：

```bash
python main.py validate \
  -c configs/dinov3/coco/instance/eomt_base_640.yaml \
  --data.path /mnt/data2/datasets/bladder_neck/coco_instance_v1 \
  --data.num_classes 3 \
  --data.batch_size 1 \
  --trainer.devices 1
```
