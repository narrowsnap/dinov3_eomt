python tools/video_instance_infer.py \
    --config configs/dinov3/bladder_neck/instance/eomt_base_640_ft.yaml \
    --ckpt eomt/o8gimh1d/checkpoints/epoch=11-step=1608.ckpt \
    --input-video /mnt/data/projects/bladder_neck/videos/2vW38wO4dz_00-40-50_00-41-30.mp4 \
    --output-video /mnt/data/projects/bladder_neck/videos/2vW38wO4dz_00-40-50_00-41-30_output.mp4 \
    --class-names prostate cutting_area bladder \
    --device cuda:0

    