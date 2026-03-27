python main.py fit \
    -c configs/dinov3/bladder_neck/instance/eomt_base_640_ft_q32.yaml \
    --trainer.devices 4 \
    --data.batch_size 8 \
    --data.num_workers 12