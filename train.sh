python main.py fit \
    -c configs/dinov3/instruments/instance/eomt_base_640_ft_vh1webla.yaml \
    --trainer.devices 4 \
    --data.batch_size 8 \
    --data.num_workers 8