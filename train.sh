python main.py fit \
    -c configs/dinov3/wuye/instance/eomt_base_ft.yaml \
    --trainer.devices 4 \
    --data.batch_size 8 \
    --data.num_workers 8