python main.py fit \
    -c configs/dinov3/needle_gold_suture/instance/eomt_base_1024_ft.yaml \
    --trainer.devices 4 \
    --data.batch_size 4 \
    --data.num_workers 8
