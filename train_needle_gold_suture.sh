PYTHON_BIN="${PYTHON_BIN:-/home/zhouyang/anaconda3/envs/eomt/bin/python}"

"${PYTHON_BIN}" main.py fit \
    -c configs/dinov3/needle_gold_suture/instance/eomt_base_640_ft.yaml \
    --trainer.devices 4 \
    --data.batch_size 8 \
    --data.num_workers 8
