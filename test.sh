python main.py validate \
    -c configs/dinov3/bladder_neck/instance/eomt_base_640_ft_q32.yaml \
    --ckpt_path eomt/tt008jii/checkpoints/epoch=99-step=6700.ckpt \
    --trainer.devices 1 \
    --data.batch_size 8 \
    --data.num_workers 12 \
    --model.eval_top_k_instances 32 \
    --model.save_predictions_dir /mnt/data/projects/bladder_neck/results/epoch99_preds \
    --model.save_prediction_score_thresh 0.25 \
    --compile_disabled
