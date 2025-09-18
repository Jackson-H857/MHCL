## single GPU

### swin + f30k    视觉编码器学习率
    python train.py --dataset f30k --gpu-id 0 --logger_name runs/f30k --num_epochs 30 --batch_size 64 --learning_rate 5e-4 --vit_type swin --embed_size 1024 --warmup 8000 \
    --decay_rate 0.3 --workers 4 \
    --use_moco 1 \
    --moco_M 2048 \
    --loss_lamda 1 \
    --mu 90 \
    --gama 0.5 \
    --moco_r 0.999 \
        

