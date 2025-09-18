
#coco
python train.py --dataset coco --gpu-id 0 --logger_name runs/coco/coco_swin_7_1 --num_epochs 30 --batch_size 64 --learning_rate 5e-4 --vit_type swin --embed_size 512 --lr_schedules [9, 15, 20, 25 ] --decay_rate 0.3 --workers 4