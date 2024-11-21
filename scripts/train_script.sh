#!/bin/bash

echo "======================================"
echo "Starting training with parameters:"
echo "======================================"
echo "Hardware Configuration:"
echo "Number of GPUs: 4"
echo "Memory: 250GB"
echo "CPU Cores: 20"
echo ""
echo "Training Configuration:"
echo "Batch Size: 64"
echo "Epochs: 7"
echo "Learning Rate: 1.8748186634898305e-05"
echo "Weight Decay: 4.5269353639276774e-05"
echo "Warmup Steps: 1430"
echo "Workers: 4"
echo "Precision: amp"
echo "Model: Clip_ViT-L-14"
echo "Delta-i: 1.2233"
echo "Delta-t: 0.615"
echo "Loss: 2"
echo "Gradient Clip: 1.0"
echo ""
echo "======================================"
echo "Beginning finetuning"
echo "======================================"

# Set a random port to avoid conflicts
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)

python -m torch.distributed.run \
   --nproc_per_node=4 \
   --master_port=$MASTER_PORT \
   -m training.main \
   --batch-size=64 \
   --epochs=7 \
   --lr=1.8748186634898305e-05 \
   --wd=4.5269353639276774e-05 \
   --warmup=1430 \
   --workers=4 \
   --precision=amp \
   --model=Clip_ViT-L-14 \
   --dataset-mode=croco_d \
   --delta-i=1.2233 \
   --delta-t=0.615 \
   --loss=2 \
   --norm_gradient_clip=1.0 \
   --train-images-for-croco="dynamic_croco_images/train2017" \
   --train-captions-for-croco="croco_dataset/croco_dynamic_train.json" \
   --train-images-for-croco-d="croco_d_images/train2017" \
   --train-captions-for-croco-d="croco_dataset/croco_d_train.json" \
   --train-images-for-stative="stative_croco_images/train2017" \
   --train-captions-for-stative="croco_dataset/croco_stative_train.json"  \
   --val-images-for-croco="dynamic_croco_images/test2017" \
   --val-captions-for-croco="croco_dataset/croco_dynamic_test.json" \
   --val-images-for-croco-d="croco_d_images/test2017" \
   --val-captions-for-croco-d="croco_dataset/croco_d_test.json" \
   --val-images-for-stative="stative_croco_images/test2017" \
   --val-captions-for-stative="croco_dataset/croco_stative_test.json" \
   --val-images-for-vg-relation="vg_relation_images" \
   --val-captions-for-vg-relation="croco_dataset/vg_relation.json"

