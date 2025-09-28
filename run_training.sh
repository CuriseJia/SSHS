#!/bin/bash
# CochAV训练启动脚本

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # 注释掉，让程序自动处理
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 训练参数
TRAIN_CONFIG="AudioCOCO/config1.json"
VAL_CONFIG="AudioCOCO/config1.json"  # 可以分割为训练/验证集
IMAGE_ROOT="/path/to/coco/images"     # 请修改为实际路径
AUDIO_ROOT="/path/to/audio/coco"      # 请修改为实际路径
CHECKPOINT_DIR="./checkpoints/cochav_$(date +%Y%m%d_%H%M%S)"

# 创建检查点目录
mkdir -p $CHECKPOINT_DIR

# GPU选择（可选）
# 使用所有可用GPU（默认）
GPU_OPTIONS=""
# 使用指定GPU，例如: GPU_OPTIONS="--gpu_ids 0,1,2,3"
# 使用单GPU，例如: GPU_OPTIONS="--gpu_ids 0"
# 强制使用CPU，例如: GPU_OPTIONS="--force_cpu"

# 启动训练
python train.py \
    --train_config $TRAIN_CONFIG \
    --val_config $VAL_CONFIG \
    --image_root $IMAGE_ROOT \
    --audio_root $AUDIO_ROOT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --coch_config default \
    --img_size 224 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --accumulation_steps 2 \
    --use_amp \
    --scheduler cosine \
    --num_workers 8 \
    --log_interval 50 \
    --save_interval 10 \
    --experiment_name "cochav_audiococo" \
    --epsilon 0.65 \
    --epsilon2 0.4 \
    --tri_map \
    --Neg \
    $GPU_OPTIONS
