#!/bin/bash

# Define variables
BATCH_SIZE=150
EPOCHS=400
ACCUM_ITER=1
MODEL='mae_vit_base_patch16'
INPUT_SIZE=224
MASK_RATIO=0.75
NORM_PIX_LOSS
WEIGHT_DECAY=0.05
LR=7.8125e-06
BLR=1e-3
MIN_LR=0.0
WARMUP_EPOCHS=40
DATA_PATH="/home/hasan/.fastai/data/imagewang-160"
OUTPUT_DIR='./output_dir'
LOG_DIR='./output_dir'
DEVICE='cuda'
SEED=0
RESUME=''
START_EPOCH=0
NUM_WORKERS=4
WORLD_SIZE=1
LOCAL_RANK=-1
DIST_URL='env://'

## Call the Python script with the variables
#python pretraining_training.py \
    #--batch_size $BATCH_SIZE \
    #--epochs $EPOCHS \
    #--accum_iter $ACCUM_ITER \
    #--model $MODEL \
    #--input_size $INPUT_SIZE \
    #--mask_ratio $MASK_RATIO \
    #--norm_pix_loss $NORM_PIX_LOSS \
    #--weight_decay $WEIGHT_DECAY \
    #--lr $LR \
    #--blr $BLR \
    #--min_lr $MIN_LR \
    #--warmup_epochs $WARMUP_EPOCHS \
    #--data_path $DATA_PATH \
    #--output_dir $OUTPUT_DIR \
    #--log_dir $LOG_DIR \
    #--device $DEVICE \
    #--seed $SEED \
    ##--resume $RESUME \
    #--start_epoch $START_EPOCH \
    #--num_workers $NUM_WORKERS \
    #--pin_mem $PIN_MEM \
    #--world_size $WORLD_SIZE \
    #--local_rank $LOCAL_RANK \
    #--dist_on_itp $DIST_ON_ITP \
    #--dist_url $DIST_URL

python pretraining_training.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --accum_iter $ACCUM_ITER \
    --model $MODEL \
    --input_size $INPUT_SIZE \
    --mask_ratio $MASK_RATIO \
    --norm_pix_loss \
    --weight_decay $WEIGHT_DECAY \
    --lr $LR \
    --blr $BLR \
    --min_lr $MIN_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --device $DEVICE \
    --seed $SEED \
    --start_epoch $START_EPOCH \
    --num_workers $NUM_WORKERS \
    --pin_mem \
    --world_size $WORLD_SIZE \
    --local_rank $LOCAL_RANK 