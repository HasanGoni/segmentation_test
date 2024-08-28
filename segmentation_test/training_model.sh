#!/bin/bash

#INIT_MODEL_PATH="/home/ai_sintercra.work/data/projects/easy_pin_detection/Current_model/tf_model"
INIT_MODEL_PATH="/home/hasan/Schreibtisch/projects/Infineon/projects/part2/2023_easy_pin_detection/model"
#INIT_MODEL_PATH="/home/hasan/Schreibtisch/projects/part2/clean_trn_data_wo2B
INITIAL_MODEL_NAME="time_214629_val_frGrnd0.9532_epoch_191.H5"

# where to save current model
MODEL_PATH="/home/hasan/Schreibtisch/projects/Infineon/projects/part2/models"
MODEL_NAME="tsting"

# image and mask path
IMAGE_PATH="/home/hasan/Schreibtisch/projects/Infineon/projects/part2/clean_trn_data_wo2B/images"
MASK_PATH="/home/hasan/Schreibtisch/projects/Infineon/projects/part2/clean_trn_data_wo2B/masks"
BATCH_SIZE=3
IMAGE_HEIGHT=1152
IMAGE_WIDTH=1632
EPOCH=200


#############################################################
######################## GPU ################################
# -- Based on situtation ucomment the below lines and replace
# -- with bsub commnd
# -- Try with first line if it is not working then try with 
# -- current setup(24G), if still no gpu is available then
# -- try with second bsub command but BATCH_SIZE should be less
# -- than 6, otherwise memory error will be there
#############################################################

# best for training new gpus, but it is difficult to get those gpus
#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA30" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 10 python training_model.py \

# telling without gpus, but actually without 24G gpus this script will not run, if it is not running and its urgent, then one way is to reduce batch size 
#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 10 python training_model.py \

# Using 24G gpus, but this is old GPUS, so it is also slow but will work
#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmodel=TeslaP40" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 10 python training_model.py \

#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA30" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 10 python training_model.py \

#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 10 python training_model.py \
#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmodel=TeslaP40" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 4 python training_model.py \
#bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA30" -R "osrel==70 && ui==aiml-python" -P ma_ai -n 10 python training_model.py \
python training_model.py \
    --initial_model_name "$INITIAL_MODEL_NAME" \
    --initial_model_path "$INIT_MODEL_PATH" \
    --epoch_number "$EPOCH"\
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --image_path "$IMAGE_PATH" \
    --mask_path "$MASK_PATH" \
    --image_height "$IMAGE_HEIGHT" \
    --image_width "$IMAGE_WIDTH" \
    --batch_size "$BATCH_SIZE" \
    --pretrained \