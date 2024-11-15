#! /bin/bash


#--dataset_name nateraw/image-folder \
python mae_pretraining.py \
    --train_dir $HOME/Schreibtisch/projects/data/CurrentTrainingData20240209/images_new \
    --output_dir ./outputs/ \
    --remove_unused_columns False \
    --label_names pixel_values \
    --do_train \
    --do_eval
#python mae_pretraining.py\
    #--dataset_name cifar10 \
    #--output_dir ./vit-mae-demo \
    #--remove_unused_columns False \
    #--label_names pixel_values \
    #--mask_ratio 0.75 \
    #--norm_pix_loss \
    #--do_train \
    #--do_eval \
    #--base_learning_rate 1.5e-4 \
    #--lr_scheduler_type cosine \
    #--weight_decay 0.05 \
    #--num_train_epochs 800 \
    #--warmup_ratio 0.05 \
    #--per_device_train_batch_size 8 \
    #--per_device_eval_batch_size 8 \
    #--logging_strategy steps \
    #--logging_steps 10 \
    #--eval_strategy epoch \
    #--save_strategy epoch \
    #--load_best_model_at_end True \
    #--save_total_limit 3 \
    #--seed 1337