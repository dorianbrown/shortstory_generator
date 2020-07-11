#!/bin/bash

export train_file=data/processed/filter500_transf_train.txt
export eval_file=data/processed/filter500_transf_test.txt
export output_dir=tmp

python src/huggingface/run_language_modeling.py \
    --output_dir $output_dir \
    --model_type gpt2 \
    --model_name gpt2 \
    --do_train \
    --train_data_file $train_file \
    --do_eval \
    --eval_data_file $eval_file \
    --evaluate_during_training \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --learning_rate 0.001 \
    --logging_dir $output_dir \
    --logging_steps 1000 \
    --save_steps 1000 \
    --fp16 \
    --fp16_opt_level O1
