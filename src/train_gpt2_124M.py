#!/usr/bin/env python

from gpt_2_simple import finetune, start_tf_sess
from datetime import datetime

train_data = "data/processed/124M_filter500_train.npz"
valid_data = "data/processed/124M_filter500_valid.npz"
run_name = f"124M_filter500_adam_lr0001"

sess = start_tf_sess()

finetune(
    sess=sess,
    dataset=train_data,
    model_name="124M",
    learning_rate=0.0001,
    run_name=run_name,
    restore_from="latest",
    save_every=1000,
    sample_every=1000,
    print_every=100,
    optimizer="adam",
    val_dataset=valid_data,
    val_batch_size=2,
    val_batch_count=40,
    val_every=1000
)
