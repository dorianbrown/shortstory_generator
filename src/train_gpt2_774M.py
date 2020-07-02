#!/usr/bin/env python

from gpt_2_simple import finetune, start_tf_sess

train_data = "data/processed/774M_train.npz"
valid_data = "data/processed/774M_valid.npz"
run_name = "774M_sgd_1000tok_lr001"

sess = start_tf_sess()

finetune(
    sess=sess,
    dataset=train_data,
    model_name="774M",
    learning_rate=0.001,
    run_name=run_name,
    save_every=1000,
    sample_every=1000,
    print_every=100,
    optimizer="sgd",
    val_dataset=valid_data,
    val_batch_size=2,
    val_batch_count=40,
    val_every=1000
)
