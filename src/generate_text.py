#!/usr/bin/env python

from gpt_2_simple import generate, start_tf_sess, load_gpt2
import argparse
import re

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Preprocessing data for GPT2 models')
parser.add_argument('--run_name')
parser.add_argument('--prefix')
parser.add_argument('--samples', type=int, default=1)
parser.add_argument('--top_p', type=float, default=0.9)
args = parser.parse_args()

sess = start_tf_sess()
load_gpt2(sess, run_name=args.run_name, checkpoint_dir="checkpoint", multi_gpu=False)

gen_lst = generate(
    sess=sess,
    run_name=args.run_name,
    prefix=args.prefix,
    truncate="<|endoftext|>",
    nsamples = args.samples,
    temperature=1,
    top_p=args.top_p,
    return_as_list=True
)

print("\n\n======\n\n".join(gen_lst))
