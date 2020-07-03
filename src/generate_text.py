#!/usr/bin/env python

from gpt_2_simple import generate, start_tf_sess
import argparse

parser = argparse.ArgumentParser(description='Preprocessing data for GPT2 models')
parser.add_argument('--run_name')
parser.add_argument('--prefix')
args = parser.parse_args()

sess = start_tf_sess()

generate(
    sess=sess,
    prefix=args.prefix,
    truncate="<|endoftext|>",
    temperature=1,
    top_p=0.9
)
