#!/usr/bin/env python

import numpy as np
import re
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import shutil
import math
import argparse

from gpt_2_simple import encoder


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessing data for GPT2 models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', default='data/external')
    parser.add_argument('--output', default='data/processed')
    parser.add_argument('--maxtokens', type=int, default=1_000)
    parser.add_argument('--filter', type=int, default=800)
    parser.add_argument('--preview', type=int, default=0)
    parser.add_argument('--dataset', default='all')
    parser.add_argument('--desc', type=str, default='')

    return parser.parse_args()


def cleanup_text(text):
    # Do all formatting rules here
    text = re.sub(r"``\s", r'"', text)
    text = re.sub(r"\s''", r'"', text)
    text = re.sub(r"\s([\.\,\!\?])", r"\1", text)
    text = re.sub(r"\s(\w*\'\w)", r"\1", text)
    return text


def load_dataset(dir, prefix):
    with open(f"{dir}/{prefix}.wp_target", "r") as f:
        target = f.readlines()
    with open(f"{dir}/{prefix}.wp_source", "r") as f:
        source = f.readlines()
    return target, source


def prep_csv(prefix):
    target, source = load_dataset(args.input, prefix)

    txt = [s.strip() + " <|endprompt|> " + t.strip() for t, s in zip(target, source)]
    df = pd.DataFrame({"txt": txt})

    # FIXME: Strings like "Sufferin' like" are getting changed to "Sufferin'like"
    regex_lst = [
        [r"^\[\s*[a-zA-Z]{2}\s*\]\s*", ""],  # Remove "[ WP ]" tokens
        [r"(?i)edit\s?:[\W\w]+$", ""],  # Remove trailing "EDIT: ..."
        [r"\s*<newline>\s*", "\n"],  # Replace <newline> with \n
        [r"[\n]{3,}", r"\n\n"],  # Change multiple newlines to single one
        [r"``\s", '"'],  # Turning [`` quote ''] into ["quote"]
        [r"“\s", '"'],  # Turning [`` quote ''] into ["quote"]
        [r"\s''", '"'],  # see above
        [r"\s”", '"'],  # see above
        [r"\s([\.\,\!\?;:])", r"\1"],  # Making punctuation attached to word
        [r"\(.+\)", "\1"],  # Turning "( words )" into "(words)"
        [r"\s([nN]'[tT])", r"\1"],  # First get " n't " part of contraction
        [r"\s'\s?(\w)", r"'\1"],  # Sticking most contractions back together
        [r"\s’\s?(\w)", r"'\1"],  # Catching ’ apostrophes (not ')
        [r"\s([\*\_]+\s)", r"\1"],  # Correct reddit formatting "*word *" => "*word*"
        [r"\n", " <newline> "]
    ]

    print("Running regexes")
    for regex in tqdm(regex_lst):
        df.txt = df.txt.str.replace(*regex)

    df.txt = df.txt.str.strip()

    df.txt = "<|startoftext|> " + df.txt + " <|endoftext|>\n"

    text_length = df.txt.apply(lambda x: len(x.split(" ")))
    df = df[text_length < args.filter]

    def first_1k_tokens(text):
        return " ".join(text.split(" ")[:args.maxtokens])

    df.txt = df.txt.apply(first_1k_tokens)

    if args.preview:
        print("\n==============\n\n".join(df.txt.sample(args.preview).to_list()))

    if args.desc:
        filename = f"{args.output}/{args.desc}_{prefix}.txt"
    else:
        filename = f"{args.output}/{prefix}.txt"

    print(f"Writing to {filename}")
    with open(filename, 'a') as f:
        f.writelines(df.txt.to_list())


def main():
    if args.dataset == 'all':
        prefixes = ['test', 'valid', 'train']
    else:
        prefixes = [args.dataset]

    for prefix in prefixes:
        print(f"Processing {prefix}")
        prep_csv(prefix)


if __name__ == "__main__":
    args = parse_args()
    print(f"Running with args: {args}")
    main()
