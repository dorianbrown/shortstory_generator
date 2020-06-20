#!/usr/bin/env python

import numpy as np
import re


def load_dataset(dir, prefix):
    with open(f"{dir}/{prefix}.wp_target", "r") as f:
        target = f.readlines()
    with open(f"{dir}/{prefix}.wp_source", "r") as f:
        source = f.readlines()
    return target, source


def main(data_dir):
    prefixes = ['train', 'test', 'valid']
    target, source = [], []

    for prefix in prefixes:
        t, s = load_dataset(data_dir, prefix)
        target.extend(t)
        source.extend(s)

    source = [s.strip().replace("[ WP ]", "[WP]") for s in source]
    # Removing trailing edits from story
    target = [re.sub(r"(?i)edit\s?:[\W\w]+$", "", t) for t in target]
    target = ["<|startoftext|> " + t.strip() + " <|endoftext|>" for t in target]

    output = [s + " " + t for s, t in zip(source, target)]

    lengths = [len(o.split(" ")) for o in output]
    keep_ind = np.where(np.array(lengths) < 1000)[0]
    output = [output[i] for i in keep_ind]

    with open("data/processed/gpt2_input_1.txt", 'w+') as f:
        for line in output[:100_000]:
            f.write(line + "\n")

    with open("data/processed/gpt2_input_2.txt", 'w+') as f:
        for line in output[100_000:200_000]:
            f.write(line + "\n")

    with open("data/processed/gpt2_input_3.txt", 'w+') as f:
        for line in output[200_000:]:
            f.write(line + "\n")

    return output


if __name__ == "__main__":
    main("data/external")
