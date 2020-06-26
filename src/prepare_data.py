#!/usr/bin/env python

import numpy as np
import re
from tqdm import tqdm


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


def main(data_dir):
    prefixes = ['train', 'test', 'valid']
    target, source = [], []

    for prefix in prefixes:
        t, s = load_dataset(data_dir, prefix)
        target.extend(t)
        source.extend(s)

    source = [s.strip().replace("[ WP ]", "[WP]") for s in source]
    # 
    source = [re.sub(r"^\[\s[A-Z]{2}\s\]", "[WP]", s) for s in source]
    # Cleanup formatting text
    for i in tqdm(range(len(target))):
        target[i] = re.sub(r"(?i)edit\s?:[\W\w]+$", "", target[i])
        target[i] = cleanup_text(target[i])
        target[i] = "<|startoftext|> " + target[i].strip() + " <|endoftext|>"

    output = [s + " " + t for s, t in zip(source, target)]

    lengths = [len(o.split(" ")) for o in output]
    keep_ind = np.where(np.array(lengths) < 500)[0]
    print(f"Generating training data with {len(keep_ind)} data points")
    output = [output[i] for i in keep_ind]

    filename = "gpt2_input_lt400"

    with open(f"data/processed/{filename}_1.txt", 'w+') as f:
        for line in output[:100_000]:
            f.write(line + "\n")

    with open(f"data/processed/{filename}_2.txt", 'w+') as f:
        for line in output[100_000:200_000]:
            f.write(line + "\n")

    with open(f"data/processed/{filename}_3.txt", 'w+') as f:
        for line in output[200_000:]:
            f.write(line + "\n")

    return output


if __name__ == "__main__":
    main("data/external")
