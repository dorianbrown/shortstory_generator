#!/usr/bin/env python


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

    output = [s + t for s, t in zip(source, target)]
    
    return output

if __name__ == "__main__":
    main("data/external")
