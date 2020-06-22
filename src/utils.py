#!/usr/bin/env python

import re


def format_output(string):
    wp, story = string.strip().split("<|startoftext|> ")

    # Do all formatting rules here
    story = re.sub(r"\s?<newline>\s?", "\n", story)
    story = re.sub(r"``\s", r'"', story)
    story = re.sub(r"\s''", r'"', story)
    story = re.sub(r"\s([\.\,\!\?])", r"\1", story)
    story = re.sub(r"\s(\w*\'\w)", r"\1", story)

    return wp + "\n\n---\n\n" + story


if __name__ == "__main__":
    with open("gen/gpt2_gentext_20200620_095539.txt", 'r') as f:
        lines = f.readlines()

    print(format_output(lines[0]))
