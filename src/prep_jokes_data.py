#!/usr/bin/env python

import json
import pandas as pd

jokes_fn = "data/external/reddit_jokes.json"
jokes_output = "data/interim/reddit_jokes.txt"

df = pd.read_json(jokes_fn)
df = df[df.score > 20]
df["output"] = "<|startoftext|> " + df["title"] + " " + df["body"] + " <|endoftext|>"
df.output = df.output.str.replace(r"\n", " <newline> ")
output = df.output.to_list()

print(f"Saving {len(df)} jokes")
with open(jokes_output, 'a') as f:
    for out in output:
        f.write(out + "\n")
