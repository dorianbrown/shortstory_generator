#!/bin/bash

curl https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz | tar -xvzf -
mv writingPrompts/* ../data/external/
rm -rf writingPrompts
