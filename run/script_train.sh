#!/bin/bash

# used to run training


mkdir -p models

../src/nrg.py \
    --model models/model.npz \
    --datasets data/corpus.src data/corpus.tgt \
    --dictionaries data/vocab.src data/vocab.tgt
