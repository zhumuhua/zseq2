#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import os
import warnings
import sys

def train(dim_word=512, # word vector dimansionality
          dim=1000, # the number of LSTM units
        ):

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--datasets', type=str, required=True, metavar='PATH', nargs=2,
                     help='training corpus (source and target)')
    data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs=2,
                     help='network vocabularies (one for source and one for targe)')
    data.add_argument('--model', type=str, default='model.npz', metavar='PATH', dest='saveto',
                     help='model file name (default: %(default)s)')

    args = parser.parse_args()

    print vars(args)

    train(**vars(args))
