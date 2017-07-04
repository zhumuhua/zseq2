#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import sys
import os
import unittest
import requests

sys.path.append(os.path.abspath('../src'))
from generator import main as generator

if __name__ == '__main__':
    generator(['models/model.npz'], open('data/valid_src'), open('data/valid_out', 'w'), k=12, normalize=False, n_process=1, suppress_unk=True, print_word_probabilities=False)
