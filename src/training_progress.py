'''
implementation of Training_Progress, copied from nematus
'''

import sys
import json

import util

class TrainingProgress(object):
    def load_from_json(self, file_name):
        self.__dict__.update(util.unicode_to_utf8(json.load(open(file_name, 'rb'))))

    def save_to_json(self, file_name):
        json.dump(self, __dict__, open(file_name, 'wb'), indent=2)
