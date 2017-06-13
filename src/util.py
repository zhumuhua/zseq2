'''
Utility functions (copied from nematus)
'''

import sys
import json
import cPickle as pkl

#json loads strings as unicode; when using Python2, conversion is needed
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)

