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

def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {}.json is missing'.format(basename))
            sys.exit(1)
