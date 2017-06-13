'''
Layer definitions
'''

import json
import cPickle as pkl
import numpy
from collections import OrderedDict

from initializers import *
from theano_util import *

# embedding layer initialization
def param_init_embedding_layer(options, params, n_words, dims, embedding_name, prefix='', suffix=''):
    params[prefix+embedding_name+suffix] = norm_weight(n_words, dims)
    return params

# embedding layer process
def embedding_layer(tparams, ids, factors=None, embedding_name='', prefix='', suffix=''):
    do_reshape = False
    if factors == None:
        if ids.ndim > 1:
            do_reshape = True
            n_timesteps = ids.shape[0]
            n_samples = ids.shape[1]
        emb = tparams[prefix+embedding_name+suffix][ids.flatten()]
    else:
        if ids.ndim > 2:
            do_reshape = True
            n_timesteps = ids.shape[1]
            n_samples = ids.shape[2]
        emb_list = [tparams[prefix+embedding_name+suffix][ids[0].flatten()] ]
        emb = concatenate(emb_list, axis=1)
    if do_reshape:
        emb = emb.reshape((n_timesteps, n_samples, -1))

    return emb

def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    #if nin is None:
        #nin = options['dim_proj']
    #if dim is None:
        #dim = options['dim_proj']

    # embedding to gates transformation weights, bias, (for input x)
    W = numpy.concatenate([norm_weight(nin, dim), norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2*dim, )).astype('float32')

    # recurrent transformation weights for gates (for s_{t-1})
    U = numpy.concatenate([ortho_weight(dim), ortho_weight(dim)], axis=1)
    params[pp(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[pp(prefix, 'Wx')] = Wx
    params[pp(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[pp(prefix, 'Ux')] = Ux

    return params

def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              emb_dropout=None,
              rec_dropout=None,
              truncate_gradient=-1,
              profile=False,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3: # normal case
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

