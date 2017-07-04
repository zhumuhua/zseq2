'''
Theano utility functions (copied from nematus)
'''

import json
import cPickle as pkl
import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# make prefix-appended name
def pp(pp, name):
    return '%s_%s' % (pp, name)

#load parameters of the model
def load_params(path, params, with_prefix=''):
    pp = numpy.load(path)
    new_params = OrderedDict()
    for kk, vv in params.iteritems():
        if kk not in pp:
            continue
        new_params[with_prefix+kk] = pp[kk]
    params.update(new_params)
    return params

# load parameters of the optimizer
def load_optimizer_params(path, optimizer_name):
    params = {}
    return params

# initialize Theano variables according to the initial parameters
# (normal parameters should be converted into Theano variables)
def init_theano_params(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def tanh(x):
    return tensor.tanh(x)

def linear(x):
    return x

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# return name of word embedding for factor i
# special handling of factor 0 for backward compatibility
def embedding_name(i):
    if i == 0:
        return 'Wemb'
    else:
        return 'Wemb' + str(i)
