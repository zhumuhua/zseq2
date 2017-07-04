'''
Layer definitions
'''

import json
import cPickle as pkl
import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from initializers import *
from theano_util import *

# embedding layer initialization
def param_init_embedding_layer(options, params, n_words, dims, embedding_name, prefix='', suffix=''):
    params[prefix+embedding_name+suffix] = norm_weight(n_words, dims)
    return params

# embedding layer process
def embedding_layer(tparams, ids, factors=None, prefix='', suffix=''):
    do_reshape = False
    if factors == None:
        if ids.ndim > 1:
            do_reshape = True
            n_timesteps = ids.shape[0]
            n_samples = ids.shape[1]
        emb = tparams[prefix+embedding_name(0)+suffix][ids.flatten()]
    else:
        if ids.ndim > 2:
            do_reshape = True
            n_timesteps = ids.shape[1]
            n_samples = ids.shape[2]
        emb_list = [tparams[prefix+embedding_name(factor)+suffix][ids[factor].flatten()] for factor in xrange(factors)]
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

# for encoding
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

    dim = tparams[pp(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1, state_below.shape[0], 1)

    # utilify function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embedding
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]

    state_belowx = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'Wx')]) + tparams[pp(prefix, 'bx')]

    def _step_slice(m_, x_, xx_, h_, U, Ux, rec_droput):
        preact = tensor.dot(h_*rec_dropout[0], U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_*rec_dropout[1], Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        h = u * h + (1. - u) * h
        h = m_[:, None] * h + (1. -m_)[:, None] * h_

        return h

    # scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Ux')],
                   rec_dropout]
    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps = nsteps,
                                truncate_gradient=truncate_gradient,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval



def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', W = None, b = None, **kwargs):
    if W == None:
        W = tparams[pp(prefix, 'W')]
    if b == None:
        b = tparams[pp(prefix, 'b')]

    return eval(activ)(tensor.dot(state_below, W)+b)

def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, emb_dropout=None,
                   rec_dropout=None, ctx_dropout=None,
                   pctx_=None,
                   truncate_gradient=-1,
                   profile=False,
                   **kwargs):
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be ready for one step'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # initial/previous state (for one-step is false case)
    if init_state is None:
        init_state = tensor.alloc(0., n_sample, dim)

    # projected context # seems this pctx_ must be available
    if pctx_ is None:
        pctx_ = tensor.dot(context*ctx_dropout[0], tparams[pp(prefix, 'Wc_att')]) + tparams[pp(prefix, 'b_att')]

    state_belowx = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'Wx')]) + tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:,:,n*dim:(n+1)*dim]
        return _x[:,n*dim:(n+1)*dim]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout, U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx, U_nl, Ux_nl, b_nl, bx_nl):
        # m_  mask
        # x_ state_below_
        # xx_ state_belowx
        # h_ init_state
        preact1 = tensor.dot(h_*rec_dropout[0], U) # U * h
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_**rec_dropout[1], Ux) # Ux * h
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1.-u1)*h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # computation attention
        pstate_ = tensor.dot(h1*rec_dropout[2], W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ = tensor.tanh(pctx__) # tanh(context * W1 + hidden_state * W2)

        alpha = tensor.dot(pctx__*ctx_dropout[1], U_att) + c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))

        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        preact2 = tensor.dot(h1*rec_dropout[3], U_nl) + b_nl
        preact2 += tensor.dot(ctx_*ctx_dropout[2], Wc)
        preact2 += tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1*rec_dropout[4], Ux_nl) + bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_*ctx_dropout[3], Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. -m_)[:, None] * h1

        return h2, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'W_comb_att')],
                   tparams[pp(prefix, 'U_att')],
                   tparams[pp(prefix, 'c_tt')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] + shared_vars))
    else:
        print 'None Implementation'

    return rval
