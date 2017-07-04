#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import json
import numpy
import copy
import argparse

import os
import warnings
import sys
import time

import itertools

from subprocess import Popen

from collections import OrderedDict

profile = False

from data_iterator import TextIterator
from training_progress import TrainingProgress
from util import *
from theano_util import *
#from alignment_util import *

from layers import *
from initializers import *
#from optimizers import *
#from metrics.scorer_provider import ScorerProvider

#from domain_interpolation_data_iterator import DomainInterpolatorTextIterator

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params = param_init_embedding_layer(options, params, options['n_words_src'], options['dim_word'], embedding_name='embedding', suffix='')
    params = param_init_embedding_layer(options, params, options['n_words_tgt'], options['dim_word'], embedding_name='embedding', suffix='_dec')

    # encoder: bidirectional RNN
    params = param_init_gru(options, params, prefix='encoder', nin=options['dim_word'], dim=options['dim'])
    params = param_init_gru(options, params, prefix='encoder_r', nin=options['dim_word'], dim=options['dim'])
    return params

# bidirectional RNN encoder: take input x (optionally with mask), and product sequence of context vectors (ctx) ?why product context vectors here?
def build_encoder(tparams, options, trng, use_noise, x_mask=None, sampling=False):
    x = tensor.tensor3('x', dtype='int64')
    x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')

    # for the backward rnn, we just need to invert x
    xr = x[:,::-1]
    if x_mask is None:
        xr_mask = None
    else:
        xr_mask = x_mask[::-1]

    # [0]: factors; [1]: timesteps; [2]: samples (batch?)
    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    # dropout come here
    if options['use_dropout']:
        print 'None Implementation'
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))

    # word embedding for forward rnn (source)
    emb = embedding_layer(tparams, x, factors=options['factors'], suffix='')

    # encoding for forward rnn
    proj = gru_layer(tparams, emb, options,
                     prefix='encoder',
                     mask=x_mask,
                     emb_dropout=emb_dropout,
                     rec_dropout=rec_dropout, truncate_gradient=options['encoder_truncate_gradient'],
                     profile=False)

    # word embedding for backward rnn (source)
    embr = embedding_layer(tparams, xr, suffix='', factors=options['factors'])
    if options['use_dropout']:
        print 'None Implementation'

    projr = gru_layer(tparams, embr, options, prefix='encoder_r', mask=xr_mask,
                      emb_dropout=emb_dropout_r, rec_dropout=rec_dropout_r,
                      truncate_gradient=options['encoder_truncate_gradient'], profile=profile)

    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    return x,ctx


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.)) # well, I cannot understand this

    x_mask = tensor.matrix('x_mask', dtype='float32')
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')

    #print tparams
    x, ctx = build_encoder(tparams, options, trng, use_noise, x_mask, sampling=False)

    # final return
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost

# build a sampler (two functions that necessary for decoding)
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):
    if options['use_dropout'] and options['model_version'] < 0.1:
        print 'None Implementation'
    else:
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    x, ctx = build_encoder(tparams, options, trng, use_noise, x_mask=None, sampling=True)
    n_samples = x.shape[2]

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)

    if options['use_dropout'] and options['model_version'] < 0.1:
        # ctx_mean *=  retain_probability_hidden
        print 'None Implementation'

    init_state = fflayer(tparams, ctx_mean, options, prefix='ff_state', activ='tanh')

    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)


    y = tensor.vector('y_sapmler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    decoder_embedding_suffix = '' if options['tie_encoder_decoder_embeddings']  else '_dec'
    emb = embedding_layer(tparams, y, suffix=decoder_embedding_suffix)

    if options['use_dropout'] and options['model_version'] < 0.1:
        # emb = emb * target_dropout
        print 'None Implementation'

    # if y[i] is -1, then this means it is the first word, should be set to zeros
    emb = tensor.switch(y[:, None] <0, tensor.zeros((1, options['dim_word'])), emb)

    # one step of conditional gru
    proj = gru_cond_layer(tparams, emb, options, prefix='decoder', mask=None, context=ctx, one_step=True, init_state=init_state, emb_dropout=emb_dropout_d, ctx_dropout=ctx_dropout_d,rec_dropout=rec_dropout_d, truncate_gradient=options['decoder_truncate_gradient'], profile=profile)

    # the next hidden state
    next_state = proj[0]

    # the weighted averages of context for this target word y
    ctxs = proj[1]

    # alignment matrix (attention model)
    dec_alphas = proj[2]

    if options['use_dropout'] and options['model_version'] < 0.1:
        print 'None Implementation'
    else:
        next_state_up = next_state

    logit_lstm = fflayer(tparams, next_state_up, options, prefix='ff_logit_lstm', activ='linear')
    logit_prev = fflayer(tparams, emb, options, prefix='ff_logit_prev', activ='linear')
    logit_ctx = fflayer(tparams, ctxs, options, prefix='ff_logit_ctx', activ='linear')

    logit = tensor.tanh(logit_lstm + logit_prev + logit_ctx)

    if options['use_dropout'] and options['model_version'] < 0.1:
        print 'None Implementation'

    logit_W = tparams['Wemb'+decoder_embedding_suffix].T if options['tie_decoder_embeddings'] else None

    logit = fflayer(tparams, logit, options, prefix='ff_logit', activ='linear', W=logit_W)

    # compute softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sapmle from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]

    if return_alignment:
        outs.append(dec_alphas)

    f_next = theano.function(inps, outs, name='f_next', profile=profile)

    return f_init, f_next



# use functions returned by build_sampler to generate translations
# use f_init and f_next iteratively
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30, stochastic=True, argmax=False, return_alignment=False, suppress_unk=False, return_hyp_graph=False):
    if k >1 and argmax:
        assert not stochastic, 'Either sampling or k-best beam search, not both'

    sample = [] # final translation results (word ids )
    sample_score = [] # final translation sequence scores
    sample_word_probs = [] # probability of words
    alignment = []
    hyp_graph = None
    if stochastic:
        print 'None Implementation'
    else:
        live_k = 1 # at initial, only one active hypothesis

    if return_hyp_graph:
        from hypgraph import HypGraph
        hyp_graph = HypGraph()

    dead_k = 0 # number of hyperthesis that already terminated eos

    hyp_samples = [ [] for i in xrange(live_k) ]
    word_probs = [ []  for i in xrange(live_k) ]
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for i in xrange(live_k)]

    # here we consider ensemble decoding, so we have multiple models
    num_models = len(f_init)
    next_state = [None] * num_models
    ctx0 = [None] * num_models
    next_p = [None] * num_models
    dec_alphas = [None] * num_models

    for i in xrange(num_models):
        ret = f_init[i](x) # use ith model to do encoding and initialization
        next_state[i] = numpy.tile( ret[0], (live_k, 1)) # there are live_k active hypothesis
        ctx0[i] = ret[1]
    next_w = -1 * numpy.ones((live_k, )).astype('int64') # BOS indicatoj

    # x is a sequence of word ids, following by eos (0)
    for step in xrange(maxlen):
        print step
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])
            inps = [next_w, ctx, next_state[i]]
            ret = f_next[i](*inps)
            next_p[i], next_w_temp, next_state[i] = ret[0], ret[1], ret[2]
            if return_alignment:
                dec_alphas[i] = ret[3]

            if suppress_unk:
                next_p[i][:,1] = -numpy.inf

        if stochastic:
            if argmax:
                nw = sum(next_p)[0].argmax()
            else:
                print 'None Implementation'
        else:
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]

            # averaging the attention weights across models
            if return_alignment:
                mean_alignment = sum(dec_alphas) / num_models

            voc_size = next_p[0].shape[1]
            trans_indices = ranks_flat / voc_size # from which previous hypothesis
            word_indices  = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_word_probs = []
            new_hyp_states = []

            if return_alignment:
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            # generate new hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti] + [wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])

                if return_alignment:
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    new_hyp_alignment[idx].append(mean_alignment[ti])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if return_hyp_graph:
                    word, history = new_hyp_samples[idx][-1], new_hyp_samples[idx][:-1]
                    score = new_hyp_scores[idx]
                    word_prob = new_word_probs[idx][-1]
                    hyp_graph.add(word, history, word_prob=word_prob, cost=score)
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(copy.copy(new_hyp_samples[idx]))
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(copy.copy(new_hyp_samples[idx]))
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(copy.copy(new_hyp_states[idx]))
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break
            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = [numpy.array(state) for state in zip(*hyp_states)]

    # dump every remaining one
    if not argmax and live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])
            sample_word_probs.append(word_probs[idx])
            if return_alignment:
                alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]



    return sample, sample_score, sample_word_probs, alignment, hyp_graph


def train(dim_word=512, # word vector dimansionality
          dim=1000, # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          max_epochs=5000, # finish training after this number of epochs
          finish_after=200000, # finish training after this number of updates
          n_words_src=None, # source vocab size
          n_words_tgt=None, # target vocab size
          maxlen=100, # maximum length of sentences to be kept
          optimizer='adam',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=10000,
          datasets=['source.not.exists','target.not.exists'],
          valid_datasets=None,
          dictionaries=['source.dict.not.exists','target.dict.not.exists'],
          reload_ = False,
          reload_training_progress=True, # reload training progress (only used if reload_ is True),
          objective='CE', # CE: cross entropy (==maximum likelihood)
          shuffle_each_epoch=True,
          sort_by_length=True,
          maxibatch_size=20 # how many minibatches to load at one time
        ):
    model_options = OrderedDict(sorted(locals().copy().items()))

    #print model_options

    worddicts = [None]*len(dictionaries)
    worddicts_r = [None]*len(dictionaries)

    for ii, dd in enumerate(dictionaries): # for each dictionary, load it
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    #print worddicts[0]
    #print worddicts_r
    #for kk, vv in worddicts[0].iteritems():
        #print kk
        #print vv

    if n_words_src is None:
        n_words_src = len(worddicts[0])
        model_options['n_words_src'] = n_words_src

    if n_words_tgt is None:
        n_words_tgt = len(worddicts[1])
        model_options['n_words_tgt'] = n_words_tgt

    #print model_options

    #initialize training progress
    training_progress = TrainingProgress()
    training_progress.bad_counter = 0
    training_progress.uidx = 0 # update index
    training_progress.eidx = 0 # epoch index
    training_progress.estop = False # early stop?
    training_progress.history_errs = []
    training_progress.domain_interpolation_cur = None # not used at present

    # reload training progress
    training_progress_file = saveto + '.progress.json'

    if reload_ and reload_training_progress and os.path.exists(training_progress_file):
        training_progress.load_from_json(training_progress_file)
        if (training_progress.estop==True) or (training_progress.eidx > max_epochs) or (training_progress.uidx >= finish_after):
            print >> sys.stderr, 'Training is alread complete.'
            return numpy.inf

    #print 'training_progress{}'.format(training_progress)

    # loading data
    train = TextIterator(datasets[0], datasets[1],
                        dictionaries[0], dictionaries[1],
                        n_words_source=n_words_src, n_words_target=n_words_tgt,
                        batch_size=batch_size,
                        maxlen=maxlen,
                        skip_empty=True,
                        shuffle_each_epoch=shuffle_each_epoch,
                        sort_by_length=sort_by_length,
                        maxibatch_size=maxibatch_size)
    if valid_datasets != None and validFreq:
        valid = TextIterator(valid_datasets[0], datasets[1],
                            dictionaries[0], dictionaries[1],
                            n_words_source=n_words_src, n_words_garget=n_words_tgt,
                            batch_size=valid_batch_size,
                            maxlen=maxlen)
    else:
        valid = None

    params = init_params(model_options)

    optimizer_params = {}

    if reload_ and os.path.exists(saveto): # load previous model
        params = load_params(saveto, params) # implementation not finished

        optimizer_params = load_optimizer_params(saveto, optimizer)

    tparams = init_theano_params(params)

    build_model(tparams, model_options)


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

    #print vars(args)

    train(**vars(args))
