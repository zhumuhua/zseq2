#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import argparse

import os
import warnings
import sys

from collections import OrderedDict

from util import *
from layers import *
from training_progress import TrainingProgress
from data_iterator import TextIterator

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

    # word embedding for forward rnn (source)
    emb = embedding_layer(tparams, x, factors=1, suffix='', embedding_name="embedding")

    # encoding for forward rnn
    proj = gru_layer(tparams, emb, options,
                     prefix='encoder',
                     mask=x_mask,
                     emb_dropout=None,
                     rec_dropout=rec_dropout, truncate_gradient=options['encoder_truncate_gradient'],
                     profile=False)



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
