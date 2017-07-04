#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy

from multiprocessing import Process, Queue
from Queue import Empty
from util import load_dict, load_config
from compat import fill_options

def translate_model(queue, rqueue, pid, models, options, k, normalize, verbose, nbest, return_alignment, suppress_unk, return_hyp_graph):
    from theano_util import (load_params, init_theano_params)
    from nrg import (build_sampler, gen_sample, init_params)

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    from theano import shared
    trng = RandomStreams(1234)
    use_noise = shared(numpy.float32(0.))

    fs_init = []
    fs_next = []
    print models

    for model, option in zip(models, options):
        # load model parameters and set theano shared variables
        param_list = numpy.load(model).files
        param_list = dict.fromkeys([key for key in param_list if not key.startswith('adam_')], 0)
        #print param_list
        params = load_params(model, param_list)

        # output models in plain texts
        numpy.set_printoptions(threshold='nan')
        #for kk, vv in params.iteritems():
            #print kk
            #print vv
        tparams = init_theano_params(params)

        f_init, f_next = build_sampler(tparams, option, use_noise, trng, return_alignment=return_alignment)

        fs_init.append(f_init)
        fs_next.append(f_next)

    def _translate(seq):
        sample, score, word_probs, alignment, hyp_graph = gen_sample(fs_init, fs_next,
                                   # factors, time-steps, n-sample
                                   numpy.array(seq).T.reshape([len(seq[0]), len(seq), 1]),
                                   trng=trng, k=k, maxlen=200,
                                   stochastic=False, argmax=False, # these two is a kind of search method
                                   return_alignment=return_alignment,
                                   suppress_unk=suppress_unk,
                                   return_hyp_graph=return_hyp_graph)
        if normalize: # length normalization
            lengths = numpy.array([len(s) for s in sample])
            scores = scores / length

        if nbest: # return n-best
            return sample, score, word_probs, alignment, hyp_graph
        else: # return the top best
            sidx = numpy.argmin(score)
            return sample[sidx], score[sidx], word_probs[sidx], alignment[sidx], hyp_graph

    while True:
        req = queue.get()
        if req is None:
            break

        idx, x = req[0], req[1]
        seq = _translate(x)

        rqueue.put((idx, seq))

def main(models, source_file, saveto, save_alignment=None, k=5,
        normalize=False, n_process=5, chr_level=False, verbose=False,
        nbest=False, suppress_unk=False, print_word_probabilities=False, return_hyp_graph=False):
    options = []
    for model in models: # actually, there is only one model
        options.append(load_config(model))
        fill_options(options[-1])

    dictionaries = options[0]['dictionaries']
    dictionaries_source = dictionaries[:-1] # 0 - n-1 are source dictionaries
    dictionary_target = dictionaries[-1]

    # load source dictionaries and invert
    word_dicts = [] # list of word-id mapping
    word_idicts = [] # list of id-word mapping
    for dictionary in dictionaries_source:
        word_dict = load_dict(dictionary)
        if options[0]['n_words_src']:
            for kk, vv in word_dict.items():
                if vv >= options[0]['n_words_src']:
                    del word_dict[kk]

        word_idict = dict()
        for kk, vv in word_dict.iteritems():
            word_idict[vv] = kk
        word_idict[0] = '<eos>'
        word_idict[1] = 'UNK'
        word_dicts.append(word_dict)
        word_idicts.append(word_idict)

    # load target dictionaries and invert
    word_dict_trg = load_dict(dictionary_target)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # create input and output queues for process; note that Queue is used to communicateion between Processes
    queue = Queue()
    rqueue = Queue()
    processes = [None] *n_process
    for pidx in xrange(n_process):
        processes[pidx] = Process(
                target=translate_model,
                args=(queue, rqueue, pidx, models, options, k, normalize, verbose, nbest, save_alignment is not None, suppress_unk, return_hyp_graph))
        processes[pidx].start()

    # put data into queue
    def _send_jobs(f):
        source_sentences = []
        for idx, line in enumerate(f):
            if chr_level: #into single characters
                words = list(line.decode('utf-8').strip())
            else: # into words (separated by spaces)
                words = line.strip().split()

            x = []
            for w in words:
                word = w
                w = [word_dicts[i][f] if f in word_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                x.append(w)
            x += [[0]*options[0]['factors']] # end with "EOS"
            queue.put((idx, x))
            source_sentences.append(words)

        return idx+1, source_sentences

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    # this inner function is used to get translation results
    def _retrieve_jobs(n_samples):
        trans = [None]*n_samples
        out_idx = 0
        for idx in xrange(n_samples):
            resp = None
            while resp is None:
                try:
                    resp = rqueue.get(True, 5)
                except Empty:
                    for midx in xrange(n_process):
                        if not processes[midx].is_alive():
                            # kill all other processes and raise exception if one dies
                            queue.cancel_join_thread()
                            rqueue.cancel_join_thread()
                            for idx in xrange(n_process):
                                processes[idx].terminate()
                                sys.exit(1)
            trans[resp[0]] = resp[1]
            while out_idx < n_samples and trans[out_idx] != None:
                yield trans[out_idx]
                out_idx += 1


    sys.stderr.write('Translating...{0}\n'.format(source_file.name))
    n_samples, source_sentences = _send_jobs(source_file)
    _finish_processes()

    for i, trans in enumerate(_retrieve_jobs(n_samples)):
        print trans
