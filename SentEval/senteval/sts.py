# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import pandas as pd
import logging

from scipy.stats import spearmanr, pearsonr

from senteval.utils import cosine
from senteval.sick import SICKEval


def save_example_results(params, sys_scores, gs_scores, input1, input2, dataset):    
    input1 = [" ".join(i1) for i1 in input1]
    input2 = [" ".join(i2) for i2 in input2]
    r = [[s, g, i1, i2] for s, g, i1, i2 in zip(sys_scores, gs_scores, input1, input2)]
    df = pd.DataFrame(r, columns = ["sys_scores", "gs_scores", "input1", "input2"])

    model_name = params.model_name.split("/")[-1]
    savedir = "sts-results/{}/{}".format(model_name, params.pooler)
    os.makedirs(savedir, exist_ok=True)
    df.to_csv("{}/{}-{}.tsv".format(savedir, params.current_task, dataset), sep="\t")
        

class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]

            # ValueError
            #sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            #sent2 = np.array([s.split() for s in sent2])[not_empty_idx]

            # Fixed ver (2024). https://github.com/facebookresearch/SentEval/issues/94            
            sent1_lengths = [len(s.split()) for s in sent1]
            max_length = max(sent1_lengths)
            sent1 = np.array([s.split() + [''] * (max_length - len(s.split())) for s in sent1])[not_empty_idx]
            sent2_lengths = [len(s.split()) for s in sent2]
            max_length = max(sent2_lengths)
            sent2 = np.array([s.split() + [''] * (max_length - len(s.split())) for s in sent2])[not_empty_idx]

            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    if 'symbolic' in params['pooler']:
                        n = len(enc2)
                    else:
                        n = enc2.shape[0]
                        
                    for kk in range(n):
                        sys_score = self.similarity(enc1[kk], enc2[kk]) 
                        sys_scores.append(sys_score)
            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)

            save_example_results(params, sys_scores, gs_scores, input1, input2, dataset)
            
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
