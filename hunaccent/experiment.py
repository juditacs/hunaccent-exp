import json
import numpy as np
from sys import stdin
from argparse import ArgumentParser
from sklearn import cross_validation
from ConfigParser import ConfigParser
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
import logging

logging.getLogger().setLevel(logging.INFO)
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

from featurize import Featurizer


class Experiment(object):

    def __init__(self, featurizer, pipeline, cv=5, limit=None, sample_per_class=None, balanced=False):
        self.featurizer = featurizer
        self.pipeline = pipeline
        self.cv = cv
        self.input_ = None
        self.limit = limit
        self.sample_per_class = sample_per_class
        self.balanced = balanced

    def set_input_(self, input_):
        self.input_ = input_

    def run(self):
        X, y = self.featurizer.featurize(self.input_, limit=self.limit)
        logging.info('Featurized')
        self.X = X
        self.y = y
        #self.full_result = list(cross_validation.cross_val_score(self.pipeline['pipeline'], X, y, cv=self.cv))
        #self.full_avg = sum(self.full_result) / float(len(self.full_result))
        self.full_result = None
        self.full_avg = None
        logging.info('Acc: {0}'.format(self.full_avg))
        self.full_size = [X.shape[0], X.shape[1]]
        self.group_results = {}
        self.group_size = {}
        self.group_avg = {}
        for group in self.featurizer.accent_groups.iterkeys():
            logging.info('Group {}'.format(group))
            filt_x, filt_y = self.featurizer.filter_to_group(group, balanced=self.balanced, sample_per_class=self.sample_per_class)
            y_size = {}
            for x in filt_y:
                if not x in y_size:
                    y_size[x] = 0
                y_size[x] += 1
            self.group_size[group] = [filt_x.shape[0], filt_x.shape[1], y_size.items()]
            self.group_results[group] = list(cross_validation.cross_val_score(self.pipeline['pipeline'], filt_x, filt_y, cv=self.cv))
            self.group_avg[group] = sum(self.group_results[group]) / float(len(self.group_results[group]))
            logging.info('Group acc: {0}'.format(self.group_avg[group]))
        logging.info(str(self.group_avg))

    def save_results(self, fh):
        res_d = {}
        for attr in ['cv', 'limit', 'full_result', 'group_results', 'full_size', 'group_size', 'full_avg', 'group_avg', 'sample_per_class']:
            res_d[attr] = getattr(self, attr)
        res_d['featurizer'] = self.featurizer.get_params()
        res_d['pipeline'] = self.pipeline['params']
        fh.write(json.dumps(res_d) + '\n')


class ExperimentHandler(object):

    def __init__(self, config_file):
        self.experiments = {}
        self.featurizers = {}
        self.pipelines = {}
        self.parse_config(config_file)

    def parse_config(self, config_file):
        cfg = ConfigParser()
        cfg.optionxform = str
        cfg.read(config_file)
        self.results_fn = cfg.get('global', 'results')
        self.read_featurizers(cfg)
        self.read_pipelines(cfg)
        self.read_experiments(cfg)

    def read_featurizers(self, cfg):
        for sec in cfg.sections():
            if not sec.startswith('featurizer'):
                continue
            name = sec[11:]
            lower = cfg.getboolean(sec, 'lower')
            filter_punct = cfg.getboolean(sec, 'filter_punct')
            word_only = cfg.getboolean(sec, 'word_only')
            accents = cfg.get(sec, 'accents')
            window = cfg.getint(sec, 'window')
            f = Featurizer(name=name, accent_map=accents, lower=lower, filter_punct=filter_punct, word_only=word_only, window=window)
            self.featurizers[name] = f

    def read_pipelines(self, cfg):
        for sec in cfg.sections():
            if not sec.startswith('pipeline'):
                continue
            name = sec[9:]
            cls_type = cfg.get(sec, 'classifier')
            cls_kwargs = {}
            for option in cfg.options(sec):
                if option.startswith(cls_type):
                    opt = option[len(cls_type) + 1:]
                    val = cfg.get(sec, option)
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                    cls_kwargs[opt] = val
            standardize = cfg.getboolean(sec, 'standardize')
            ext = []
            if standardize:
                ext.append(('standardize', StandardScaler()))
            if cls_type == 'SVC':
                ext.append(('svm', SVC(**cls_kwargs)))
            elif cls_type == 'linearsvc':
                ext.append(('svm', LinearSVC(**cls_kwargs)))
            elif cls_type == 'logreg':
                ext.append(('logreg', LogisticRegression(**cls_kwargs)))
            p = Pipeline(ext)
            self.pipelines[name] = {'pipeline': p}
            self.pipelines[name]['params'] = cls_kwargs
            self.pipelines[name]['params']['classifier'] = cls_type
            self.pipelines[name]['params']['standardize'] = standardize

    def read_experiments(self, cfg):
        self.cfg = cfg
        for sec in cfg.sections():
            if not sec.startswith('experiment'):
                continue
            name = sec[11:]
            featurizer = self.featurizers[cfg.get(sec, 'featurizer')]
            pipeline = self.pipelines[cfg.get(sec, 'pipeline')]
            cv = cfg.getint(sec, 'cv')
            limit = cfg.getint(sec, 'limit')
            try:
                sample_per_class = cfg.getint(sec, 'sample_per_class')
            except:
                sample_per_class = 1000
            e = Experiment(featurizer, pipeline, cv, limit, sample_per_class=sample_per_class)
            self.experiments[name] = e

    def set_input(self, input_):
        for e in self.experiments.itervalues():
            e.set_input_(input_)

    def run_all(self):
        res_fn = open(self.results_fn, 'a+')
        for ename, e in self.experiments.iteritems():
            logging.info('Running experiment: {0}'.format(ename))
            logging.info(str(self.cfg.items('experiment_{0}'.format(ename))))
            e.run()
            e.save_results(res_fn)
        res_fn.close()


def parse_args():
    p = ArgumentParser()
    p.add_argument('-c', '--config', default='../cfg/default.cfg', type=str)
    p.add_argument('input_file', nargs='?', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    e = ExperimentHandler(args.config)
    if args.input_file:
        with open(args.input_file) as f:
            e.set_input(f)
            e.run_all()
    else:
        e.set_input(stdin)
        e.run_all()

if __name__ == '__main__':
    main()
