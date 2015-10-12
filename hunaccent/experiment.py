from sys import stdin
from argparse import ArgumentParser
from sklearn import cross_validation
from ConfigParser import ConfigParser
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from featurize import Featurizer


class Experiment(object):

    def __init__(self, featurizer, pipeline, cv=5):
        self.featurizer = featurizer
        self.pipeline = pipeline
        self.cv = cv
        self.input = None

    def set_input(self, input):
        self.input = input

    def run(self):
        X, y = self.featurizer.featurize(self.input)
        self.X = X
        self.y = y
        print(cross_validation.cross_val_score(self.pipeline, X, y, cv=self.cv))
        for group in self.featurizer.accent_groups.iterkeys():
            filt_x, filt_y = self.featurizer.filter_to_group(group)
            print(group)
            print(cross_validation.cross_val_score(self.pipeline, filt_x, filt_y, cv=self.cv))


class ExperimentHandler(object):

    def __init__(self, config_file):
        self.experiments = {}
        self.featurizers = {}
        self.pipelines = {}
        self.parse_config(config_file)

    def parse_config(self, config_file):
        cfg = ConfigParser()
        cfg.read(config_file)
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
                    cls_kwargs[option] = cfg.get(sec, option)
            standardize = cfg.getboolean(sec, 'standardize')
            ext = []
            if standardize:
                ext.append(('standardize', StandardScaler()))
            if cls_type == 'SVC':
                ext.append(('svm', SVC(**cls_kwargs)))
            elif cls_type == 'logreg':
                ext.append(('logreg', LogisticRegression(**cls_kwargs)))
            p = Pipeline(ext)
            self.pipelines[name] = p

    def read_experiments(self, cfg):
        for sec in cfg.sections():
            if not sec.startswith('experiment'):
                continue
            name = sec[9:]
            featurizer = self.featurizers[cfg.get(sec, 'featurizer')]
            pipeline = self.pipelines[cfg.get(sec, 'pipeline')]
            cv = cfg.getint(sec, 'cv')
            e = Experiment(featurizer, pipeline, cv)
            self.experiments[name] = e

    def set_input(self, input):
        for e in self.experiments.itervalues():
            e.set_input(input)

    def run_all(self):
        for e in self.experiments.itervalues():
            e.run()


def parse_args():
    p = ArgumentParser()
    p.add_argument('-c', '--config', default='../cfg/default.cfg', type=str)
    p.add_argument('input_file', nargs='?', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    e = ExperimentHandler('../cfg/default.cfg')
    if args.input_file:
        with open(args.input_file) as f:
            e.set_input(f)
            e.run_all()
    else:
        e.set_input(stdin)
        e.run_all()

if __name__ == '__main__':
    main()
