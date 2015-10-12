from string import punctuation
import re
import numpy as np
from sklearn.feature_extraction import DictVectorizer


class Featurizer(object):

    punct_re = re.compile(r'^[{0}]+$'.format(
        re.escape(punctuation)), re.UNICODE)

    @staticmethod
    def convert_accent_map(map_str):
        if not isinstance(map_str, unicode):
            map_str = map_str.decode('utf8')
        m = {}
        for i in xrange(0, len(map_str), 2):
            m[map_str[i]] = map_str[i + 1]
        return m

    def __init__(self, accent_map, name="", filter_punct=False, lower=True, word_only=True, window=3):
        self.lower = lower
        self.name = name
        self.filter_punct = filter_punct
        self.word_only = word_only
        self.accent_map = Featurizer.convert_accent_map(accent_map)
        self.accent_set = set(self.accent_map.iterkeys())
        self.accent_chars = set(self.accent_map.itervalues()) | self.accent_set
        self.window = window
        self.dictvec = DictVectorizer()
        self.lab_d = {}
        self.lab_i = 0

    @property
    def accent_groups(self):
        if not hasattr(self, '_accent_groups'):
            self._accent_groups = {}
            for src, tgt in self.accent_map.iteritems():
                if not tgt in self._accent_groups:
                    self._accent_groups[tgt] = set()
                    self._accent_groups[tgt].add(tgt)
                self._accent_groups[tgt].add(src)
        return self._accent_groups

    def filter_to_group(self, group):
        filt_x = []
        filt_y = []
        for i, ch in enumerate(self.y):
            if ch in self.accent_groups[group]:
                filt_y.append(self.lab_d[ch])
                filt_x.append(self.X[i])
        return self.dictvec.fit_transform(filt_x).toarray(), np.array(filt_y)

    def convert_labels(self, labels):
        self.lab_d = {}
        self.lab_i = 0
        converted = []
        for l in labels:
            if not l in self.lab_d:
                self.lab_d[l] = self.lab_i
                self.lab_i += 1
            converted.append(self.lab_d[l])
        self.rev_d = dict([(v, k) for k, v in self.lab_d.iteritems()])
        return np.array(converted)  # , rev_d

    def featurize(self, stream):
        if self.word_only:
            vec, labels = self.get_featdict_from_lines(stream)
        else:
            vec, labels = self.get_with_context(stream)
        self.X = vec
        self.y = labels
        return self.dictvec.fit_transform(vec).toarray(), self.convert_labels(labels)

    def get_with_context(self, stream):
        text = u''
        for l in stream:
            fd = l.decode('utf8').split('\t')
            if len(fd) < 2:
                continue
            if 'PUNCT' in fd[1]:
                text += u'{0} '.format(fd[0])
            else:
                text += u' {0}'.format(fd[0])
        return self.extract_ngram_feats(text.strip())

    def extract_ngram_feats(self, text):
        ngrams = ['BEG'] * self.window + list(''.join(self.accent_map.get(c, c) for c in text)) + ['END'] * self.window
        vectors = []
        labels = []
        for i, c in enumerate(text):
            if c in self.accent_chars:
                labels.append(c)
                feats = self.extract_features(ngrams, i + self.window)
                vectors.append(feats)
        return vectors, labels

    def get_featdict_from_lines(self, lines):
        vectors = []
        labels = []
        for line in lines:
            word = self.get_word_from_line(line)
            if not word:
                continue
            if not set(word) & self.accent_set:
                continue
            word_nodia = ''.join(self.accent_map.get(c, c) for c in word)
            ngrams = ['BEG'] * self.window + list(word_nodia) + ['END'] * self.window
            for i, c in enumerate(word):
                if c in self.accent_chars:
                    feats = self.extract_features(ngrams, i + self.window)
                    vectors.append(feats)
                    labels.append(c)
        return vectors, labels

    def extract_features(self, ngrams, pos):
        feats = {}
        for i in xrange(-self.window, 0):
            feats[i] = ngrams[pos + i]
        for i in xrange(1, self.window + 1):
            feats[i] = ngrams[pos + i]
        return feats

    def get_word_from_line(self, line):
        word = line.decode('utf8').strip()
        if not word.strip():
            return
        if self.filter_punct:
            if Featurizer.punct_re.match(word):
                return
        if self.lower:
            word = word.lower()
        return word
