from string import punctuation
import re


class CorpusReader(object):

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

    def __init__(self, accent_map, filter_punct=False, lower=True, by_word=True):
        self.lower = lower
        self.filter_punct = filter_punct
        self.by_word = by_word
        self.accent_map = CorpusReader.convert_accent_map(accent_map)
        self.accent_set = set(self.accent_map.iterkeys())
        self.accent_chars = set(self.accent_map.itervalues()) | self.accent_set

    def get_featdict_from_lines(self, lines, window=3, symmetric=True):
        vectors = []
        labels = []
        if not self.by_word:
            #TODO fix this
            raise NotImplemented("only by_word parsing supported right now")
        #for line in lines.read():
        line = lines.read().decode('utf8').replace('\n', ' ').encode('utf8')
        #print line
        word = self.get_word_from_line(line)
#        if not word:
#            continue
#        if not set(word) & self.accent_set:
#            continue
        word_nodia = ''.join(self.accent_map.get(c, c) for c in word)
        ngrams = ['BEG'] * window + list(word_nodia) + ['END'] * window
        for i, c in enumerate(word):
            if c in self.accent_chars:
                feats = self.extract_features(ngrams, i + window, window)
                vectors.append(feats)
                labels.append(c)
        return vectors, labels

    def extract_features(self, ngrams, pos, window):
        feats = {}
        for i in xrange(-window, 0):
            feats[i] = ngrams[pos + i]
        for i in xrange(1, window + 1):
            feats[i] = ngrams[pos + i]
        return feats

    def get_word_from_line(self, line):
        word = line.decode('utf8').strip()
        if not word.strip():
            return
        if self.filter_punct:
            if CorpusReader.punct_re.match(word):
                return
        if self.lower:
            word = word.lower()
        return word
