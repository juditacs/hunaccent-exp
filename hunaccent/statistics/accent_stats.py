#-*- coding: utf-8 -*-
from argparse import ArgumentParser
from sys import stdin
from collections import defaultdict
from string import punctuation
import re
import logging

logging.getLogger().setLevel(logging.INFO)


class AccentStatistic(object):

    punct_re = re.compile(r'^[{0}]+$'.format(
        re.escape(punctuation)), re.UNICODE)

    def __init__(self, accent_map=None, filter_punct=False, detect_accent=False, lower=False):
        self.accent_map = accent_map
        self.filter_punct = filter_punct
        self.detect_accent = detect_accent
        self.word_types = set()
        self.token_num = 0
        self.accented_word_num = 0
        self.latinized_forms = defaultdict(set)
        self.accent_set = set(accent_map.iterkeys())
        self.char_freq = defaultdict(int)
        self.lower = lower
        self.word_freq = defaultdict(int)

    def process_lines(self, lines):
        for line in lines:
            word = line.decode('utf8', 'ignore').strip()
            if not word.strip():
                continue
            if self.filter_punct:
                if AccentStatistic.punct_re.match(word):
                    continue
            word = word.lower() if self.lower else word
            self.word_types.add(word)
            self.word_freq[word] += 1
            for c in word:
                self.char_freq[c] += 1
            self.token_num += 1
            if self.token_num % 1000000 == 0:
                logging.info('{} tokens read'.format(self.token_num))
            if set(word) & self.accent_set:
                self.accented_word_num += 1
                latinized = ''.join(self.accent_map.get(c, c) for c in word)
                self.latinized_forms[latinized].add(word)

    @property
    def lexdif(self):
        if not hasattr(self, '_lexdif'):
            diff_form = len(self.word_types)
            for tgt, sources in self.latinized_forms.iteritems():
                diff_form -= (len(sources) - 1)
            self._lexdif = float(len(self.word_types)) / diff_form
        return self._lexdif

    @property
    def accented_ratio(self):
        return float(self.accented_word_num) / self.token_num

    @property
    def ambiguous_token_ratio(self):
        baseline_err = 0
        n = 0
        for k, v in self.ambiguous.iteritems():
            freqs = {k: self.word_freq[k]}
            n += self.word_freq[k]
            for word in v:
                freqs[word] = self.word_freq[word]
                n += self.word_freq[word]
            #print map(lambda x: unicode(x).encode('utf8'), (k, v, n, self.token_num))
            most_freq = max(freqs.iteritems(), key=lambda x: x[1])
            baseline_err += sum(freqs.itervalues()) - most_freq[1]
        return float(n) / self.token_num, 1 - float(baseline_err) / self.token_num

    def print_fancy(self):
        ambig, bs_acc = self.ambiguous_token_ratio
        print('tokens: {0}\n'
              'types: {1}\n'
              'accented ratio: {2}\n'
              'lexdif: {3}\n'
              'ambiguous word type ratio: {4}\n'
              'non-ascii character ratio: {5}\n'
              'ambig words: {6}\n'
              'baseline acc: {7}\n'
              'accent char ratio: {8}\n'
              'type/token: {9}'.format(
                  self.token_num,
                  len(self.word_types),
                  self.accented_ratio,
                  self.lexdif,
                  self.ambiguous_type_ratio,
                  self.non_ascii_ratio,
                  ambig,
                  bs_acc,
                  self.get_accent_char_sum(),
                  len(self.word_types) / float(self.token_num),
              ))

    @property
    def ambiguous(self):
        if not hasattr(self, '_ambig'):
            self._ambig = {}
            for tgt, src in sorted(self.latinized_forms.iteritems()):
                sources = src
                if tgt in self.word_types:
                    sources.add(tgt)
                if len(sources) > 1:
                    self._ambig[tgt] = sources
        return self._ambig

    @property
    def ambiguous_type_ratio(self):
        return float(sum(map(len, self.ambiguous.itervalues()))) / len(self.word_types)

    def get_accent_char_sum(self):
        s = 0
        for c, cnt in self.char_freq.iteritems():
            if c in self.accent_map:
                s += cnt
        return float(s) / sum(self.char_freq.itervalues())

    def print_ambiguous(self):
        for tgt, sources in self.ambiguous.iteritems():
            print(u'{0}\t{1}'.format(tgt, '\t'.join(map(unicode, sources))).encode('utf8'))

    def get_non_ascii_chars(self):
        char_sum = sum(self.char_freq.itervalues())
        for c, cnt in sorted(self.char_freq.iteritems(), key=lambda x: -x[1]):
            if ord(c) > 127:
                yield c, cnt, float(cnt) / char_sum

    @property
    def non_ascii_ratio(self):
        char_sum = sum(self.char_freq.itervalues())
        s = sum(i[1] for i in self.get_non_ascii_chars())
        return float(s) / char_sum

    def print_non_ascii_chars(self):
        print('\n'.join(u'{0}\t{1}\t{2}'.format(c, cnt, r).encode('utf8') for c, cnt, r in self.get_non_ascii_chars()))

    def print_char_stats(self):
        N = float(sum(self.char_freq.itervalues()))
        accents = set(self.accent_set) | set(self.accent_map.itervalues())
        for i, tgt in enumerate(sorted(set(self.accent_map.itervalues()))):
            print(u'{0}  && {1}\\% \\\\'.format(tgt, round(100 * self.char_freq[tgt] / N, 4)).encode('utf8'))
            for src, t in sorted(self.accent_map.iteritems()):
                if t != tgt:
                    continue
                if src == tgt:
                    continue
                print(u'{0}  & {1} & {2}\\% \\\\'.format(src, tgt, round(100 * self.char_freq[src] / N, 4)).encode('utf8'))
            if i < 4:
                print('\\midrule')


def parse_args():
    p = ArgumentParser()
    p.add_argument('--filter-punct', action='store_true', default=False)
    p.add_argument('-l', '--lower', action='store_true', default=False)
    p.add_argument('--char-stats', action='store_true', default=False)
    p.add_argument('-v', '--verbose', action='store_true', default=False)
    p.add_argument('--accents', type=str, default='áaéeíióoöoőoúuüuűu',
                   help='accent mapping')
    p.add_argument('input_file', nargs='*')
    return p.parse_args()


def convert_mapping(str_map):
    mapping = {}
    map_u = str_map.decode('utf8')
    mapping = {
            map_u[i]: map_u[i + 1] for i in xrange(0, len(map_u), 2)
    }
    return mapping


def main():
    args = parse_args()
    accent_map = convert_mapping(args.accents)
    stats = AccentStatistic(accent_map, filter_punct=args.filter_punct, lower=args.lower)
    if args.input_file:
        for fn in args.input_file:
            with open(fn) as f:
                stats.process_lines(f)
    else:
        stats.process_lines(stdin)
    if args.verbose:
        stats.print_ambiguous()
        stats.print_non_ascii_chars()
    stats.print_fancy()
    if args.char_stats:
        stats.print_char_stats()

if __name__ == '__main__':
    main()
