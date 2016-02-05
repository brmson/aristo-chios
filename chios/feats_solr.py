"""
Search-based scorer.
"""

import csv
import nltk.data
import pysolr
import shelve
import numpy as np


segmenter = nltk.data.load('tokenizers/punkt/english.pickle')


class SolrFeatures:
    def __init__(self, url='http://enwiki.ailao.eu:8983/solr', core='collection1', dump_reports=False):
        self.solr = pysolr.Solr(url + '/' + core, timeout=10)
        self.core = core
        self.scorecache = shelve.open('data/solrscore-'+core+'.cache')

        self.dump_reports = dump_reports
        self.report = None

    def _report(self, row):
        if not self.dump_reports:
            return
        if self.report is None:
            self.reportf = open('solraoccur-'+self.core+'.csv', 'w')
            self.report = csv.DictWriter(self.reportf, fieldnames=['qId', 'isCorrect', 'qText', 'aText', 'qaText', 'qNE', 'aNE', 'foundCo', 'core', 'score', 'title', 'sent', 'abstract'])
            self.report.writeheader()
        self.report.writerow(row)

    def score(self, q):
        qtoks = q.tokens()
        qne = q.ne()
        ascores = np.array([self._score_answer(qtoks, qne, a, {'q': q, 'a': a}) for a in q.answers])
        return ascores

    def _score_answer(self, qtoks, qne, a, report):
        query = ' '.join(qtoks + ['+'+t for t in a.tokens()])
        if query in self.scorecache:
            return self.scorecache[query]

        # results = list(self.solr.search(query, fl='*,score', defType='edismax', qf='text^1', pf='text~4^8 text~8^4'))
        results = list(self.solr.search(query, fl='*,score', defType='edismax', qf='text^1', pf='text~4^2'))
        if results:
            #print('<<%s>> %s' % (query, ['%s:%.3f' % (r['titleText'], r['score']) for r in results[:5]]))
            score = results[0]['score']
            cooc = int(self._abstract_cooccur(results[0], qne, a.ne(), report))
            if len(results) > 1:
                score += results[1]['score']
                cooc += int(self._abstract_cooccur(results[1], qne, a.ne(), report))
            score = [1, score, cooc]
        else:
            score = [0, 0, 0]
        self.scorecache[query] = score
        return score

    def _abstract_cooccur(self, result, qne, ane, report):
        """ return True if the result abstract contains a sentence that refers
        to both qne and ane """
        if not qne or not ane:
            return False

        foundCo = False
        text = result['text'].strip()  # skip leading \n
        try:
            abstract = text[:text.index('\n')]  # first paragraph
        except ValueError:  # esp. on ck12
            abstract = text
        sentences = segmenter.tokenize(abstract)
        for sent in sentences:
            #print(sent)
            if ane[0].surface.lower() in sent.lower() and \
               qne[0].surface.lower() in sent.lower():
                foundCo = True

            self._report({
                'qId': report['q'].id,
                'isCorrect': int(report['a'].is_correct),
                'qText': report['q'].text,
                'aText': report['a'].text,
                'qaText': report['q'].qaint(report['a']),
                'qNE': qne[0].surface,
                'aNE': ane[0].surface,
                'foundCo': int(foundCo),
                'core': self.core,
                'score': result['score'],
                'title': result.get('titleText', ''),
                'sent': sent,
                #'abstract': abstract,
            })
        return foundCo

    def labels(self):
        return ['solr'+self.core+'i', 'solr'+self.core+'0sc', 'solr'+self.core+'0ao']
