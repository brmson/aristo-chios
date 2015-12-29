"""
Search-based scorer.
"""

import pysolr
import shelve
import numpy as np


class SolrFeatures:
    def __init__(self, url='http://enwiki.ailao.eu:8983/solr', core='collection1'):
        self.solr = pysolr.Solr(url + '/' + core, timeout=10)
        self.core = core
        self.scorecache = shelve.open('data/solrscore-'+core+'.cache')

    def score(self, q):
        qtoks = q.tokens()
        ascores = np.array([[self._score_answer(qtoks, a.tokens())] for a in q.answers])
        return ascores

    def _score_answer(self, qtoks, atoks):
        query = ' '.join(qtoks + ['+'+t for t in atoks])
        if query in self.scorecache:
            return self.scorecache[query]
        #results = list(self.solr.search(query, fl='*,score', defType='edismax', qf='text^1', pf='text~4^8 text~8^4'))
        results = list(self.solr.search(query, fl='*,score', defType='edismax', qf='text^1', pf='text~4^2'))
        if not results:
            return 0
        score = results[0]['score']
        self.scorecache[query] = score
        return score

    def labels(self):
        return ['solr'+self.core+'0sc']
