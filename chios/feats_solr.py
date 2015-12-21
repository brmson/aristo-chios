"""
Search-based scorer.
"""

import pysolr
import shelve
import numpy as np


class SolrFeatures:
    def __init__(self):
        # TODO: Configurable URL
        self.solr = pysolr.Solr('http://enwiki.ailao.eu:8983/solr/', timeout=10)
        self.scorecache = shelve.open('data/solrscore.cache')

    def score(self, q):
        qtoks = q.tokens()
        ascores = np.array([[self._score_answer(qtoks, a.tokens())] for a in q.answers])
        return ascores

    def _score_answer(self, qtoks, atoks):
        query = ' '.join(qtoks + ['+'+t for t in atoks])
        if query in self.scorecache:
            return self.scorecache[query]
        results = list(self.solr.search(query, fl='*,score', defType='edismax', qf='text^1', pf='text~4^2'))
        if not results:
            return 0
        score = results[0]['score']
        self.scorecache[query] = score
        return score
