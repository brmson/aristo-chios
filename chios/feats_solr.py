"""
Search-based scorer.
"""

import pysolr
import numpy as np


class SolrFeatures:
    def __init__(self):
        # TODO: Configurable URL
        self.solr = pysolr.Solr('http://enwiki.ailao.eu:8983/solr/', timeout=10)

    def score(self, q):
        qtoks = q.get_question()
        ascores = np.array([self._score_answer(qtoks, atoks) for atoks in q.get_answers()])
        return ascores

    def _score_answer(self, qtoks, atoks):
        query = ' '.join(qtoks + ['+'+t for t in atoks])
        results = list(self.solr.search(query, fl='*,score', defType='edismax', qf='text^1', pf='text~4^2'))
        if not results:
            return 0
        return results[0]['score']
