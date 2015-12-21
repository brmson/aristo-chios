"""
Scorer based on counting token co-occurences in abstracts of named entities
linked from the question or answer.

TODO: tfidf? we could abuse spacy's prob() for idf
"""

import pysolr
import shelve
import numpy as np


class AbstractCooccurrenceFeatures:
    def __init__(self):
        # TODO: Configurable URL
        self.solr = pysolr.Solr('http://enwiki.ailao.eu:8983/solr/', timeout=10)
        self.countcache = shelve.open('data/abstractcount.cache')

    def score(self, q):
        qtoks = q.tokens()
        qne = q.ne()
        acounts = np.array([self._count_answer(qtoks, qne, a) for a in q.answers])
        ascores = acounts / (np.max(acounts, axis=0)+0.1)
        return ascores

    def _count_answer(self, qtoks, qne, a):
        """ return (question_entity_counts, answer_entity_counts) tuple """
        atoks = a.tokens()

        query = ' '.join(qtoks) + '||' + ' '.join(atoks)
        if query in self.countcache:
            return self.countcache[query]

        qe_counts = 0
        for e in qne:
            qe_counts += self._count_cooccur(e, atoks)
        ae_counts = 0
        for e in a.ne():
            ae_counts += self._count_cooccur(e, qtoks)

        self.countcache[query] = (qe_counts, ae_counts)
        return (qe_counts, ae_counts)

    def _count_cooccur(self, entity, tokens):
        """ count the number of times any of the tokens is within
        the entity abstract

        XXX: use dbpedia for this instead? """
        if entity.pageId is None:
            return 0
        results = list(self.solr.search('id:%s' % (entity.pageId,), fl='*'))
        if not results:
            return 0
        text = results[0]['text'][1:]  # skip leading \n

        text = text[:text.index('\n')]  # first paragraph
        count = 0
        # print('<<%s>> :: %s' % (text, tokens))
        for t in tokens:
            count += text.count(t)
        return count

    def labels(self):
        return ['qe_atoks', 'ae_qtoks']
