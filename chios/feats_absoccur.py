"""
Scorer based on counting token co-occurences in abstracts of named entities
linked from the question or answer.

TODO: tfidf? we could abuse spacy's prob() for idf
"""

import numpy as np
import pysolr
import shelve
from spacy.en import English


nlp = English(parser=False)  # parser=False radically cuts down the load time


class AbstractCooccurrenceFeatures:
    def __init__(self, top_ents=1):
        # TODO: Configurable URL
        self.solr = pysolr.Solr('http://enwiki.ailao.eu:8983/solr/', timeout=10)
        self.countcache = shelve.open('data/abstractcount.cache')
        self.top_ents = top_ents

    def score(self, q):
        qtoks = q.tokens()
        qne = q.ne()
        acounts = np.array([self._count_answer(qtoks, qne, a) for a in q.answers])
        ascores = acounts / (np.max(acounts, axis=0)+0.1)
        return ascores

    def _count_answer(self, qtoks, qne, a):
        """ return (question_entity_counts, answer_entity_counts) tuple """
        atoks = a.tokens()

        qe_counts = 0
        for e in qne[:self.top_ents]:
            qe_counts += self._count_cooccur(e, atoks)
        ae_counts = 0
        for e in a.ne()[:self.top_ents]:
            if e.surface == a.text:  # only if the NE spans the whole answer
                ae_counts += self._count_cooccur(e, qtoks)

        return (qe_counts, ae_counts)

    def _count_cooccur(self, entity, tokens):
        """ count the number of times any of the tokens is within
        the entity abstract

        XXX: use dbpedia for this instead? """
        if entity.pageId is None:
            return 0
        cache_key = str(entity.pageId) + ' || ' + ' '.join(tokens)
        if cache_key in self.countcache:
            return self.countcache[cache_key]

        results = list(self.solr.search('id:%s' % (entity.pageId,), fl='*'))
        if not results:
            return 0
        text = results[0]['text'][1:]  # skip leading \n

        text = text[:text.index('\n')]  # first paragraph
        abstract = nlp(text)

        count = 0
        for t in tokens:
            matches = [atok.text for atok in abstract if atok.text.lower() == t]
            count += len(matches)

        self.countcache[cache_key] = count
        return count

    def labels(self):
        return ['qe_atoks', 'ae_qtoks']
