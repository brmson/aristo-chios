"""
Scorer based on detecting co-occurences of question and answer named entities.
"""

import nltk.data
import numpy as np
import pysolr
import shelve
from spacy.en import English


segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
nlp = English(parser=False)  # parser=False radically cuts down the load time


class EntityCooccurrenceFeatures:
    def __init__(self, top_ents=1):
        # TODO: Configurable URL
        self.solr = pysolr.Solr('http://enwiki.ailao.eu:8983/solr/collection1', timeout=10)
        self.countcache = shelve.open('data/coentcount.cache')
        self.top_ents = top_ents

    def score(self, q):
        qtoks = q.tokens()
        qne = q.ne()
        ascores = np.array([self._count_answer(qtoks, qne, a) for a in q.answers])
        return ascores

    def _count_answer(self, qtoks, qne, a):
        """ return (question_entity_counts, answer_entity_counts) tuple """
        if not qne or not a.ne():
            return (0, 0)

        qe_counts = 0
        for e in qne[:self.top_ents]:
            qe_counts += self._count_cooccur(e, a.ne()[0])
        ae_counts = 0
        for e in a.ne()[:self.top_ents]:
            if e.surface == a.text:  # only if the NE spans the whole answer
                ae_counts += self._count_cooccur(e, qne[0])

        return (qe_counts / self.top_ents, ae_counts / self.top_ents)

    def _count_cooccur(self, entity, otherent):
        """ count the number of times any of the tokens is within
        the entity abstract

        XXX: use dbpedia for this instead? """
        if entity.pageId is None:
            return 0
        cache_key = str(entity.pageId) + ' || ' + str(otherent.label)
        if cache_key in self.countcache:
            return self.countcache[cache_key]

        else:
            results = list(self.solr.search('id:%s' % (entity.pageId,), fl='*'))
            if not results:
                return 0
            text = results[0]['text'][1:]  # skip leading \n

            sentences = segmenter.tokenize(text)
            for sent in sentences:
                if entity.surface.lower() not in sent.lower():
                    continue
                if otherent.surface.lower() in sent.lower():
                    print('<<%s>> : %s' % (sent, otherent.surface))
                    self.countcache[cache_key] = 1
                    return 1

        self.countcache[cache_key] = 0
        return 0

    def labels(self):
        return ['q_aecount', 'a_qecount']
