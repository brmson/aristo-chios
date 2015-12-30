"""
Scorer based on counting token co-occurences in abstracts of named entities
linked from the question or answer.

TODO: tfidf? we could abuse spacy's prob() for idf
"""

import csv
import numpy as np
import pysolr
import shelve
from spacy.en import English


nlp = English(parser=False)  # parser=False radically cuts down the load time


class AbstractCooccurrenceFeatures:
    def __init__(self, dump_reports=False, top_ents=1):
        # TODO: Configurable URL
        self.solr = pysolr.Solr('http://enwiki.ailao.eu:8983/solr/', timeout=10)
        self.countcache = shelve.open('data/abstractcount.cache')
        self.top_ents = top_ents

        self.dump_reports = dump_reports
        self.report = None

    def _report(self, row):
        if not self.dump_reports:
            return
        if self.report is None:
            self.reportf = open('absoccur.csv', 'w')
            self.report = csv.DictWriter(self.reportf, fieldnames=['qId', 'isCorrect', 'qText', 'aText', 'base', 'entity', 'nMatchTokens', 'matchTokens', 'abstract'])
            self.report.writeheader()
        self.report.writerow(row)

    def score(self, q):
        qtoks = q.tokens()
        qne = q.ne()
        ascores = np.array([self._count_answer(qtoks, qne, q, a) for a in q.answers])
        return ascores

    def _count_answer(self, qtoks, qne, q, a):
        """ return (question_entity_counts, answer_entity_counts) tuple """
        atoks = a.tokens()

        qe_counts = 0
        for e in qne[:self.top_ents]:
            qe_counts += self._count_cooccur(e, atoks, {'base': 'question', 'q': q, 'a': a})
        ae_counts = 0
        for e in a.ne()[:self.top_ents]:
            if e.surface == a.text:  # only if the NE spans the whole answer
                ae_counts += self._count_cooccur(e, qtoks, {'base': 'answer', 'q': q, 'a': a})

        return (qe_counts / self.top_ents, ae_counts / self.top_ents)

    def _count_cooccur(self, entity, tokens, report):
        """ count the number of times any of the tokens is within
        the entity abstract

        XXX: use dbpedia for this instead? """
        if not tokens or entity.pageId is None:
            return 0
        cache_key = str(entity.pageId) + ' || ' + ' '.join(tokens)
        if cache_key in self.countcache:
            count = self.countcache[cache_key]

        else:
            results = list(self.solr.search('id:%s' % (entity.pageId,), fl='*'))
            if not results:
                return 0
            text = results[0]['text'][1:]  # skip leading \n

            text = text[:text.index('\n')]  # first paragraph
            abstract = nlp(text)

            count = 0
            for t in tokens:
                matches = [atok for atok in abstract if atok.text.lower() == t]
                if matches:
                    self._report({
                        'qId': report['q'].id,
                        'isCorrect': int(report['a'].is_correct),
                        'qText': report['q'].text,
                        'aText': report['a'].text,
                        'base': report['base'],
                        'entity': entity.surface,
                        'nMatchTokens': len(matches),
                        'matchTokens': '; '.join([t.text for t in matches]),
                        'abstract': abstract,
                    })
                    count += len(matches) / len(abstract)

            self.countcache[cache_key] = count

        return count

    def labels(self):
        return ['qe_atoks', 'ae_qtoks']
