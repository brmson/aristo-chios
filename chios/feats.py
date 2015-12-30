""" Feature generator for answers of a question.

Ties in the individual specialized feature generators. """

import numpy as np

# import chios.feats_glove
import chios.feats_solr
import chios.feats_absoccur
import chios.feats_entoccur


class FeatureGenerator:
    def __init__(self, glove_dim, dump_reports=False):
        self.feats = [
            # chios.feats_glove.GloveFeatures(glove_dim),
            chios.feats_solr.SolrFeatures(dump_reports=dump_reports),  # TODO: Configurable enwiki URL
            chios.feats_solr.SolrFeatures(dump_reports=dump_reports, core='ck12'),
            chios.feats_absoccur.AbstractCooccurrenceFeatures(dump_reports=dump_reports),
            chios.feats_entoccur.EntityCooccurrenceFeatures(dump_reports=dump_reports),
        ]

    def score(self, q):
        """ generate four-row, n-column feature matrix for the answers """
        s = np.hstack([f.score(q) for f in self.feats])
        return s

    def labels(self):
        """ output labels for columns in the feature matrix """
        return [l for f in self.feats for l in f.labels()]
