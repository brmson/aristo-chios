""" Feature generator for answers of a question.

Ties in the individual specialized feature generators. """

import numpy as np

import chios.feats_glove
import chios.feats_solr
import chios.feats_absoccur


class FeatureGenerator:
    def __init__(self, glove_dim):
        self.feat_glove = chios.feats_glove.GloveFeatures(glove_dim)
        self.feat_solr = chios.feats_solr.SolrFeatures()
        self.feat_absoccur = chios.feats_absoccur.AbstractCooccurrenceFeatures()

    def score(self, q):
        """ generate four-row, n-column feature matrix for the answers """
        s1 = self.feat_glove.score(q)
        s2 = self.feat_solr.score(q)
        s3 = self.feat_absoccur.score(q)
        s = np.hstack((s1, s2, s3))
        return s
