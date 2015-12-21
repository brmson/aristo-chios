""" Feature generator for answers of a question.

Ties in the individual specialized feature generators. """

import numpy as np

# import chios.feats_glove
import chios.feats_solr
import chios.feats_absoccur


class FeatureGenerator:
    def __init__(self, glove_dim):
        # self.feat_glove = chios.feats_glove.GloveFeatures(glove_dim)
        self.feat_solr = chios.feats_solr.SolrFeatures()
        self.feat_absoccur = chios.feats_absoccur.AbstractCooccurrenceFeatures()

    def score(self, q):
        """ generate four-row, n-column feature matrix for the answers """
        s = np.hstack((
            # self.feat_glove.score(q),
            self.feat_solr.score(q),
            self.feat_absoccur.score(q)
        ))
        return s

    def labels(self):
        """ output labels for columns in the feature matrix """
        return (
            # self.feat_glove.labels() +
            self.feat_solr.labels() + self.feat_absoccur.labels())
