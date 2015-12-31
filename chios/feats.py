""" Feature generator for answers of a question.

Ties in the individual specialized feature generators. """

import numpy as np

# import chios.feats_glove
import chios.feats_solr
import chios.feats_absoccur
import chios.feats_entoccur


class FeatureGenerator:
    def __init__(self, glove_dim, dump_reports=False, powerset=False):
        """ If dump_reports, generate extra CSV files with feature-specific
        data analysis.  If powerset, generate a feature for each pair of original
        features too (allowing the classifier to act on combinations). """
        self.feats = [
            # chios.feats_glove.GloveFeatures(glove_dim),
            chios.feats_solr.SolrFeatures(dump_reports=dump_reports),  # TODO: Configurable enwiki URL
            chios.feats_solr.SolrFeatures(dump_reports=dump_reports, core='ck12'),
            chios.feats_absoccur.AbstractCooccurrenceFeatures(dump_reports=dump_reports),
            chios.feats_entoccur.EntityCooccurrenceFeatures(dump_reports=dump_reports),
        ]
        self.powerset = powerset

    def score(self, q):
        """ generate four-row, n-column feature matrix for the answers """
        scores = np.hstack([f.score(q) for f in self.feats])
        if self.powerset:
            scores = np.hstack((scores, np.array([s1 * s2 for s1 in scores.T for s2 in scores.T]).T))
        return scores

    def labels(self):
        """ output labels for columns in the feature matrix """
        labels = [l for f in self.feats for l in f.labels()]
        if self.powerset:
            labels += [l1+'*'+l2 for l1 in labels for l2 in labels]
        return labels
