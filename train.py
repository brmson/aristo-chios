#!/usr/bin/python3

import argparse
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

import chios.question as cq
import chios.feats_glove
import chios.feats_solr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove-dim', type=int, default=50, help='embedding size (50, 100, 200, 300 only)')
    parser.add_argument('TSVFILE', help='questions set')
    args = parser.parse_args()

    questions = cq.load_questions(args.TSVFILE)
    feat_glove = chios.feats_glove.GloveFeatures(args.glove_dim)
    feat_solr = chios.feats_solr.SolrFeatures()

    fvs = []
    labels = []
    for q in questions:
        s1 = feat_glove.score(q)[:, np.newaxis]
        s2 = feat_solr.score(q)[:, np.newaxis]
        s = np.hstack((s1, s2))
        l = np.array([i == q.correct for i in range(4)])[:, np.newaxis]
        fvs.append(s)
        labels.append(l)
    fvs = np.vstack(tuple(fvs))
    labels = np.vstack(tuple(labels))

    cfier = LogisticRegression(class_weight='auto')
    cfier.fit(fvs, labels)
    joblib.dump(cfier, 'data/model', compress=3)
