#!/usr/bin/python3

import argparse
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys

import chios.question as cq
import chios.feats_glove
import chios.feats_solr
import chios.feats_absoccur


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove-dim', type=int, default=50, help='embedding size (50, 100, 200, 300 only)')
    parser.add_argument('TSVFILE', help='questions set')
    args = parser.parse_args()

    questions = cq.load_questions(args.TSVFILE)
    feat_glove = chios.feats_glove.GloveFeatures(args.glove_dim)
    feat_solr = chios.feats_solr.SolrFeatures()
    feat_absoccur = chios.feats_absoccur.AbstractCooccurrenceFeatures()
    print('Initialized.', file=sys.stderr)

    fvs = []
    labels = []
    for i, q in enumerate(questions):
        print('\rQuestion %d/%d' % (i, len(questions)), file=sys.stderr, end='')
        s1 = feat_glove.score(q)
        s2 = feat_solr.score(q)
        s3 = feat_absoccur.score(q)
        s = np.hstack((s1, s2, s3))
        l = np.array([i == q.correct for i in range(4)])[:, np.newaxis]
        fvs.append(s)
        labels.append(l)
    fvs = np.vstack(tuple(fvs))
    labels = np.vstack(tuple(labels))

    print('', file=sys.stderr)
    cfier = LogisticRegression(class_weight='balanced')
    cfier.fit(fvs, labels[:,0])
    joblib.dump(cfier, 'data/model', compress=3)
