#!/usr/bin/python3

import argparse
import csv
import joblib
import numpy as np

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

    cfier = joblib.load('data/model')
    # cfier.coef_ = np.array([[0, 1]])

    outf = open('prediction.csv', 'w')
    csv = csv.DictWriter(outf, fieldnames=['id', 'correctAnswer'])
    csv.writeheader()
    for q in questions:
        s1 = feat_glove.score(q)
        s2 = feat_solr.score(q)
        s = np.hstack((s1, s2))
        p = cfier.predict_proba(s)[:, 1]
        a = p.argmax()
        csv.writerow({'id': q.id, 'correctAnswer': 'ABCD'[a]})
