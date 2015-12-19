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

    prf = open('prediction.csv', 'w')
    anf = open('analysis.csv', 'w')
    prcsv = csv.DictWriter(prf, fieldnames=['id', 'correctAnswer'])
    prcsv.writeheader()
    ancsv = csv.DictWriter(anf, fieldnames=['id', 'l', 'c', 'i', 'p', 'question', 'answer'])
    ancsv.writeheader()
    for q in questions:
        s1 = feat_glove.score(q)
        s2 = feat_solr.score(q)
        s = np.hstack((s1, s2))
        p = cfier.predict_proba(s)[:, 1]
        a = p.argmax()
        prcsv.writerow({'id': q.id, 'correctAnswer': 'ABCD'[a]})
        for i in range(4):
            ancsv.writerow({
                'id': q.id,
                'l': 'ABCD'[i],
                'c': '*' if i == q.correct else '.',
                'i': '+' if i == a else '-',
                'p': p[i],
                'question': ' '.join(q.get_question()),
                'answer': ' '.join(q.get_answers()[i])})
