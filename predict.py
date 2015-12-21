#!/usr/bin/python3

import argparse
import csv
import joblib
import numpy as np
import sys

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

    print('Initialized.', file=sys.stderr)

    prf = open('prediction.csv', 'w')
    anf = open('analysis.csv', 'w')
    prcsv = csv.DictWriter(prf, fieldnames=['id', 'correctAnswer'])
    prcsv.writeheader()
    ancsv = csv.DictWriter(anf, fieldnames=['id', 'question', 'qNE', 'l', 'c', 'i', 'p', 'answer', 'aNE'])
    ancsv.writeheader()
    for i, q in enumerate(questions):
        print('\rQuestion %d/%d' % (i, len(questions)), file=sys.stderr, end='')
        s1 = feat_glove.score(q)
        s2 = feat_solr.score(q)
        s = np.hstack((s1, s2))
        p = cfier.predict_proba(s)[:, 1]
        choice = p.argmax()
        prcsv.writerow({'id': q.id, 'correctAnswer': 'ABCD'[choice]})
        qne = q.ne()
        for i, a in enumerate(q.answers):
            ancsv.writerow({
                'id': q.id,
                'question': ' '.join(q.tokens()),
                'qNE': '; '.join(['%s(%.3f)' % (ne.label, ne.score) for ne in qne]),
                'l': 'ABCD'[i],
                'c': '*' if i == q.correct else '.',
                'i': '+' if i == choice else '-',
                'p': p[i],
                'answer': ' '.join(a.tokens()),
                'aNE': '; '.join(['%s(%.3f)' % (ne.label, ne.score) for ne in a.ne()]),
            })

    print('', file=sys.stderr)
