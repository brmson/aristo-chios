#!/usr/bin/python3

from __future__ import print_function  # for vim and jedi using py2

import argparse
import csv
import joblib
import sys

import chios.question as cq
import chios.feats


def predict_and_dump(questions, featgen, cfier):
    prf = open('prediction.csv', 'w')
    anf = open('analysis.csv', 'w')
    prcsv = csv.DictWriter(prf, fieldnames=['id', 'correctAnswer'])
    prcsv.writeheader()
    ancsv = csv.DictWriter(anf, fieldnames=['id', 'question', 'qNE', 'l', 'c', 'i', 'p', 'answer', 'aNE'])
    ancsv.writeheader()

    for i, q in enumerate(questions):
        print('\rQuestion %d/%d' % (i, len(questions)), file=sys.stderr, end='')

        s = featgen.score(q)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove-dim', type=int, default=50, help='embedding size (50, 100, 200, 300 only)')
    parser.add_argument('TSVFILE', help='questions set')
    args = parser.parse_args()

    questions = cq.load_questions(args.TSVFILE)
    featgen = chios.feats.FeatureGenerator(args.glove_dim)
    cfier = joblib.load('data/model')
    # cfier.coef_ = np.array([[0, 1]])

    print('Initialized.', file=sys.stderr)

    predict_and_dump(questions, featgen, cfier)
