#!/usr/bin/python3

from __future__ import print_function  # for vim and jedi using py2

import argparse
import joblib
import numpy as np
import sys

import chios.question as cq
import chios.feats


def train_cfier(questions, featgen):
    fvs = []
    labels = []
    for i, q in enumerate(questions):
        print('\rQuestion %d/%d' % (i, len(questions)), file=sys.stderr, end='')
        s = featgen.score(q)
        l = np.array([i == q.correct for i in range(4)])[:, np.newaxis]
        fvs.append(s)
        labels.append(l)
    fvs = np.vstack(tuple(fvs))
    labels = np.vstack(tuple(labels))

    print('', file=sys.stderr)

    from sklearn.linear_model import LogisticRegression
    cfier = LogisticRegression(class_weight='balanced')

    # Slightly better but slower:
    # from sklearn.svm import SVC
    # cfier = SVC(kernel='linear', class_weight='balanced', probability=True)

    cfier.fit(fvs, labels[:, 0])
    return cfier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove-dim', type=int, default=50, help='embedding size (50, 100, 200, 300 only)')
    parser.add_argument('TSVFILE', help='questions set')
    args = parser.parse_args()

    questions = cq.load_questions(args.TSVFILE)
    featgen = chios.feats.FeatureGenerator(args.glove_dim, dump_reports=True)
    print('Initialized.', file=sys.stderr)

    cfier = train_cfier(questions, featgen)

    joblib.dump(cfier, 'data/model', compress=3)
