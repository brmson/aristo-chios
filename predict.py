#!/usr/bin/python3

import argparse
import csv

import chios.question as cq
import chios.feats_glove


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove-dim', type=int, default=100, help='embedding size (50, 100, 200, 300 only)')
    parser.add_argument('TSVFILE', help='questions set')
    args = parser.parse_args()

    questions = cq.load_questions(args.TSVFILE)
    feat_glove = chios.feats_glove.GloveFeatures(args.glove_dim)

    outf = open('prediction.csv', 'w')
    csv = csv.DictWriter(outf, fieldnames=['id', 'correctAnswer'])
    csv.writeheader()
    for q in questions:
        s = feat_glove.score(q)
        a = s.argmax()
        csv.writerow({'id': q.id, 'correctAnswer': 'ABCD'[a]})
