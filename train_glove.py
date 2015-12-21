#!/usr/bin/python3
# Retrain GloVe transformation matrix

from __future__ import print_function

import argparse

import os
import chios.question as cq


def dump_embsel_traindata(questions, outputdir):
    """ generate traindata for consumption by ./std_run.sh -p of
    https://github.com/brmson/Sentence-selection """
    try:
        os.mkdir(outputdir)
    except IOError:
        pass
    for q in questions:
        with open('%s/%s.txt' % (outputdir, q.id), 'w') as f:
            for i, a in enumerate(q.answers):
                print('<Q> ' + ' '.join(q.tokens()), file=f)
                print('%d 0 %s' % (1 if i == q.correct else 0, ' '.join(a.tokens())), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove-dim', type=int, default=50, help='embedding size (50, 100, 200, 300 only)')
    parser.add_argument('TSVFILE', help='questions set')
    parser.add_argument('OUTPUTDIR', help='directory with output train data')
    args = parser.parse_args()

    questions = cq.load_questions(args.TSVFILE)
    dump_embsel_traindata(questions, args.OUTPUTDIR)
