#!/usr/bin/python3
# Usage: ./eval.py ../training_set.tsv prediction.csv

import csv
import sys


def count_corr(gsname, pname):
    with open(gsname, 'r') as f:
        goldstandard = list(csv.reader(f, delimiter='\t'))
    with open(pname, 'r') as f:
        prediction = list(csv.reader(f, delimiter=','))

    corr = 0
    wrong = 0
    for i in range(1, len(goldstandard)):
        if goldstandard[i][2] == prediction[i][1]:
            corr += 1
        else:
            wrong += 1

    return corr, wrong


if __name__ == "__main__":
    corr, wrong = count_corr(sys.argv[1], sys.argv[2])

    print('Accuracy: %.5f (%d)' % (corr / (corr+wrong), corr))
