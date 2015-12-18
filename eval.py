#!/usr/bin/python3
# Usage: ./eval.py ../training_set.tsv prediction.csv

import csv
import sys

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        goldstandard = list(csv.reader(f, delimiter='\t'))
    with open(sys.argv[2], 'r') as f:
        prediction = list(csv.reader(f, delimiter=','))

    corr = 0
    wrong = 0
    for i in range(1, len(goldstandard)):
        if goldstandard[i][2] == prediction[i][1]:
            corr += 1
        else:
            wrong += 1

    print('Accuracy: %.5f (%d)' % (corr / (corr+wrong), corr))
