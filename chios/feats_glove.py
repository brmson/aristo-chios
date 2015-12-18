"""
GloVe-based scorer.
"""

import numpy as np
from scipy import linalg


def _get_glove_dict(glovepath):
    """Returns discionary of used words"""
    glovedict = dict()
    with open(glovepath, 'r') as f:
        for line in f:
            l = line.split()
            word = l[0]
            glovedict[word] = np.array(l[1:]).astype(float)
    return glovedict


class GloveFeatures:
    def __init__(self, N):
        self.N = N
        self.glovedict = _get_glove_dict('data/glove.6B.%sd.txt' % (N,))

    def score(self, q):
        qvec = self._get_vec(q.get_question())
        avecs = [self._get_vec(a) for a in q.get_answers()]
        ascores = np.array([self._score_answer(qvec, avec) for avec in avecs])
        return ascores

    def _get_vec(self, tokens):
        vec = np.zeros(self.N)
        for w in tokens:
            if w in self.glovedict:
                vec += self.glovedict[w]
        vec /= linalg.norm(vec)  # XXX just average?
        return vec

    def _score_answer(self, qvec, avec):
        return avec.dot(qvec)
