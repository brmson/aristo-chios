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
        qvec = self._get_vec(q.tokens())
        avecs = [self._get_vec(a.tokens()) for a in q.answers]
        ascores = np.nan_to_num(np.array([[self._score_answer(qvec, avec)] for avec in avecs]))
        return ascores
        # Alternative: REturn powerset of vectors
        # afeats = np.array([[qvec[i] * avec[j] for i in range(self.N) for j in range(self.N)] for avec in avecs])
        # return np.hstack((ascores[:, np.newaxis], afeats))

    def _get_vec(self, tokens):
        vec = np.zeros(self.N)
        for w in tokens:
            if w in self.glovedict:
                vec += self.glovedict[w]
            # else: print('??? ' + w, file=sys.stderr)
        if tokens:
            vec /= len(tokens)
        return vec

    def _score_answer(self, qvec, avec):
        # cosine distance
        return avec.dot(qvec) / (linalg.norm(qvec) * linalg.norm(avec))
