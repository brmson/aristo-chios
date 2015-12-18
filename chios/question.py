"""
Question representation in Chios.
"""

import csv
from nltk.corpus import stopwords
import re


_stops = set(stopwords.words("english"))


def _tokenize(s):
    """ Function to convert a document to a sequence of words. """
    # from nltk.tokenize import word_tokenize
    # ...but this might be better for our purposes!

    s = re.sub("[^a-zA-Z]", " ", s)
    return s.lower().split()


class Question:
    def __init__(self, gsrec):
        self.id = gsrec['id']
        self.question = _tokenize(gsrec['question'])
        self.correct = 'ABCD'.index(gsrec['correctAnswer']) if 'correctAnswer' in gsrec else None
        self.answers = [_tokenize(gsrec['answer'+l]) for l in ['A', 'B', 'C', 'D']]

    def get_question(self, no_stopwords=True):
        words = self.question
        if no_stopwords:
            words = [w for w in words if w not in _stops]
        return words

    def get_answers(self, no_stopwords=True):
        answers = self.answers
        if no_stopwords:
            answers = [[w for w in words if w not in _stops] for words in answers]
        return answers


def load_questions(filename):
    with open(filename, 'r') as f:
        qlines = csv.DictReader(f, delimiter='\t')
        questions = [Question(qline) for qline in qlines]
    return questions
