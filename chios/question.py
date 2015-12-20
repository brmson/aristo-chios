"""
Question representation in Chios.

TODO: We now use both spacy and non-spacy representation of question and
answers.  We may want to throw-away the non-spacy one.
"""

import csv
from nltk.corpus import stopwords
import re
from spacy.en import English

import chios.entlink as entlink


_stops = set(stopwords.words("english"))
nlp = English(parser=False)  # parser=False radically cuts down the load time
entlinker = entlink.EntityLinker()


def _tokenize(s):
    """ Function to convert a document to a sequence of words. """
    # from nltk.tokenize import word_tokenize
    # ...but this might be better for our purposes!

    s = re.sub("[^a-zA-Z]", " ", s)
    return s.lower().split()


class Question:
    def __init__(self, gsrec):
        self.id = gsrec['id']
        self.question_text = gsrec['question']
        self.question = _tokenize(self.question_text)
        self.correct = 'ABCD'.index(gsrec['correctAnswer']) if 'correctAnswer' in gsrec else None
        self.answers_text = [gsrec['answer'+l] for l in ['A', 'B', 'C', 'D']]
        self.answers = [_tokenize(at) for at in self.answers_text]

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

    def spacy_question(self):
        return nlp(self.question_text)

    def spacy_answers(self):
        return [nlp(a) for a in self.answers_text]

    def get_question_ne(self):
        return entlinker.linkText(self.spacy_question())

    def get_answers_ne(self):
        return [entlinker.linkText(sa) for sa in self.spacy_answers()]


def load_questions(filename):
    with open(filename, 'r') as f:
        qlines = csv.DictReader(f, delimiter='\t')
        questions = [Question(qline) for qline in qlines]
    return questions
