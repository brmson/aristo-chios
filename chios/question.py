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


# TODO: abc for Question and Answer, something like "TextArtifact"


class Question:
    def __init__(self, gsrec):
        self.id = gsrec['id']
        self.answers = [Answer(gsrec['answer'+l]) for l in ['A', 'B', 'C', 'D']]
        if 'correctAnswer' in gsrec:
            self.correct = 'ABCD'.index(gsrec['correctAnswer'])
            self.answers[self.correct].is_correct = True
        else:
            self.correct = None
        self.text = gsrec['question']

        # cache
        self.tokens_ = _tokenize(self.text)
        self.spacy_ = nlp(self.text)

    def tokens(self, no_stopwords=True):
        words = self.tokens_
        if no_stopwords:
            words = [w for w in words if w not in _stops]
        return words

    def spacy(self):
        return self.spacy_

    def ne(self):
        return entlinker.linkText(self.spacy())

    def qaint(self, ans):
        """ Return a sentence that combines the question and answer
        to a single statement based on simple substitution rules """

        # first, transform the answer
        def repl(m):
            return self.answers['ABCD'.index(m.group())].text
        text = re.sub('\b[ABCD]\b', repl, ans.text)

        if '___' in self.text:
            return re.sub('___*', text, self.text)
        elif re.match('.*[^.?!]$', self.text):
            return self.text + ' ' + text
        elif 'Wh' in self.text or 'wh' in self.text:
            # last match
            return re.sub(r'\b\w+h[wW]\b', text[::-1], self.text[::-1], count=1)[::-1]
        elif 'how' in self.text or 'How' in self.text:
            return re.sub(r'\b[hH]ow\b', text + ' is how', self.text)
        else:
            # print('??? %s || %s' % (self.text, text))
            return self.text + ' ' + text


class Answer:
    def __init__(self, text, is_correct=False):
        self.text = text
        self.is_correct = is_correct

        # cache
        self.tokens_ = _tokenize(text)
        self.spacy_ = nlp(text)

    def tokens(self, no_stopwords=True):
        words = self.tokens_
        if no_stopwords:
            words = [w for w in words if w not in _stops]
        return words

    def spacy(self):
        return self.spacy_

    def ne(self):
        return entlinker.linkText(self.spacy())


def load_questions(filename):
    with open(filename, 'r') as f:
        qlines = csv.DictReader(f, delimiter='\t')
        questions = [Question(qline) for qline in qlines]
    return questions
