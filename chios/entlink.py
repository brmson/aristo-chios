""" Entity linking, using https://github.com/brmson/label-lookup
(the crosswikis variant) and a super-simplistic aggressive NER
(as we look to link mundane words like "cell", not that much of proper
names and such) """

from collections import namedtuple
import requests
import shelve
import spacy.parts_of_speech as pos


EntLink = namedtuple('EntLink', ['surface', 'label', 'pageId', 'score'])


class EntityLinker:
    def __init__(self, url='http://dbp-labels.ailao.eu:5001/'):
        self.url = url
        self.linkcache = shelve.open('data/linktext.cache')

    def linkText(self, text):
        """ convert a longer text (spacy Doc) to a score-serted list
        of tuples
            (label, pageId, score)
        describing entities linked in the text """
        stext = str(text)
        tokens = list(text)

        if stext in self.linkcache:
            return self.linkcache[stext]

        # NER: We use a simple strategy of checking every (i) noun,
        # (ii) filtered 2,3-grams  (TODO 4-grams?)
        ents = []

        # First, all nouns
        for tok in tokens:
            if tok.pos == pos.NOUN:
                link = self.linkLabel(tok.text)
                if link is not None:
                    ents.append(link)

        # Now, all n-grams
        for n in [2, 3]:
            for tgram in zip(*(tokens[i:] for i in range(n))):
                # admissibility filter, empiric
                if tgram[-1].pos == pos.ADP or tgram[-1].pos == pos.DET or tgram[-1].pos == pos.CONJ or tgram[-1].pos == pos.PUNCT or tgram[-1].lemma_ == 'be':
                    continue
                if (tgram[0].pos == pos.DET and len(tgram) == 2) or tgram[0].pos == pos.CONJ or tgram[0].pos == pos.PUNCT or tgram[0].lemma_ == 'be':
                    continue

                label = ' '.join([t.text for t in tgram])
                link = self.linkLabel(label)
                if link is not None:
                    ents.append(link)

        ents = list(set(ents))  # deduplicate
        ents = sorted(ents, key=lambda l: l.score, reverse=True)
        self.linkcache[stext] = ents
        return ents

    def linkLabel(self, label):
        """ try to resolve label to a wikipedia page, returning
        (label, pageId, score)
        or None if unsuccessful """
        r = requests.get(self.url + 'search/' + requests.utils.quote(label) + '?addPageId=1')
        if r.status_code != 200:
            if r.status_code != 500:  # apparently, 500 is sometimes common, huh
                raise RuntimeError('linkLabel %s %s failed with status %d' % (self.url, label, r.status_code))
            return None
        results = r.json()['results']
        if not results:
            return None

        # typically noise by rare many-gram links
        if results[0]['prob'] == 1.0:
            return None

        # print(label, results[0]['name'], results[0]['pageId'], results[0]['prob'])
        return EntLink(label, results[0]['name'], results[0]['pageId'], results[0]['prob'])
