#!/usr/bin/python3
# 1. setup solr according to yodaqa:data/enwiki/README.md
# 2. unzip
# http://www.ck12.org/flx/show/epub/user:anBkcmVjb3VydEBnbWFpbC5jb20./Concepts_b_v8_vdt.epub
# and pass the directory name as parameter to this script
# (the script reimplements the suggestions in
# https://www.kaggle.com/c/the-allen-ai-science-challenge/forums/t/17024/how-to-get-started-with-lucene-wikipedia-benchmark/97124)
# 3. copy ck12 to solr directory as another core and run dataimport

from bs4 import BeautifulSoup, Comment
import sys


id = 0


def process(bs):
    headers = bs.find_all(['h1','h2','h3','h4','h5','h6'])
    for i, h in enumerate(headers):
        h = h.next_element
        title = str(h).strip()
        if title == 'Explore More' or title == 'Example' or title == 'Practice' or title == 'Review':
            continue  # boring

        text = ''
        for e in h.next_elements:
            if i < len(headers)-1 and e == headers[i+1]:
                break  # next heading
            if e.name is not None:
                continue  # tag
            if isinstance(e, Comment):
                continue
            if True in ['img' in p.attrs.get('class', '') for p in e.parents]:
                continue  # image caption
            text2 = str(e).strip()
            if text and text2:
                text += ' '
            text += text2

        if text:
            global id
            print(' <section id="%d">' % (id,))
            print('  <title>%s</title>' % (title.replace('&', '&amp;').replace('<', '&lt;'),))
            print('  <text>%s</text>' % (text.replace('&', '&amp;').replace('<', '&lt;'),))
            print(' </section>')
            id += 1


if __name__ == "__main__":
    htmldir = sys.argv[1]

    print('<ck12>')
    for i in range(1, 125):
        with open('%s/%d.html' % (htmldir, i), 'r') as f:
            process(BeautifulSoup(f.read(), 'lxml'))
    print('</ck12>')
