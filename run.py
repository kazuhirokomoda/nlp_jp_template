# -*- coding: utf-8 -*-

# imports
import sys
#import codecs
import MeCab
from urllib.request import urlopen
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# local imports
from word_divider import *

# (Python2) Prevent UnicodeEncodeError/UnicodeDecodeError
#sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
#sys.stdin = codecs.getreader('utf_8')(sys.stdin)

"""
NLP Japanese Template
Start Japanese text analysis now!

Prerequisite:
- python3
- utf-8 text

ref. http://www.nltk.org/book-jp/ch12.html
ref. http://diveintopython3-ja.rdy.jp/files.html
"""

text_path_train_test = 'data/train_test.txt'
#text_path_train = 'data/train.txt'
#text_path_test = 'data/test.txt'

if __name__ == '__main__':

    # Read file and make "documents" as array
    documents = make_documents_from_file(text_path_train_test)
    print('Documents (split by lines): ')
    print(documents)
    print()

    # Make an instance of WordDivider for dividing Japanese text
    wd = WordDivider(is_normalize=True, remove_stopwords=True)

    # Obtain Bag-of-words representation
    tf_vectorizer = CountVectorizer(analyzer=wd.extract_words)
    tf = tf_vectorizer.fit_transform(documents)
    print('Vocabulary and corresponding ID: ')
    print(tf_vectorizer.vocabulary_)
    print()
    print('List of vocabulary: ')
    print(tf_vectorizer.get_feature_names())
    print()
    print('(doc, wordID) count: ')
    print(tf)
    print()
