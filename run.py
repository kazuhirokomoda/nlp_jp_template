# -*- coding: utf-8 -*-

# imports
import sys
#import codecs
import MeCab
from urllib.request import urlopen
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# local imports
from normalize_neologd import *

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
ref. http://tyamagu2.xyz/articles/ja_text_classification/

正規化
ref. https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp

StopWord
ref. http://testpy.hatenablog.com/entry/2016/10/05/004949
"""

class WordDivider:
    INDEX_CATEGORY = 0
    INDEX_SUB_CATEGORY = 1
    INDEX_ROOT_FORM = 6
    TARGET_CATEGORIES = ["名詞", "動詞"] #["名詞", "動詞", "形容詞", "副詞", "連体詞", "感動詞"]
    REMOVE_SUB_CATEGORIES = ["非自立"]

    def __init__(self, dictionary="mecabrc", is_normalize=False, remove_stopwords=False):
        self.dictionary = dictionary
        self.tagger = MeCab.Tagger(self.dictionary)
        self.is_normalize = is_normalize
        self.stopwords = []
        self.remove_stopwords = remove_stopwords


    def set_stopwords(self):
        slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
        slothlib_file = urlopen(slothlib_path)
        slothlib_stopwords = [line.decode("utf-8").strip() for line in slothlib_file]
        slothlib_stopwords = [ss for ss in slothlib_stopwords if not ss==u'']
        return slothlib_stopwords


    def extract_words(self, text):
        if not text:
            return []

        if self.remove_stopwords:
            self.stopwords = self.set_stopwords()

        words = []
        # normalize text before MeCab processing
        if self.is_normalize:
            text = normalize_neologd(text)

        node = self.tagger.parseToNode(text)
        while node:
            features = node.feature.split(',')

            if features[self.INDEX_CATEGORY] in self.TARGET_CATEGORIES and features[self.INDEX_SUB_CATEGORY] not in self.REMOVE_SUB_CATEGORIES:
                if features[self.INDEX_ROOT_FORM] == "*":
                    word_to_append = node.surface
                else:
                    # prefer root form
                    word_to_append = features[self.INDEX_ROOT_FORM]

                # remove stopwords
                if not self.remove_stopwords or word_to_append not in self.stopwords:
                    words.append(word_to_append)

            node = node.next

        return words


def make_documents_from_file(file_path):
    documents = []
    line_number = 0
    with open(file_path, encoding='utf-8') as a_file:
        for a_line in a_file:
            line_number += 1
            documents.append(a_line)
    return documents


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

    # 
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
