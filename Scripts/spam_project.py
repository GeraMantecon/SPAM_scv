import os
import sys
import json
import collections
import string
import re
import math
import argparse
from parser import parse_raw_spam
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		verb_lem = [str(self.wnl.lemmatize(t,'v')) for t in word_tokenize(doc)]
		return [str(self.wnl.lemmatize(t,'n')) for t in verb_lem]

def pre_process(stopword=False,punctuation=False,lematize=False):
    with open('../Resources/spam.txt') as f:
        spams = f.readlines()
    spams = [eval(spam)['body'] for spam in spams]
    if stopwords:
        spams = [filter(lambda x: x in set(stopwords.words("english")),spam.lower().strip()) for spam in spams]
    if punctuation:
        spams = [filter(lambda x: x in set(string.punctuation),spam.lower().strip()) for spam in spams]
    if lematize:
        tk = LemmaTokenizer()
        spams = [" ".join(tk(spam.lower().strip())) for spam in spams]
    return spams

def main(argv):
    parser = argparse.ArgumentParser(description='SCV Spam Classification Viaduct')
    parser.add_argument('FEATURE', type=str, help='Desired feature upon which the clustering algorithm is goin to be trained: POS, BOW, BIGRAMS, TRIGRAMS, W2V, NAMED_ENTITIES')
    parser.add_argument('-sw','--stopword', action ='store_true', help='Remove stopwords.')
    parser.add_argument('-p','--punctuation', action ='store_true', help='Remove punctuation marks.')
    parser.add_argument('-l', '--lematize', action ='store_true',help='Lemmatize tokens.')
    args = parser.parse_args()
    if not os.path.isfile('../Resources/spam.txt'):
        parse_raw_spam()
    spam_messages = pre_process(args.stopword, args.punctuation, args.lematize)
    print 'STOPWORDS: '+str(args.stopword)
    print 'PUNCTUATION: '+str(args.punctuation)
    print 'LEMATIZE: '+str(args.lematize)
    with open('test.txt','w') as f:
        for spam in spam_messages:
            f.write(spam+'\n')

if __name__ == '__main__':
    main(sys.argv[1:])
